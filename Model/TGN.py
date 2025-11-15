import torch
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True
# torch.autograd.set_detect_anomaly(True)
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastNeighborLoader,
    LastAggregator,
    TimeEncoder
)
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

MEMORY_DIM = 128      # TGN 記憶體維度
NODE_FEAT_DIM = 64    # 節點初始特徵維度 (V4 中未使用, 但保留)
TEMPORAL_DIM = 64     # TGN 內部時間編碼的維度

TGN_LR = 0.0001
TGN_EPOCHS = 50      # (示範用) 現實中可能需要 10-50
TGN_BATCH_SIZE = 4096 # (來自您的代碼)

CLF_LR = 0.001
CLF_EPOCHS = 3000     # (示範用, 增加 epoch)
CLF_BATCH_SIZE = 4096

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def build_edge_attr(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    df = data['txn_amt'].copy().to_frame()
    df = df.assign(txn_date=pd.to_timedelta(data['txn_date'], unit='D') + pd.to_datetime('2024-01-01'))
    data['txn_time'] = pd.to_datetime(data['txn_time'])
    df = df.assign(from_acct=data['from_acct'])
    df = df.assign(to_acct=data['to_acct'])
    df = df.assign(hour=data['txn_time'].dt.hour.astype(int))
    df = df.assign(minute=data['txn_time'].dt.minute.astype(int))
    df = df.assign(is_night=(((df['hour']) >= 23) | (df['hour'] < 6)))
    df = df.assign(weekday=df['txn_date'].dt.weekday)
    df = df.assign(first_digit=(data['txn_amt'].astype(str).str[0].astype(int)))
    df = df.assign(is_self_txn=data['is_self_txn'].map({'Y': 1, 'N': 0, 'UNK': 0}).fillna(0).astype(int))
    df = df.assign(is_foreign=data['currency_type'] != 'TWD')
    df = df.assign(txn_amt=np.log1p(data['txn_amt']))#.astype(int)
    df = df.drop(columns=['txn_date', 'hour', 'minute'])
    df = df.assign(t=data['txn_date'].astype(str).str.zfill(2)+data['txn_time'].dt.hour.astype(str).str.zfill(2)+data['txn_time'].dt.minute.astype(str).str.zfill(2))
    return df

class FocalLoss(torch.nn.Module):
    """
    Focal Loss - 專門用於處理 F1 Score 和極度不平衡數據。
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha # 賦予正樣本的權重
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        pt = torch.exp(-BCE_loss)
        at = self.alpha * targets.float() + (1 - self.alpha) * (1 - targets.float())
        F_loss = at * (1 - pt)**self.gamma * BCE_loss
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        else:
            return F_loss

class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)
    
class LinkPredictor(torch.nn.Module):
    """
    TGN 內部的連結預測器 (Link Predictor)
    (遵循官方範例 `examples/tgn.py`)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_src = torch.nn.Linear(in_channels, in_channels)
        self.lin_dst = torch.nn.Linear(in_channels, in_channels)
        self.lin_final = torch.nn.Linear(in_channels, out_channels)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)

class TGNModel(torch.nn.Module):
    """
    TGN 核心模型 (階段一)
    [V6] 遵循官方 PyG 範例重寫
    """
    def __init__(self, num_nodes, raw_msg_dim, memory_dim, time_dim):
        super(TGNModel, self).__init__()
        self.num_nodes = num_nodes

        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=raw_msg_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            message_module=IdentityMessage(raw_msg_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        )
        
        self.gnn = GraphAttentionEmbedding(
            in_channels=memory_dim,     # 輸入節點狀態 (A 和 B 的記憶)
            out_channels=memory_dim,    # 輸出的訊息維度 (必須匹配)
            msg_dim=raw_msg_dim,      # 原始邊/交易特徵維度
            time_enc=self.memory.time_enc   # TGNMemory 內部的時間編碼維度
        )
         
        self.time_enc = TimeEncoder(time_dim)

        # [V6] 使用官方範例的 LinkPredictor
        self.link_pred = LinkPredictor(in_channels=memory_dim, out_channels=1)

        self.criterion = torch.nn.BCEWithLogitsLoss()

    @torch.no_grad()
    def get_final_memory(self, data_loader):
        self.memory.eval()
        self.memory.reset_state()
        print("滾動 TGN 獲取最終記憶 (Rolling TGN to get final memory)...")
        for batch in tqdm(data_loader, desc="TGN Roll-forward"):
            batch = batch.to(device, non_blocking=True)
            self.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
            
        return self.memory.memory.detach().cpu()

class NodeClassifier(torch.nn.Module):
    """
    節點分類器 (階段二)
    """
    def __init__(self, in_dim, hidden_dim=64, out_dim=1):
        super(NodeClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(in_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_tgn_stage1(model, data, optimizer, device, epoch):
    loader = TemporalDataLoader(
        data,
        batch_size=TGN_BATCH_SIZE,
        neg_sampling_ratio=1.0
    )
    neighbor_loader = LastNeighborLoader(data.num_nodes, size=15, device=device)
    assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)

    model.memory.train()
    model.gnn.train()
    model.link_pred.train()
    model.memory.reset_state()
    neighbor_loader.reset_state()
    
    total_loss = 0
    pbar = tqdm(loader, desc=f"TGN Epoch {epoch+1}/{TGN_EPOCHS}", leave=False)
    
    criterion = torch.nn.BCEWithLogitsLoss().to(device=device)
    data_t, data_msg = data.t.to(device=device, non_blocking=True), data.msg.to(device=device, non_blocking=True)
    
    for batch in pbar:
        optimizer.zero_grad()

        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # z 的形狀是 [n_id.num_nodes, memory_dim]
        z, last_update = model.memory(n_id)
        z = model.gnn(z, last_update, edge_index, data_t[e_id], data_msg[e_id])
        
        pos_out = model.link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
        neg_out = model.link_pred(z[assoc[batch.src]], z[assoc[batch.neg_dst]])
        
        loss = criterion(pos_out.squeeze(), torch.ones_like(pos_out.squeeze(), device=device))
        loss += criterion(neg_out.squeeze(), torch.zeros_like(neg_out.squeeze(), device=device))

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

        with torch.no_grad():
            model.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
            neighbor_loader.insert(batch.src.long(), batch.dst.long())
            
        # 10. (官方範例) 分離記憶體以防止梯度流過整個歷史
        model.memory.detach()
        
    return total_loss / len(loader)

if __name__ == "__main__":
    import time
    start_time = time.time()
    print("start TGN training...")

    alert_acct = pd.read_parquet(r'40_初賽資料_V3 1/初賽資料/acct_alert.parquet', engine='pyarrow')
    all_data = pd.read_parquet(r'40_初賽資料_V3 1/初賽資料/acct_transaction.parquet', engine='pyarrow').sort_values(by='txn_date').reset_index(drop=True)

    all_data = build_edge_attr(all_data)
    msg_df = all_data.drop(columns=['t', 'from_acct', 'to_acct'])
    
    le = LabelEncoder()
    le.fit(pd.concat([all_data['from_acct'], all_data['to_acct']], axis=0).unique())

    src = torch.tensor(le.transform(all_data['from_acct'].values))
    dst = torch.tensor(le.transform(all_data['to_acct'].values))
    t = torch.tensor(all_data['t'].values.astype(int))
    msg = torch.tensor(msg_df.values.astype(float), dtype=torch.float32)
    data_tgn = TemporalData(src=src, dst=dst, t=t, msg=msg).to(device=device)


    num_nodes = len(le.classes_)
    tgn_model = TGNModel(
        num_nodes=num_nodes,
        raw_msg_dim=msg_df.shape[1], # 應為 10
        memory_dim=MEMORY_DIM,
        time_dim=TEMPORAL_DIM
    ).to(device)

    optimizer_tgn = torch.optim.Adam(tgn_model.parameters(), lr=TGN_LR)

    for epoch in range(TGN_EPOCHS):
        loss = train_tgn_stage1(tgn_model, data_tgn, optimizer_tgn, device, epoch)
        torch.cuda.empty_cache()
        print(f"TGN Epoch {epoch+1}/{TGN_EPOCHS}, Link Pred Loss: {loss:.4f}")

    # torch.save(tgn_model.state_dict(), 'tgn_model.pth')

    # read = torch.load('tgn_model.pth')
    # tgn_model.load_state_dict(read)
    torch.cuda.empty_cache()
    # 1. 獲取 TGN 最終記憶 (在 CPU 上)
    rollforward_loader = TemporalDataLoader(data_tgn, batch_size=TGN_BATCH_SIZE * 16)
    with torch.no_grad():
        final_memory = tgn_model.get_final_memory(rollforward_loader)
    
    # [修改] 轉換為 NumPy
    final_memory_np = final_memory.numpy()
    print(f"最終記憶向量已獲取，形狀 (Final memory shape): {final_memory_np.shape}")

    # 2. 準備 (X, y) 數據集 (使用 NumPy)
    print("從所有節點準備 (X, y) 數據集...")
    pos_acct_set = set(alert_acct['acct'].values)
    all_accts_in_encoder = le.classes_

    X_mem = []
    y_labels = []

    for i, acct in enumerate(tqdm(all_accts_in_encoder, desc="準備分類器數據")):
        # 使用索引 i 從 final_memory_np 獲取特徵
        mem = final_memory_np[i]
        label = 1 if acct in pos_acct_set else 0
        X_mem.append(mem)
        y_labels.append(label)

    X = np.stack(X_mem)
    y = np.array(y_labels)

    pd.concat([pd.Series(le.classes_), pd.DataFrame(X), pd.DataFrame(y)], axis=1).to_csv('tgn_output.csv', index=False)
    print(f"\n訓練完成！總耗時: {(time.time() - start_time) / 60:.2f} 分鐘。")
 