"""
PU setting: Calibrated OOF probabilities + Reliable Negative (RN) selection
- Data: Feature.csv  (must contain label column 'is_alert')
- Outputs (per seed):
    oof_scores_seed{seed}.csv
    reliable_negatives_seed{seed}.csv
    positives_below_tau_seed{seed}.csv
    reliable_negatives_seed{seed}_conservative.csv
- Multi-seed extras:
    oof_scores_merged.csv
    reliable_negatives_consensus_{mode}.csv
    reliable_negatives_consensus_{mode}_conservative.csv
"""
import math
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import average_precision_score, roc_auc_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# ===================== 可調參數（依需要修改） =====================

INPUT_CSV = "Feature.csv"      # 你的帳戶等級特徵檔
LABEL_COL = "is_alert"         # True=正 (P)，False=未標 (U)
ID_COLS   = ["acct", "ID", "acct_id", "account"]  # 會被當作 ID 的欄位（若存在就丟掉）

# 使用哪些模型家族（至少 RF & LR；SVM 可視資料量開或關）
USE_MODELS = ["RF", "LR", "SVM"]   # 例如 ["RF","LR"] 或 ["RF","LR","SVM"]

# K 折、校準方法與門檻
K_FOLDS     = 5                     # 你指定 5（可改 10）
CALIBRATOR  = "sigmoid"             # 'sigmoid' (Platt) 或 'isotonic'
TAU         = 0.04                  # RN 門檻：各模型 OOF 機率都 < TAU 才納入

# 多種子設定（可多跑幾個增穩）
SEEDS = [495, 2485, 95265]

# RN 共識模式：'intersection' 或 'majority'
RN_CONSENSUS_MODE = "intersection"
RN_MIN_VOTES      = None           # majority 模式最少票數，不給就取 ceil(#runs/2)

# RF / LR / SVM 的超參數（保守安全、抗噪心智）
RF_PARAMS = dict(
    n_estimators=1200,
    max_depth=12,
    min_samples_leaf=8,
    min_samples_split=5,
    max_features="sqrt",
    bootstrap=True,
    max_samples=None,
    criterion="gini",
    class_weight="balanced_subsample",
    n_jobs=-1,
)
LR_PARAMS = dict(
    C=0.1,
    penalty="l2",
    solver="lbfgs",
    max_iter=2000,
    class_weight="balanced",
)
SVM_PARAMS = dict(
    C=0.1,
    loss="squared_hinge",
    penalty="l2",
    dual=True,
    tol=1e-3,
    max_iter=8000,
    class_weight="balanced",
)

# ==== 保守 RN 守門參數（可調） ====
CONSERVATIVE_RN = True
TAU_MEAN_FACTOR = 0.7   # mean_prob < TAU * 0.7
SIGMA_MAX       = 0.02  # 各模型機率標準差上限（意見需一致）
POS_GUARD_PERCENTILE = 20.0  # 正對正最近鄰距離的分位數門檻（距正類需 > 此門檻）

# ==== Ray Tune 調參（保持結構不變，僅前置調參後覆寫 *_PARAMS） ====
USE_RAY_TUNE = True      # 要啟用 Ray Tune 請改 True
TUNE_METRIC  = "roc_auc"       # "ap" = average precision；或 "roc_auc"
TUNE_KFOLDS  = 3          # 調參時用較小 K 提速（不影響主流程的 K_FOLDS）
TUNE_NUM_SAMPLES = 36     # 總試驗數量（調大更準、調小更快）
TUNE_MAX_CONCURRENT_TRIALS = 4  # 最大同時試驗數（依硬體調整）

# ===============================================================


def load_data(input_csv: str, label_col: str, id_cols: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """讀檔、處理欄位，回傳 (X, y, id_series)。"""
    df = pd.read_csv(input_csv)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {input_csv}.")

    # 取得 ID 欄位（若存在）
    id_candidates = [c for c in id_cols if c in df.columns]
    if len(id_candidates) > 0:
        ids = df[id_candidates[0]].astype(str)
    else:
        ids = pd.Series(df.index.astype(str), name="row_id")

    # 準備 X、y
    y = df[label_col].astype(int)  # True/False -> 1/0
    drop_cols = [label_col] + id_candidates
    X = df.drop(columns=drop_cols, errors="ignore").copy()

    # 所有特徵嘗試轉為數值；不可轉的設 NaN（後面 Imputer 會補）
    X = X.apply(pd.to_numeric, errors="coerce")

    return X, y, ids


def build_model_pipelines(random_state: int) -> dict[str, Pipeline]:
    """建立 RF / LR / SVM 的 Pipeline（含 Imputer/Scaler）。"""
    models: dict[str, Pipeline] = {}

    if "RF" in USE_MODELS:
        rf = RandomForestClassifier(random_state=random_state, **RF_PARAMS)
        models["RF"] = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", rf),
        ])

    if "LR" in USE_MODELS:
        lr = LogisticRegression(random_state=random_state, **LR_PARAMS)
        models["LR"] = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lr", lr),
        ])

    if "SVM" in USE_MODELS:
        # LinearSVC 無 predict_proba，但有 decision_function；CalibratedClassifierCV 會幫轉成機率
        svm = LinearSVC(random_state=random_state, **SVM_PARAMS)
        models["SVM"] = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("svm", svm),
        ])

    if len(models) == 0:
        raise ValueError("No models selected. Please set USE_MODELS to include at least one of ['RF','LR','SVM'].")
    return models


def _calibrated_cv_fit_predict(estimator: Pipeline, X_tr, y_tr, X_va, calibrator: str) -> np.ndarray:
    """
    兼容新版/舊版 sklearn 的 CalibratedClassifierCV 參數命名（estimator vs base_estimator）。
    回傳校準後對 X_va 的 P(y=1)。
    """
    try:
        cal = CalibratedClassifierCV(estimator=estimator, method=calibrator, cv=5)
    except TypeError:
        cal = CalibratedClassifierCV(base_estimator=estimator, method=calibrator, cv=5)
    cal.fit(X_tr, y_tr)
    proba = cal.predict_proba(X_va)[:, 1]
    return proba


def oof_calibrated_probs(
    X: pd.DataFrame, y: pd.Series, model_name: str, base_pipeline: Pipeline,
    k_folds: int, calibrator: str, seed: int
) -> np.ndarray:
    """
    以 StratifiedKFold 產生該模型之 OOF 機率（Calibrated）。
    - calibrator: 'sigmoid' (Platt) 或 'isotonic'
    """
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X), dtype=float)

    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        est = clone(base_pipeline)
        proba = _calibrated_cv_fit_predict(est, X_tr, y_tr, X_va, CALIBRATOR)
        oof[va] = proba

        print(f"[seed={seed}] {model_name} | fold {fold}/{k_folds} done.")
    return oof


def run_one_seed(
    X: pd.DataFrame, y: pd.Series, ids: pd.Series,
    seed: int, k_folds: int, calibrator: str, tau: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    跑一次種子：
      - 產每個模型的 OOF 機率
      - 依 tau 在 U（y=0）選 RN（各模型皆 < tau）
      - 依 tau 在 P（y=1）標記 POS_LOW（各模型皆 < tau）
      - 回傳 (oof_df, rn_df, poslow_df)
    """
    model_pipes = build_model_pipelines(random_state=seed)

    # 蒐集 OOF
    oof_dict: dict[str, np.ndarray] = {}
    for name, pipe in model_pipes.items():
        oof_dict[name] = oof_calibrated_probs(
            X=X, y=y, model_name=name, base_pipeline=pipe,
            k_folds=k_folds, calibrator=calibrator, seed=seed
        )

    # 組 OOF DataFrame
    oof_df = pd.DataFrame({"id": ids, "is_alert": y.astype(bool)})
    for name, arr in oof_dict.items():
        oof_df[f"{name}_oof_prob"] = arr

    # ---- 門檻判定：各模型皆 < tau ----
    all_below = np.ones(len(X), dtype=bool)
    for name in oof_dict:
        all_below &= (oof_dict[name] < tau)

    unlabeled_mask = (y.values == 0)           # U
    positive_mask  = (y.values == 1)           # P

    rn_mask      = unlabeled_mask & all_below  # 可靠負樣本
    poslow_mask  = positive_mask  & all_below  # 正樣本但低於 tau（可視為可疑/需複核）

    # 寫入欄位做「標記」
    oof_df["all_models_below_tau"] = all_below
    oof_df["is_reliable_negative"] = rn_mask
    oof_df["is_pos_below_tau"]     = poslow_mask

    # 個別清單
    rn_df = oof_df.loc[rn_mask,   ["id"]].rename(columns={"id": "reliable_negative_id"}).reset_index(drop=True)
    poslow_df = oof_df.loc[poslow_mask, ["id"]].rename(columns={"id": "pos_below_tau_id"}).reset_index(drop=True)

    return oof_df, rn_df, poslow_df


def pick_rn_conservative(
    oof_df: pd.DataFrame,
    y: pd.Series,
    X_features: pd.DataFrame,
    model_cols: list[str],
    tau_rn: float = TAU,
    tau_mean_factor: float = TAU_MEAN_FACTOR,
    sigma_max: float = SIGMA_MAX,
    pos_guard_percentile: float = POS_GUARD_PERCENTILE,
    positive_label: int = 1,
    unlabeled_label: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """
    超保守 RN 挑選：
      - 所有模型 OOF 機率：max_prob < tau_rn
      - 模型一致性：std_prob < sigma_max（意見需一致）
      - 平均也要低：mean_prob < tau_rn * tau_mean_factor
      - 距離正類足夠遠：最近正樣本距離 > 正對正最近鄰距離的 pos_guard_percentile 分位數
    只從 U（y==0）挑；回傳 (rn_df, report_dict)
    """
    tau_mean = tau_rn * tau_mean_factor

    is_u = (y.values == unlabeled_label)
    probs = oof_df[model_cols].to_numpy()
    max_prob = probs.max(axis=1)
    mean_prob = probs.mean(axis=1)
    std_prob = probs.std(axis=1)

    # 1) 機率門檻 + 一致性
    mask_prob = (max_prob < tau_rn) & (std_prob < sigma_max) & (mean_prob < tau_mean)

    # 2) 距離正類守門
    scaler = StandardScaler()
    Z = scaler.fit_transform(X_features.values)

    pos_idx = np.where(y.values == positive_label)[0]
    if len(pos_idx) < 2:
        mask_dist = np.ones(len(y), dtype=bool)
        delta = np.nan
    else:
        # 每點到最近正樣本距離
        nn_pos = NearestNeighbors(n_neighbors=1, n_jobs=-1)
        nn_pos.fit(Z[pos_idx])
        d_to_pos, _ = nn_pos.kneighbors(Z, return_distance=True)
        d_to_pos = d_to_pos.ravel()

        # 正對正最近鄰（k=2 取第 2 個）
        nn_pp = NearestNeighbors(n_neighbors=2, n_jobs=-1)
        nn_pp.fit(Z[pos_idx])
        d_pp, _ = nn_pp.kneighbors(Z[pos_idx], return_distance=True)
        d_pp = d_pp[:, 1]
        delta = np.percentile(d_pp, pos_guard_percentile)

        mask_dist = (d_to_pos > delta)

    rn_mask = is_u & mask_prob & mask_dist
    rn_df = oof_df.loc[rn_mask, ["id"]].copy().rename(columns={"id": "reliable_negative_id"}).reset_index(drop=True)

    report = {
        "U_candidates": int(is_u.sum()),
        "tau_rn": float(tau_rn),
        "tau_mean": float(tau_mean),
        "sigma_max": float(sigma_max),
        "pos_guard_percentile": float(pos_guard_percentile),
        "delta_pospos": float(delta) if not np.isnan(delta) else None,
        "picked_RN": int(rn_mask.sum()),
        "rejection_stats": {
            "prob_gate_fail": int((is_u & ~mask_prob).sum()),
            "dist_gate_fail": int((is_u & mask_prob & ~mask_dist).sum()),
        }
    }
    return rn_df, report


def consensus_from_runs(rn_list: list[pd.DataFrame], mode: str = "intersection", min_votes: int | None = None) -> pd.DataFrame:
    """將多次種子的 RN 清單做交集或多數決，共識輸出為 DataFrame 一欄 reliable_negative_id。"""
    if len(rn_list) == 0:
        return pd.DataFrame(columns=["reliable_negative_id"])

    if mode not in ("intersection", "majority"):
        raise ValueError("mode must be 'intersection' or 'majority'")

    sets = [set(df["reliable_negative_id"].astype(str)) for df in rn_list]

    if mode == "intersection":
        rn_final = set.intersection(*sets) if len(sets) > 1 else sets[0]
    else:
        if min_votes is None:
            min_votes = math.ceil(len(sets) / 2)
        from collections import Counter
        cnt = Counter()
        for s in sets:
            cnt.update(s)
        rn_final = {rid for rid, c in cnt.items() if c >= min_votes}

    return pd.DataFrame(sorted(rn_final), columns=["reliable_negative_id"])


# ===================== Ray Tune：前置調參（可選） =====================

def _maybe_import_ray():
    try:
        import ray
        from ray import tune
        from ray.tune import Tuner
        from ray.tune.schedulers import ASHAScheduler
        return ray, tune, Tuner, ASHAScheduler
    except Exception as e:
        print(f"[RayTune] Not available or failed to import: {e}")
        return None, None, None, None


def _model_from_params(model_name: str, random_state: int, params: dict) -> Pipeline:
    """依照模型名稱與參數，回傳對應的 Pipeline（保持你原 pipeline 結構）。"""
    if model_name == "RF":
        rf = RandomForestClassifier(random_state=random_state, **params)
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("rf", rf)])
    elif model_name == "LR":
        lr = LogisticRegression(random_state=random_state, **params)
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("lr", lr)])
    elif model_name == "SVM":
        svm = LinearSVC(random_state=random_state, **params)
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("svm", svm)])
    else:
        raise ValueError(model_name)


def _tune_space_for(model_name: str):
    """定義每個模型的搜尋空間（穩健、與你設定相容）。"""
    # 用到 tune 的分佈（延後 import）
    from ray import tune
    if model_name == "RF":
        return {
            "n_estimators": tune.randint(400, 1501),
            "max_depth": tune.choice([None, 8, 12, 16, 24]),
            "min_samples_leaf": tune.randint(2, 11),
            "min_samples_split": tune.choice([2, 5, 10, 20]),
            "max_features": tune.choice(["sqrt", "log2", 0.5, 0.8]),
            "bootstrap": True,
            "max_samples": tune.choice([None, 0.7, 0.9]),
            "criterion": tune.choice(["gini", "entropy"]),
            "class_weight": tune.choice(["balanced_subsample", "balanced"]),
            "n_jobs": -1
        }
    elif model_name == "LR":
        return {
            "C": tune.loguniform(1e-3, 1e1),
            "penalty": "l2",
            "solver": tune.choice(["lbfgs", "newton-cg"]),
            "max_iter": tune.choice([1000, 2000, 4000]),
            "class_weight": "balanced"
        }
    elif model_name == "SVM":
        return {
            "C": tune.loguniform(1e-2, 1e1),
            "loss": "squared_hinge",
            "penalty": "l2",
            "dual": True,
            "tol": tune.choice([1e-4, 1e-3]),
            "max_iter": tune.choice([4000, 8000, 12000]),
            "class_weight": "balanced"
        }
    else:
        raise ValueError(model_name)


def _score_oof(y_true: np.ndarray, oof_prob: np.ndarray, metric: str = "ap") -> float:
    if metric == "ap":
        return float(average_precision_score(y_true, oof_prob))
    elif metric == "roc_auc":
        return float(roc_auc_score(y_true, oof_prob))
    else:
        raise ValueError("metric must be 'ap' or 'roc_auc'")


def ray_tune_models(X: pd.DataFrame, y: pd.Series, base_seed: int = 2025) -> dict:
    ray, tune, Tuner, ASHAScheduler = _maybe_import_ray()
    if ray is None:
        print("[RayTune] Skip tuning (Ray not found).")
        return {}

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

    results = {}
    for model_name in USE_MODELS:
        space = _tune_space_for(model_name)

        def trainable(config):
            seed = base_seed
            pipe = _model_from_params(model_name, random_state=seed, params=config)
            skf = StratifiedKFold(n_splits=TUNE_KFOLDS, shuffle=True, random_state=seed)
            oof = np.zeros(len(X), dtype=float)
            for tr, va in skf.split(X, y):
                X_tr, X_va = X.iloc[tr], X.iloc[va]
                y_tr = y.iloc[tr]
                proba = _calibrated_cv_fit_predict(pipe, X_tr, y_tr, X_va, CALIBRATOR)
                oof[va] = proba
            score = _score_oof(y.values, oof, metric=TUNE_METRIC)
            # 也回報次要指標
            ap = average_precision_score(y.values, oof)
            roc = roc_auc_score(y.values, oof)
            from ray.air import session
            session.report({"score": float(score), "ap": float(ap), "roc_auc": float(roc)})

        # ⚠️ 這裡不要再帶 metric/mode，交給 TuneConfig 管理
        scheduler = ASHAScheduler(grace_period=3, reduction_factor=2)

        tuner = Tuner(
            trainable,
            param_space=space,
            tune_config=tune.TuneConfig(
                metric="score",
                mode="max",
                num_samples=TUNE_NUM_SAMPLES,
                scheduler=scheduler,
                max_concurrent_trials=TUNE_MAX_CONCURRENT_TRIALS,
            ),
        )

        print(f"[RayTune] Tuning {model_name} ...")
        result = tuner.fit()
        best = result.get_best_result(metric="score", mode="max")
        best_conf = best.config
        print(f"[RayTune] Best {model_name} score={best.metrics['score']:.5f} | config={best_conf}")
        results[model_name] = best_conf

    try:
        ray.shutdown()
    except Exception:
        pass
    return results


def _apply_tuned_params(tuned: dict):
    """把 Ray 找到的 best config 覆寫回 *_PARAMS（只覆蓋共有 key）。"""
    global RF_PARAMS, LR_PARAMS, SVM_PARAMS
    if "RF" in tuned:
        RF_PARAMS.update({k: v for k, v in tuned["RF"].items() if k in RF_PARAMS})
    if "LR" in tuned:
        LR_PARAMS.update({k: v for k, v in tuned["LR"].items() if k in LR_PARAMS})
    if "SVM" in tuned:
        SVM_PARAMS.update({k: v for k, v in tuned["SVM"].items() if k in SVM_PARAMS})


# ===================== 主流程 =====================

def main():
    print(">>> Loading data...")
    X, y, ids = load_data(INPUT_CSV, LABEL_COL, ID_COLS)
    print(f"Data shape: X={X.shape}, y={y.shape}, positives={int(y.sum())}, unlabeled={int((y==0).sum())}")

    # ----- (可選) Ray Tune 調參：只調整模型內參數；流程其餘不變 -----
    if USE_RAY_TUNE:
        print("\n>>> Ray Tune hyperparameter search (pre-run) ...")
        tuned = ray_tune_models(X, y, base_seed=2025)
        if tuned:
            _apply_tuned_params(tuned)
            print("[RayTune] Applied tuned params:")
            if "RF" in tuned: print("  RF_PARAMS:", RF_PARAMS)
            if "LR" in tuned: print("  LR_PARAMS:", LR_PARAMS)
            if "SVM" in tuned: print("  SVM_PARAMS:", SVM_PARAMS)
        else:
            print("[RayTune] No tuned params applied (Ray unavailable or tuning skipped).")

    all_oof_frames: list[pd.DataFrame] = []
    all_rn_frames:  list[pd.DataFrame] = []
    all_rn_cons_frames: list[pd.DataFrame] = []

    for seed in SEEDS:
        print(f"\n=== RUN seed={seed} | K={K_FOLDS} | calibrator={CALIBRATOR} | tau={TAU} ===")
        oof_df, rn_df, poslow_df = run_one_seed(
            X=X, y=y, ids=ids, seed=seed, k_folds=K_FOLDS, calibrator=CALIBRATOR, tau=TAU
        )

        # 儲存每輪基本結果
        oof_path    = f"oof_scores_seed{seed}.csv"
        rn_path     = f"reliable_negatives_seed{seed}.csv"
        poslow_path = f"positives_below_tau_seed{seed}.csv"
        oof_df.to_csv(oof_path, index=False)
        rn_df.to_csv(rn_path, index=False)
        poslow_df.to_csv(poslow_path, index=False)
        print(f"Saved: {oof_path}")
        print(f"Saved: {rn_path} | RN count: {len(rn_df)}")
        print(f"Saved: {poslow_path} | Pos<tau count: {len(poslow_df)}")

        # ===== 超保守 RN（守門） =====
        if CONSERVATIVE_RN:
            model_cols = [c for c in oof_df.columns if c.endswith("_oof_prob")]
            rn_cons_df, rn_report = pick_rn_conservative(
                oof_df=oof_df,
                y=y,
                X_features=X,                  # 與 oof_df 對齊的特徵（不含 id/is_alert）
                model_cols=model_cols,
                tau_rn=TAU,
                tau_mean_factor=TAU_MEAN_FACTOR,
                sigma_max=SIGMA_MAX,
                pos_guard_percentile=POS_GUARD_PERCENTILE
            )
            rn_cons_path = f"reliable_negatives_seed{seed}_conservative.csv"
            rn_cons_df.to_csv(rn_cons_path, index=False)
            print(f"Saved conservative RN: {rn_cons_path} | count={len(rn_cons_df)}")
            print("RN selection report:", rn_report)
            all_rn_cons_frames.append(rn_cons_df)

        # ===== 累積用於合併與 RN 共識 =====
        oof_seed = oof_df.copy()
        for c in list(oof_seed.columns):
            if c.endswith("_oof_prob"):
                oof_seed.rename(columns={c: f"{c}_s{seed}"}, inplace=True)
        all_oof_frames.append(oof_seed)
        all_rn_frames.append(rn_df)

    # 多種子 → 合併 OOF 檔、做 RN 共識
    if len(SEEDS) > 1:
        print("\n>>> Merging multi-seed OOF and building consensus RN ...")
        # 以第一個為基底，依 ID 合併
        merged = all_oof_frames[0]
        for i in range(1, len(all_oof_frames)):
            merged = merged.merge(all_oof_frames[i], on=["id", "is_alert"], how="inner")
        merged.to_csv("oof_scores_merged.csv", index=False)
        print("Saved: oof_scores_merged.csv")

        rn_consensus = consensus_from_runs(all_rn_frames, mode=RN_CONSENSUS_MODE, min_votes=RN_MIN_VOTES)
        rn_consensus.to_csv(f"reliable_negatives_consensus_{RN_CONSENSUS_MODE}.csv", index=False)
        print(f"Saved: reliable_negatives_consensus_{RN_CONSENSUS_MODE}.csv | RN count: {len(rn_consensus)}")

        if CONSERVATIVE_RN and len(all_rn_cons_frames) == len(SEEDS):
            rn_consensus_c = consensus_from_runs(all_rn_cons_frames, mode=RN_CONSENSUS_MODE, min_votes=RN_MIN_VOTES)
            rn_consensus_c.to_csv(f"reliable_negatives_consensus_{RN_CONSENSUS_MODE}_conservative.csv", index=False)
            print(f"Saved: reliable_negatives_consensus_{RN_CONSENSUS_MODE}_conservative.csv | RN count: {len(rn_consensus_c)}")
    else:
        print("\nSingle-seed run. Consensus file not produced (set multiple SEEDS to enable).")

    print("\nDone.")


if __name__ == "__main__":
    main()