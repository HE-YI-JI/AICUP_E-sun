"""
Account-level feature construction from raw transaction data.

This script reads raw transaction records, alert-account labels, and the
competition prediction list, filters to accounts that appear either in the
alert data or in the prediction target set, normalizes the transaction log,
and builds account-level graph and behavioral features.

The final output `Feature.csv` contains one row per account with aggregated
features (outgoing/incoming statistics, concentration measures, channel and
behavioral aggregates) and a binary `is_alert` label. This feature set is
intended as a generic input for downstream modeling pipelines such as
PU-learning or XGBoost-based classifiers, but it is not tied to the exact
parameters used in the final leaderboard submissions.
"""
import pandas as pd
import numpy as np

# Read file
main_df   = pd.read_csv(r'acct_transaction.csv')
alert_pd  = pd.read_csv(r'acct_alert.csv')
predict_pd = pd.read_csv(r'acct_predict.csv')

def ensure_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize and enrich transaction-level records.

    Ensures that required columns exist with safe default values, standardizes
    dtypes, and derives basic flags such as self-transfer, foreign-currency
    usage, and TWD-denominated transaction amounts. The result is a cleaned
    transaction-level DataFrame ready for account-level aggregation.
    """
    out = df.copy()
    required = [
        'from_acct','to_acct',
        'from_acct_type','to_acct_type',
        'is_self_txn',
        'txn_amt','txn_date','txn_time',
        'currency_type','channel_type'
    ]
    for col in required:
        if col not in out:
            if col in ['from_acct_type','to_acct_type']:
                out[col] = 'UNK'
            elif col in ['txn_amt']:
                out[col] = 0.0
            elif col in ['txn_date','txn_time']:
                out[col] = 0
            elif col == 'currency_type':
                out[col] = 'TWD'
            elif col == 'channel_type':
                out[col] = '99'
            elif col == 'is_self_txn':
                out[col] = 'N'

    out = out[required].copy()

    # Dtype conversions
    out['from_acct'] = out['from_acct'].astype(str)
    out['to_acct'] = out['to_acct'].astype(str)
    out['from_acct_type'] = out['from_acct_type'].astype(str)
    out['to_acct_type'] = out['to_acct_type'].astype(str)
    out['currency_type'] = out['currency_type'].astype(str).str.upper()
    out['channel_type'] = out['channel_type'].astype(str)
    out['txn_amt'] = pd.to_numeric(out['txn_amt'], errors='coerce').fillna(0.0)

    # Helper functions
    def map_fraud_level(x):
        try:
            code = int(x)
        except Exception:
            return 3.8
        if code in (1, 3, 4):
            return 4
        elif code in (5, 6, 7, 99):
            return 3
        elif code in (2):
            return 2
    def map_launder_level(x):
        try:
            code = int(x)
        except Exception:
            return 3.2
        if code in (2):
            return 4
        elif code in (1, 3, 4, 6, 99):
            return 3
        elif code in (5, 7):
            return 2
    def map_is_self(x):
        x = str(x).upper().strip()
        return 1.0 if x == 'Y' else (0.0 if x == 'N' else 0.125)

    # Static currency-to-TWD mapping
    twd_per_currency = {
        "AUD": 19.9, "CAD": 22.0, "CHF": 38.5, "CNY": 4.3, "EUR": 35.4, "GBP": 41.5,
        "HKD": 4.0, "JPY": 0.2, "MXN": 1.7, "NZD": 17.4, "SEK": 3.2, "SGD": 23.8,
        "THB": 1.0, "TWD": 1.0, "USD": 30.7, "ZAR": 1.8
    }

    ft = out['from_acct_type'].str.zfill(2)
    tt = out['to_acct_type'].str.zfill(2)
    is_same_type_cnt = ((ft == tt) & (ft == '01')).sum()
    not_same_type_cnt = (ft != tt).sum()

    out['from_is_external'] = (ft == '02').astype(int)
    out['to_is_external'] = (tt == '02').astype(int)

    calib = not_same_type_cnt / (is_same_type_cnt + not_same_type_cnt + 1e-9)
    out['is_external'] = (
        (ft != tt).astype(int) +
        calib * ((ft == tt) & (ft == '02')).astype(int)
    ).astype(float)

    out['channel_code'] = pd.to_numeric(out['channel_type'], errors='coerce')
    out['fraud_level'] = out['channel_code'].apply(map_fraud_level)
    out['launder_level'] = out['channel_code'].apply(map_launder_level)
    out['is_self'] = out['is_self_txn'].apply(map_is_self)
    out['is_foreign_currency'] = (out['currency_type'] != 'TWD').astype(int)
    out['currency_time'] = out['currency_type'].map(twd_per_currency).fillna(1.0)
    out['txn_amt_TWD'] = out['txn_amt'] * out['currency_time']

    return out

def _hhi_from_counts(arr: np.ndarray) -> float:
    """Herfindahl-Hirschman Index for a distribution of counts."""
    s = float(arr.sum()) + 1e-9
    if s <= 0:
        return 0.0
    p = arr / s
    return float((p ** 2).sum())

def build_graph_features(tx: pd.DataFrame) -> pd.DataFrame:
    """
    Build account-level graph features from transaction records.

    Aggregates outgoing and incoming transaction statistics per account,
    including degree, amount sums/means/maxima, partner counts, concentration
    indices (HHI), one-shot partners, reciprocity-based measures, and derived
    ratios such as in/out flow and net flow. Returns a DataFrame with one row
    per account.
    """
    # Basic out aggregations
    g_out = tx.groupby('from_acct').agg(
        out_deg=('to_acct', 'count'),
        out_sum=('txn_amt_TWD', 'sum'),
        out_mean=('txn_amt_TWD', 'mean'),
        out_max=('txn_amt_TWD', 'max'),
        out_unique_dst=('to_acct', 'nunique'),
        out_ext_cnt=('to_is_external', 'sum')
    ).reset_index().rename(columns={'from_acct': 'acct'})

    # Basic in aggregations
    g_in = tx.groupby('to_acct').agg(
        in_deg=('from_acct', 'count'),
        in_sum=('txn_amt_TWD', 'sum'),
        in_mean=('txn_amt_TWD', 'mean'),
        in_max=('txn_amt_TWD', 'max'),
        in_unique_src=('from_acct', 'nunique'),
        in_ext_cnt=('from_is_external', 'sum')
    ).reset_index().rename(columns={'to_acct': 'acct'})

    # Remove self-transfers for pair-level stats
    not_self_tx = tx[tx['is_self_txn'].astype(str).str.upper() != 'Y'].copy()

    # Directional pair tables
    pair_out = not_self_tx.groupby(['from_acct', 'to_acct']).agg(
        pair_cnt=('txn_amt_TWD', 'count'),
        pair_sum=('txn_amt_TWD', 'sum')
    ).reset_index()
    pair_in = not_self_tx.groupby(['to_acct', 'from_acct']).agg(
        pair_cnt=('txn_amt_TWD', 'count'),
        pair_sum=('txn_amt_TWD', 'sum')
    ).reset_index()

    # Outgoing concentration & partner stats
    if not pair_out.empty:
        idx_cnt = pair_out.groupby('from_acct')['pair_cnt'].idxmax()
        top_out_cnt = pair_out.loc[idx_cnt, ['from_acct', 'pair_cnt']] \
                              .rename(columns={'from_acct': 'acct', 'pair_cnt': 'top_out_cnt'})
        idx_sum = pair_out.groupby('from_acct')['pair_sum'].idxmax()
        top_out_sum = pair_out.loc[idx_sum, ['from_acct', 'pair_sum']] \
                              .rename(columns={'from_acct': 'acct', 'pair_sum': 'top_out_sum'})

        grp = pair_out.groupby('from_acct')
        out_stats = grp.agg(
            partners_out=('pair_cnt', 'size'),
            total_out_cnt=('pair_cnt', 'sum'),
            total_out_sum=('pair_sum', 'sum')
        ).reset_index().rename(columns={'from_acct': 'acct'})

        hhi_cnt = grp['pair_cnt'].apply(lambda s: _hhi_from_counts(s.values)).reset_index().rename(
            columns={'from_acct': 'acct', 'pair_cnt': 'hhi_out_cnt'}
        )
        hhi_sum = grp['pair_sum'].apply(lambda s: _hhi_from_counts(s.values)).reset_index().rename(
            columns={'from_acct': 'acct', 'pair_sum': 'hhi_out_sum'}
        )
        one_shot = grp['pair_cnt'].apply(lambda s: int((s.values == 1).sum())).reset_index().rename(
            columns={'from_acct': 'acct', 'pair_cnt': 'one_shot_out'}
        )

        g_out = (g_out.merge(out_stats, on='acct', how='left')
                      .merge(top_out_cnt, on='acct', how='left')
                      .merge(top_out_sum, on='acct', how='left')
                      .merge(hhi_cnt, on='acct', how='left')
                      .merge(hhi_sum, on='acct', how='left')
                      .merge(one_shot, on='acct', how='left'))
        g_out['top_out_cnt_ratio'] = g_out['top_out_cnt'] / (g_out['out_deg'] + 1e-6)
        g_out['top_out_sum_ratio'] = g_out['top_out_sum'] / (g_out['out_sum'] + 1e-6)
    else:
        g_out = g_out.assign(
            partners_out=0, total_out_cnt=0, total_out_sum=0.0,
            top_out_cnt=0, top_out_sum=0.0,
            hhi_out_cnt=0.0, hhi_out_sum=0.0,
            one_shot_out=0,
            top_out_cnt_ratio=0.0, top_out_sum_ratio=0.0
        )

    # Incoming concentration & partner stats
    if not pair_in.empty:
        idx_cnt = pair_in.groupby('to_acct')['pair_cnt'].idxmax()
        top_in_cnt = pair_in.loc[idx_cnt, ['to_acct', 'pair_cnt']] \
                            .rename(columns={'to_acct': 'acct', 'pair_cnt': 'top_in_cnt'})
        idx_sum = pair_in.groupby('to_acct')['pair_sum'].idxmax()
        top_in_sum = pair_in.loc[idx_sum, ['to_acct', 'pair_sum']] \
                            .rename(columns={'to_acct': 'acct', 'pair_sum': 'top_in_sum'})

        grp_in = pair_in.groupby('to_acct')
        in_stats = grp_in.agg(
            partners_in=('pair_cnt', 'size'),
            total_in_cnt=('pair_cnt', 'sum'),
            total_in_sum=('pair_sum', 'sum')
        ).reset_index().rename(columns={'to_acct': 'acct'})

        hhi_cnt_in = grp_in['pair_cnt'].apply(lambda s: _hhi_from_counts(s.values)).reset_index().rename(
            columns={'to_acct': 'acct', 'pair_cnt': 'hhi_in_cnt'}
        )
        hhi_sum_in = grp_in['pair_sum'].apply(lambda s: _hhi_from_counts(s.values)).reset_index().rename(
            columns={'to_acct': 'acct', 'pair_sum': 'hhi_in_sum'}
        )
        one_shot_in = grp_in['pair_cnt'].apply(lambda s: int((s.values == 1).sum())).reset_index().rename(
            columns={'to_acct': 'acct', 'pair_cnt': 'one_shot_in'}
        )

        g_in = (g_in.merge(in_stats, on='acct', how='left')
                     .merge(top_in_cnt, on='acct', how='left')
                     .merge(top_in_sum, on='acct', how='left')
                     .merge(hhi_cnt_in, on='acct', how='left')
                     .merge(hhi_sum_in, on='acct', how='left')
                     .merge(one_shot_in, on='acct', how='left'))
        g_in['top_in_cnt_ratio'] = g_in['top_in_cnt'] / (g_in['in_deg'] + 1e-6)
        g_in['top_in_sum_ratio'] = g_in['top_in_sum'] / (g_in['in_sum'] + 1e-6)
    else:
        g_in = g_in.assign(
            partners_in=0, total_in_cnt=0, total_in_sum=0.0,
            top_in_cnt=0, top_in_sum=0.0,
            hhi_in_cnt=0.0, hhi_in_sum=0.0,
            one_shot_in=0,
            top_in_cnt_ratio=0.0, top_in_sum_ratio=0.0
        )

    # Reciprocity (directional & direction-agnostic)
    pairs = pair_out[['from_acct', 'to_acct']].drop_duplicates()
    if not pairs.empty:
        pairs = pairs.assign(
            key=pairs['from_acct'].astype(str) + '→' + pairs['to_acct'].astype(str),
            rev_k=pairs['to_acct'].astype(str) + '→' + pairs['from_acct'].astype(str)
        )
        key_set = set(pairs['key'])
        pairs['has_rev'] = pairs['rev_k'].isin(key_set).astype(int)

        # Only consider mutual pairs for reciprocity
        mutual_pairs = pairs[pairs['has_rev'] == 1][['from_acct', 'to_acct']].copy()

        # Count of reciprocal partners (direction-agnostic)
        recip_partner_cnt = mutual_pairs.groupby('from_acct')['to_acct'].nunique().reset_index().rename(
            columns={'from_acct': 'acct', 'to_acct': 'recip_partner_cnt'}
        )
        recip_partner_cnt['acct'] = recip_partner_cnt['acct'].astype(str)

        # Mark reciprocal edges in the non-self transaction set
        mutual_pairs = mutual_pairs.assign(flag=1)
        ns_cols = ['from_acct', 'to_acct', 'txn_amt_TWD']
        ns = not_self_tx[ns_cols].copy()
        ns = (ns.merge(mutual_pairs, on=['from_acct', 'to_acct'], how='left')
                .assign(is_recip_edge=lambda d: d['flag'].fillna(0).astype(int))
                .drop(columns=['flag']))

        # Outgoing/incoming counts & sums on reciprocal edges
        rec_out_txn_cnt = ns[ns['is_recip_edge'] == 1].groupby('from_acct').size().rename(
            'recip_out_txn_cnt').reset_index()
        rec_in_txn_cnt = ns[ns['is_recip_edge'] == 1].groupby('to_acct').size().rename(
            'recip_in_txn_cnt').reset_index()
        rec_out_sum = ns[ns['is_recip_edge'] == 1].groupby('from_acct')['txn_amt_TWD'].sum().rename(
            'recip_out_sum').reset_index()
        rec_in_sum = ns[ns['is_recip_edge'] == 1].groupby('to_acct')['txn_amt_TWD'].sum().rename(
            'recip_in_sum').reset_index()

        # Base merge of g_out and g_in
        base = pd.merge(g_out, g_in, on='acct', how='outer').fillna(0.0)
        base['acct'] = base['acct'].astype(str)

        # Merge reciprocal features
        base = (base
                .merge(recip_partner_cnt, on='acct', how='left')
                .merge(rec_out_txn_cnt.rename(columns={'from_acct': 'acct'}), on='acct', how='left')
                .merge(rec_in_txn_cnt.rename(columns={'to_acct': 'acct'}), on='acct', how='left')
                .merge(rec_out_sum.rename(columns={'from_acct': 'acct'}), on='acct', how='left')
                .merge(rec_in_sum.rename(columns={'to_acct': 'acct'}), on='acct', how='left'))

        # Fill missing values for reciprocal columns
        base[['recip_partner_cnt', 'recip_out_txn_cnt', 'recip_in_txn_cnt',
              'recip_out_sum', 'recip_in_sum']] = base[[
                  'recip_partner_cnt', 'recip_out_txn_cnt', 'recip_in_txn_cnt',
                  'recip_out_sum', 'recip_in_sum']].fillna(0)

        # Backwards compatibility: retain duplicate columns
        base['recip_out_cnt'] = base['recip_partner_cnt']
        base['recip_in_cnt'] = base['recip_partner_cnt']
    else:
        base = pd.merge(g_out, g_in, on='acct', how='outer').fillna(0.0)
        base['acct'] = base['acct'].astype(str)
        base = base.assign(
            recip_partner_cnt=0,
            recip_out_txn_cnt=0,
            recip_in_txn_cnt=0,
            recip_out_sum=0.0,
            recip_in_sum=0.0,
            recip_out_cnt=0,
            recip_in_cnt=0
        )

    # Derived reciprocal ratios & net sum
    base['recip_out_partner_ratio'] = base['recip_partner_cnt'] / (base['partners_out'] + 1e-6)
    base['recip_in_partner_ratio'] = base['recip_partner_cnt'] / (base['partners_in'] + 1e-6)
    base['recip_net_out_sum'] = base['recip_out_sum'] - base['recip_in_sum']
    base['recip_in_out_sum_ratio'] = base['recip_in_sum'] / (base['recip_out_sum'] + 1e-6)

    # Derived flow ratios
    base['ext_out_ratio'] = base['out_ext_cnt'] / (base['out_deg'] + 1e-6)
    base['ext_in_ratio'] = base['in_ext_cnt'] / (base['in_deg'] + 1e-6)
    base['net_flow_sum'] = base['out_sum'] - base['in_sum']
    base['in_out_sum_ratio'] = base['in_sum'] / (base['out_sum'] + 1e-6)
    base['in_out_deg_ratio'] = base['in_deg'] / (base['out_deg'] + 1e-6)

    # Apply caps (for counts) and log transforms (for sums)
    def map_amount(x, max_val):
        return max_val if x > max_val else x

    def map_ratio(x):
        return 20.0 if x > 20.0 else x

    def map_log(x):
        val = x + 1.0
        if val < 0:
            return -np.log10(-val)
        return np.log10(val)

    # Cap counts at thresholds
    cap_cols = [
        ('out_deg', 60), ('out_unique_dst', 60), ('out_ext_cnt', 60),
        ('partners_out', 60), ('total_out_cnt', 60), ('top_out_cnt', 60),
        ('one_shot_out', 60), ('in_unique_src', 60), ('in_deg', 60),
        ('in_ext_cnt', 60), ('partners_in', 60), ('total_in_cnt', 60),
        ('top_in_cnt', 60), ('one_shot_in', 60)
    ]
    for col, threshold in cap_cols:
        base[col] = base[col].apply(map_amount, args=(threshold,))

    # Cap reciprocity counts
    base['recip_out_txn_cnt'] = base['recip_out_txn_cnt'].apply(map_amount, args=(20,))
    base['recip_in_txn_cnt'] = base['recip_in_txn_cnt'].apply(map_amount, args=(20,))
    base['recip_partner_cnt'] = base['recip_partner_cnt'].apply(map_amount, args=(10,))
    base['recip_out_cnt'] = base['recip_out_cnt'].apply(map_amount, args=(16,))
    base['recip_in_cnt'] = base['recip_in_cnt'].apply(map_amount, args=(16,))

    # Apply log transform to sum fields
    sum_cols = [
        'out_sum', 'out_mean', 'out_max', 'total_out_sum', 'top_out_sum',
        'in_sum', 'in_mean', 'in_max', 'total_in_sum', 'top_in_sum',
        'recip_out_sum', 'recip_in_sum', 'recip_net_out_sum', 'net_flow_sum'
    ]
    for col in sum_cols:
        base[col] = base[col].apply(map_log)

    # Cap ratios
    ratio_cols = ['in_out_sum_ratio', 'in_out_deg_ratio', 'recip_in_out_sum_ratio']
    for col in ratio_cols:
        base[col] = base[col].apply(map_ratio)

    # Fill any remaining NaNs with 0
    base = base.fillna(0.0)
    return base

def add_aggregated_features(tx_norm: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    """
    Attach channel- and behavior-based aggregates to account features.

    For each account, computes mean and max channel danger levels, self-transfer
    rates, foreign-currency usage ratios, and external-account flags, then merges
    them into the existing account-level feature frame. Any missing aggregates
    are filled with zeros.
    """
    # Aggregations for out direction
    agg_out = tx_norm.groupby('from_acct').agg(
        out_channel_danger_mean=('channel_danger_level', 'mean'),
        out_channel_danger_max=('channel_danger_level', 'max'),
        out_self_rate=('is_self', 'mean'),
        out_forex_ratio=('is_foreign_currency', 'mean')
    ).reset_index().rename(columns={'from_acct': 'acct'})

    # Aggregations for in direction
    agg_in = tx_norm.groupby('to_acct').agg(
        in_channel_danger_mean=('channel_danger_level', 'mean'),
        in_channel_danger_max=('channel_danger_level', 'max'),
        in_self_rate=('is_self', 'mean'),
        in_forex_ratio=('is_foreign_currency', 'mean')
    ).reset_index().rename(columns={'to_acct': 'acct'})

    # External flags aggregation (max)
    agg_ext_out = tx_norm.groupby('from_acct').agg(
        acct_is_external_from=('from_is_external', 'max')
    ).reset_index().rename(columns={'from_acct': 'acct'})
    agg_ext_in = tx_norm.groupby('to_acct').agg(
        acct_is_external_to=('to_is_external', 'max')
    ).reset_index().rename(columns={'to_acct': 'acct'})

    # Ensure consistent dtypes
    for df in [agg_out, agg_in, agg_ext_out, agg_ext_in]:
        df['acct'] = df['acct'].astype(str)
    feat['acct'] = feat['acct'].astype(str)

    # Reindex aggregations to cover all accounts in feat
    agg_out = agg_out.set_index('acct').reindex(feat['acct']).reset_index()
    agg_in = agg_in.set_index('acct').reindex(feat['acct']).reset_index()
    agg_ext_out = agg_ext_out.set_index('acct').reindex(feat['acct']).reset_index()
    agg_ext_in = agg_ext_in.set_index('acct').reindex(feat['acct']).reset_index()

    # Merge aggregations
    feat = (feat.merge(agg_out, on='acct', how='left')
                 .merge(agg_in,  on='acct', how='left')
                 .merge(agg_ext_out, on='acct', how='left')
                 .merge(agg_ext_in,  on='acct', how='left'))

    # Fill missing aggregated values with zeros
    agg_cols = [
        'out_channel_danger_mean', 'out_channel_danger_max',
        'out_self_rate', 'out_forex_ratio',
        'in_channel_danger_mean', 'in_channel_danger_max',
        'in_self_rate', 'in_forex_ratio',
        'acct_is_external_from', 'acct_is_external_to'
    ]
    feat[agg_cols] = feat[agg_cols].fillna(0)

    feat = feat.drop(columns=['acct_is_external_from', 'acct_is_external_to', 'in_forex_ratio'])
    return feat

n_alert   = len(alert_pd)
n_predict = len(predict_pd)

alertset = set(alert_pd['acct'].sample(n=n_alert, random_state=42).astype(str))
pdset =  alertset | set(predict_pd['acct'].sample(n=n_predict, random_state=99).astype(str))

# Unifying account field types
main_df['from_acct'] = main_df['from_acct'].astype(str)
main_df['to_acct']   = main_df['to_acct'].astype(str)
mask = main_df['from_acct'].isin(pdset) | main_df['to_acct'].isin(pdset)

tx   = main_df.loc[mask].copy()
tx   = ensure_types(tx)
feat = build_graph_features(tx)
feat = add_aggregated_features(tx, feat)

pdset_str = {str(x) for x in pdset}
feat_out = feat[feat['acct'].astype(str).isin(pdset_str)].copy()
feat_out['is_alert'] = feat_out['acct'].astype(str).isin(alertset).astype(bool)

feat_out.to_csv("Feature.csv", index=False)
print("The CSV for getting reliable accounts is finished.")