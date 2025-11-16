"""
Build node-level features from transaction data for downstream XGBoost training.

This module loads alert account labels and raw transaction records, constructs
account-based aggregate features (amount statistics, digit patterns, night and
large transaction ratios), and joins them with TGN outputs to produce the final
training dataset for an XGB classifier.
"""
import datetime
import pandas as pd
import dask.array as da
import dask.dataframe as dd
import numpy as np
import yfinance as yf

alert_acct = dd.read_parquet(r'acct_alert.parquet', engine='pyarrow')

def get_fx_rate(code, date) -> float:
    """
    Fetch FX rate to TWD for a given currency code and date.

    For most currencies, this queries the <code>TWD=X</code> pair from Yahoo
    Finance starting at the given date and takes the first available closing
    price. For MXN, the rate is computed as USD→TWD multiplied by MXN→USD.
    """
    if code != 'MXN':
        return yf.Ticker(code+"TWD=X").history(start=date, end=date+datetime.timedelta(days=1400)).iloc[0]['Close']
    else:
        return get_fx_rate('USD', date) * yf.Ticker("MXNUSD=X").history(start=date, end=date+datetime.timedelta(days=1400)).iloc[0]['Close']

def build_node_feature(data: dd.DataFrame) -> dd.DataFrame:
    """
    Build account-level features from raw transaction records.

    The input is a Dask DataFrame of edge-level transactions with at least
    the following columns: from_acct, to_acct, txn_date, txn_time, txn_amt,
    currency_type, and is_self_txn. The function aggregates these records
    into one row per account ('acct') and computes:

    * Daily transaction date dispersion for senders and receivers.
    * Digit-based statistics (first/second/third/last digit of amounts).
    * Log-transformed amount statistics for sent and received transactions.
    * Night-time transaction statistics (23:00–06:00).
    * Large transaction statistics (txn_amt > 30000).
    * Ratios such as night_txn_ratio and large_txn_night_ratio.
    * Maximum counterpart count and whether the account appears in alert_acct.

    Returns a Dask DataFrame with one row per account containing all engineered features.
    """
    data = data.copy()
    df = dd.DataFrame.from_dict({})

    df['acct'] = dd.concat([data['from_acct'], data['to_acct']]).drop_duplicates()

    df = df.reset_index(drop=True)
    df = df.join(data.groupby('from_acct').txn_date.std().rename('daily_send_std'), on='acct', how='left')
    df = df.join(data.groupby('to_acct').txn_date.std().rename('daily_recv_std'), on='acct', how='left')
    
    data['txn_date'] = dd.to_timedelta(data['txn_date'], unit='D') + pd.to_datetime('2024-01-01')
    data['txn_time'] = dd.to_datetime(data['txn_date'].dt.date.astype(str) + data['txn_time'].astype(str), format='%Y-%m-%d%H:%M:%S')

    data['is_self_txn'] = data['is_self_txn'].map({'Y': 1, 'N': 0, 'UNK': 0}).fillna(0)
    data = data.assign(is_foreign=data['currency_type'] != 'TWD')
    data[data['is_foreign'] == 1]['txn_amt'] = data[data['is_foreign'] == 1].apply(
        lambda row: row['txn_amt'] * get_fx_rate(row['currency_type'], row['txn_date']), axis=1, meta=('txn_amt', 'float64')
    )
    data['first_digit'] = dd.to_numeric(data['txn_amt'].astype(str).str.get(0), errors='coerce')
    data.groupby('first_digit').size()
    
    df = df.join(data.groupby('from_acct').agg(
        send_amt_first_digit_mean=('first_digit', 'mean'),
        send_amt_first_digit_std=('first_digit', 'std'),
        send_amt_first_digit_min=('first_digit', 'min'),
        send_amt_first_digit_max=('first_digit', 'max'),
        send_amt_first_digit_count=('first_digit', 'count')
    ), on='acct', how='left')
    df = df.join(data.groupby('to_acct').agg(
        recv_amt_first_digit_mean=('first_digit', 'mean'),
        recv_amt_first_digit_std=('first_digit', 'std'),
        recv_amt_first_digit_min=('first_digit', 'min'),
        recv_amt_first_digit_max=('first_digit', 'max'),
        recv_amt_first_digit_count=('first_digit', 'count')
    ), on='acct', how='left')

    data['second_digit'] = dd.to_numeric(data['txn_amt'].astype(str).str.get(1), errors='coerce')
    df = df.join(data.groupby('from_acct').agg(
        send_amt_second_digit_mean=('second_digit', 'mean'),
        send_amt_second_digit_std=('second_digit', 'std'),
        send_amt_second_digit_min=('second_digit', 'min'),
        send_amt_second_digit_max=('second_digit', 'max'),
        send_amt_second_digit_count=('second_digit', 'count')
    ), on='acct', how='left')
    df = df.join(data.groupby('to_acct').agg(
        recv_amt_second_digit_mean=('second_digit', 'mean'),
        recv_amt_second_digit_std=('second_digit', 'std'),
        recv_amt_second_digit_min=('second_digit', 'min'),
        recv_amt_second_digit_max=('second_digit', 'max'),
        recv_amt_second_digit_count=('second_digit', 'count')
    ), on='acct', how='left') 

    data['third_digit'] = dd.to_numeric(data['txn_amt'].astype(str).str.get(2), errors='coerce')
    df = df.join(data.groupby('from_acct').agg(
        send_amt_third_digit_mean=('third_digit', 'mean'),
        send_amt_third_digit_std=('third_digit', 'std'),
        send_amt_third_digit_min=('third_digit', 'min'),
        send_amt_third_digit_max=('third_digit', 'max'),
        send_amt_third_digit_count=('third_digit', 'count')
    ), on='acct', how='left')
    df = df.join(data.groupby('to_acct').agg(
        recv_amt_third_digit_mean=('third_digit', 'mean'),
        recv_amt_third_digit_std=('third_digit', 'std'),
        recv_amt_third_digit_min=('third_digit', 'min'),
        recv_amt_third_digit_max=('third_digit', 'max'),
        recv_amt_third_digit_count=('third_digit', 'count')
    ), on='acct', how='left') 

    data['last_digit'] = dd.to_numeric(data['txn_amt'].astype(str).str.get(-1), errors='coerce')
    df = df.join(data.groupby('from_acct').agg(
        send_amt_last_digit_mean=('last_digit', 'mean'),
        send_amt_last_digit_std=('last_digit', 'std'),
        send_amt_last_digit_min=('last_digit', 'min'),
        send_amt_last_digit_max=('last_digit', 'max'),
        send_amt_last_digit_count=('last_digit', 'count')
    ), on='acct', how='left')
    df = df.join(data.groupby('to_acct').agg(
        recv_amt_last_digit_mean=('last_digit', 'mean'),
        recv_amt_last_digit_std=('last_digit', 'std'),
        recv_amt_last_digit_min=('last_digit', 'min'),
        recv_amt_last_digit_max=('last_digit', 'max'),
        recv_amt_last_digit_count=('last_digit', 'count')
    ), on='acct', how='left') 

    df = df.join(data[data.txn_amt % 1000 == 0].groupby('from_acct').size().rename('send_amt_round_count'), on='acct', how='left')
    df = df.join(data[data.txn_amt % 1000 == 0].groupby('to_acct').size().rename('recv_amt_round_count'), on='acct', how='left')
    data['txn_amt'] = da.log1p(data['txn_amt'])
    
    df = df.join(data.groupby('from_acct').agg(
        send_amt_sum=('txn_amt', 'sum'),
        send_amt_mean=('txn_amt', 'mean'),
        send_amt_std=('txn_amt', 'std'),
        send_amt_min=('txn_amt', 'min'),
        send_amt_max=('txn_amt', 'max'),
        send_count=('to_acct', 'count'),
        send_foreign_count=('is_foreign', 'count'),
        send_self_txn_ratio=('is_self_txn', 'mean')
    ), on='acct', how='left')
    df = df.join(data.groupby('from_acct').to_acct.nunique().rename('send_acct_nunique'), on='acct', how='left')

    df = df.join(data.groupby('to_acct').agg(
        recv_amt_sum=('txn_amt', 'sum'),
        recv_amt_mean=('txn_amt', 'mean'),
        recv_amt_std=('txn_amt', 'std'),
        recv_amt_min=('txn_amt', 'min'),
        recv_amt_max=('txn_amt', 'max'),
        recv_count=('from_acct', 'count'),
        recv_foreign_count=('is_foreign', 'count'),
        recv_self_txn_ratio=('is_self_txn', 'mean')
    ), on='acct', how='left')
    df = df.join(data.groupby('to_acct').from_acct.nunique().rename('recv_acct_nunique'), on='acct', how='left')

    night_txn = data[(data['txn_time'].dt.hour >= 23) | (data['txn_time'].dt.hour < 6)]
    df = df.join(night_txn.groupby('from_acct').agg(
        night_send_sum=('txn_amt', 'sum'),
        night_send_mean=('txn_amt', 'mean'),
        night_send_std=('txn_amt', 'std'),
        night_send_min=('txn_amt', 'min'),
        night_send_max=('txn_amt', 'max'),
        night_send_count=('to_acct', 'count'),
        night_send_foreign_count=('is_foreign', 'count'),
        night_send_self_txn_ratio=('is_self_txn', 'mean')
    ), on='acct', how='left')
    df = df.join(data.groupby('from_acct').to_acct.nunique().rename('night_send_acct_nunique'), on='acct', how='left')
    df = df.join(night_txn.groupby('to_acct').agg(
        night_recv_sum=('txn_amt', 'sum'),
        night_recv_mean=('txn_amt', 'mean'),
        night_recv_std=('txn_amt', 'std'),
        night_recv_min=('txn_amt', 'min'),
        night_recv_max=('txn_amt', 'max'),
        night_recv_count=('from_acct', 'count'),
        night_recv_foreign_count=('is_foreign', 'count'),
        night_recv_self_txn_ratio=('is_self_txn', 'mean')
    ), on='acct', how='left')
    df = df.join(data.groupby('to_acct').from_acct.nunique().rename('night_recv_acct_nunique'), on='acct', how='left')

    large_txn = data[data['txn_amt'] > 30000]
    df = df.join(large_txn.groupby('from_acct').agg(
        large_send_sum=('txn_amt', 'sum'),
        large_send_mean=('txn_amt', 'mean'),
        large_send_std=('txn_amt', 'std'),
        large_send_min=('txn_amt', 'min'),
        large_send_max=('txn_amt', 'max'),
        large_send_count=('to_acct', 'count'),
        large_send_foreign_count=('is_foreign', 'count'),
        large_send_self_txn_ratio=('is_self_txn', 'mean')
    ), on='acct', how='left')
    df = df.join(data.groupby('from_acct').to_acct.nunique().rename('large_send_acct_nunique'), on='acct', how='left')
    df = df.join(large_txn.groupby('to_acct').agg(
        large_recv_sum=('txn_amt', 'sum'),
        large_recv_mean=('txn_amt', 'mean'),
        large_recv_std=('txn_amt', 'std'),
        large_recv_min=('txn_amt', 'min'),
        large_recv_max=('txn_amt', 'max'),
        large_recv_count=('from_acct', 'count'),
        large_recv_foreign_count=('is_foreign', 'count'),
        large_recv_self_txn_ratio=('is_self_txn', 'mean')
    ), on='acct', how='left')

    df = df.join(data.groupby('to_acct').from_acct.nunique().rename('large_recv_acct_nunique'), on='acct', how='left')
    df = df.fillna(0)
    df['night_txn_ratio'] = ((df['night_send_sum'] + df['night_recv_sum']) / (df['send_count'] + df['recv_count'])).where(df['send_count'] + df['recv_count'] != 0, 0)
    large_txn_night = large_txn[(large_txn['txn_time'].dt.hour >= 22) | (large_txn['txn_time'].dt.hour < 6)]
    df = df.join(large_txn_night.groupby('from_acct').txn_amt.count().add(large_txn_night.groupby('to_acct').txn_amt.count(), fill_value=0).rename('large_txn_night_count'), on='acct', how='left').fillna(0)
    df['large_txn_night_ratio'] = (df['large_txn_night_count'] / (df['large_send_count'] + df['large_recv_count'])).where((df['large_send_count'] + df['large_recv_count']) != 0, 0)
    df = df.join(data.groupby(['from_acct', 'to_acct']).size().groupby('from_acct').max().add(data.groupby(['to_acct', 'from_acct']).size().groupby('to_acct').max(), fill_value=0).rename('acct_count_max'), on='acct', how='left').fillna(0)
    df['is_alert'] = df['acct'].isin(alert_acct['acct']).astype(bool)

    return df

if __name__ == "__main__":
    # Read the file produced by csv2parquet.py
    all_data = dd.read_parquet(r'acct_transaction.parquet', engine='pyarrow')
    node_feature = build_node_feature(all_data).compute()
    df = pd.read_csv('tgn_output.csv')

    # merge the features of tgn_output.csv and this 'node_feature', save to 'final_data.csv'.
    node_feature.merge(df, left_on='acct', right_on='0', how='outer').drop(columns=['0']).to_csv('final_data.csv', index=False)
