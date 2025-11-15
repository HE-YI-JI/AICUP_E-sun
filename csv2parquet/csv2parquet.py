import pandas as pd
"""
Transform the CSV file to Parquet file.
"""
alert_df = pd.read_csv('acct_alert.csv')
predict_df = pd.read_csv('acct_predict.csv')
transaction_df = pd.read_csv('acct_transaction.csv')

alert_df.to_parquet('acct_alert.parquet', engine='pyarrow')
predict_df.to_parquet('acct_predict.parquet', engine='pyarrow')
transaction_df.to_parquet('acct_transaction.parquet', engine='pyarrow')

print("Is finished!")