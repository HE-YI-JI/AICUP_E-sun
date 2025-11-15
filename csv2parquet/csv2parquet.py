import pandas as pd
"""
Transform the CSV file to Parquet file.
"""
alert_df = pd.read_csv('Preprocess/acct_alert.csv')
predict_df = pd.read_csv('Preprocess/acct_predict.csv')
transaction_df = pd.read_csv('Preprocess/acct_transaction.csv')

alert_df.to_parquet('Preprocess/acct_alert.parquet', engine='pyarrow')
predict_df.to_parquet('Preprocess/acct_predict.parquet', engine='pyarrow')
transaction_df.to_parquet('Preprocess/acct_transaction.parquet', engine='pyarrow')

print("Is finished!")