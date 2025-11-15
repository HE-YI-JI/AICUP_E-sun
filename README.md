# __Project introduction__
#### This is our project for 2025 AI cup event. We use Temporal Graph Network(TGN) to produce efficient features. Using the produced features with original features, our xgboost model accuracy get higher obviously.

## Bullet list
#### - csv2parquet
####     - csv2parquet.py
#### - Model
####     - XGBClassifier.py
#### - Preprocess
####     - NormalFeat.py
#### - PreprocessWithModel
####     - TGN.py

## How to reproduce?
#### 1. Transform the CSV file first. We use the data in another type in somewhere. You can transform all data by 'csv2parquet/csv2parquet.py.'
#### 2. Execute 'PreprocessWithModel/TGN.py'. You will receive a CSV of efficient features about the trade informs.
#### 3. Execute 'Preprocess/NormalFeat.py'. You will receive a CSV of the normal features we designed.
