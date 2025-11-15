# __Project introduction__
#### This is our project for 2025 AI cup event. We use Temporal Graph Network(TGN) to produce efficient features. Using the produced features with original features, our xgboost model accuracy get higher obviously.

## Bullet list
- csv2parquet
    - csv2parquet.py
- Model
    - XGBClassifier.py
- Preprocess
    - GetFeature4PURN.py
    - NormalFeat.py
- PreprocessWithModel
    - PURN4ReliableAcct.py
    - TGN.py

## How to reproduce?
#### 1. Execute 'Preprocess/GetFeature4PURN.py' to get 'Feature.csv'. The file would been used in 'PreprocessWithModel/PURN4ReliableAcct.py'.
#### 2. Execute 'PreprocessWithModel/PURN4ReliableAcct.py' to get 'Feature.csv'. The file would been used in 'PreprocessWithModel/PURN4ReliableAcct.py'.
#### 3. Transform the CSV file first. We use the data in another type in somewhere. You can transform all data by 'csv2parquet/csv2parquet.py.'
#### 4. Execute 'PreprocessWithModel/TGN.py'. You will receive a CSV of efficient features about the trade informs.
#### 5. Execute 'Preprocess/NormalFeat.py'. You will receive a CSV of the normal features we designed.
