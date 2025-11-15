# __Project introduction__
#### This is our project for 2025 AIcup event. We use Temporal Graph Network(TGN) to produce efficient feature. Using the produced features with original features, our xgboost model accuracy get higher obviously.

## Bullet list
#### AICUP_E-sun/
#### ├── csv2parquet/
#### │   └── csv2parquet.py
#### ├── Model/
#### │   └── XGBClassifier.py
#### ├── Preprocess/
#### │   └── NormalFeat.py
#### ├── PreprocessWithModel/
#### │   └── TGN.py
#### └── README.md

## How to reproduce?
#### 1. Transform the CSV file first. We use the data in another type in somewhere. You can Transform all data in 'csv2parquet/csv2parquet.py'
#### 2. Execute 'PreprocessWithModel/TGN.py'. You will receive a CSV of efficient features about the trade informs.
#### 3. Execute 'Preprocess/NormalFeat.py'. You will receive a CSV of the normal features we designed.
