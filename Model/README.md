# Description of File
This script trains an XGBoost model using a Positive–Unlabeled (PU) learning framework, then predicts risk probabilities for accounts listed in `acct_predict.csv`. The output is a CSV file containing the final submission results.

## File Included
- XGBClassifier.py

## How to Use
Execute `XGBClassifier.py`.  
Make sure the following files exist in the working directory:
- `final_data.csv`
- `acct_predict.csv`
- `safe_acct.csv`