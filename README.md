# Project Introduction
This is our project for the 2025 AI Cup.  
We use a Temporal Graph Network (TGN) to generate temporal features from transaction data. By combining these TGN features with handcrafted normal features, our XGBoost model achieves a significant performance improvement.

---

## Project Structure
- **csv2parquet/**
    - `csv2parquet.py`
- **Model/**
    - `XGBClassifier.py`
- **Preprocess/**
    - `NormalFeat.py`
- **PreprocessWithModel/**
    - `TGN.py`
- **Reliable_neg_acct/**
    - `safe_acct.csv`

---
## Project Env
- **Python version:**
    - 3.12
- **Package:**
    - See requirements.txt
- **Hwrdware:**
    - Ultra 9 285k
    - 64GB DDR5
    - RTX 5090

---

## How to Reproduce

### 1. Convert raw CSV data
Run `csv2parquet/csv2parquet.py` to transform all raw CSV files into Parquet format.

### 2. Generate temporal graph features (TGN)
Execute: PreprocessWithModel/TGN.py

This produces **tgn_output.csv**, containing temporal graph features derived from account–transaction interactions.

### 3. Generate normal handcrafted features
Execute: Preprocess/NormalFeat.py

This produces **final_data.csv**, which merges handcrafted features with TGN features.

---

## Final Model
After all features are prepared, run: Model/XGBClassifier.py

This script performs PU learning with XGBoost and outputs the final prediction file for submission.

---

## Notes
All feature files must be placed in the correct directory structure as shown above.  
Ensure that Parquet and CSV input files match the expected names before execution.
