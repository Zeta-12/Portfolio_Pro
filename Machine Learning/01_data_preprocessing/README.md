# Data Preprocessing Pipeline

This isn't really a standalone project — it's more of a reference implementation showing how to handle the messy stuff that comes before any actual modelling. Every other project in this repo skips straight to the interesting part, so this one exists to show the full preprocessing chain in one place.

## The dataset

`Data.csv` is a tiny 10-row table with three features (country, age, salary) and one binary target (purchased). It has missing values and a categorical column, which makes it a good sandbox for testing preprocessing steps.

## What it does

The pipeline runs four steps in sequence:

1. **Missing value imputation** — fills gaps in the numeric columns using the column mean via `SimpleImputer`
2. **One-hot encoding** — turns the country column into dummy variables with `ColumnTransformer` + `OneHotEncoder`
3. **Label encoding** — converts the binary target string (Yes/No) into 0/1
4. **Train/test split + feature scaling** — splits 80/20 and applies `StandardScaler` only to the numeric features (not the dummies)

The scaler is fitted on the training set and applied to both sets — which is the correct way to do it.

## How to run

```bash
python data_preprocessing_tools.py
```

It prints the shape of the resulting train/test arrays. No plots, no model — just clean data coming out.

## Code structure

```
DataPreprocessor
├── load_data()          → reads the CSV, returns X and y as numpy arrays
├── handle_missing()     → imputes missing values in columns 1-2
├── encode_features()    → one-hot encodes X, label-encodes y
├── split_and_scale()    → train/test split + StandardScaler on numeric columns
└── run()                → orchestrates all of the above
```

## Notes

The `DATA_FILE` class constant at the top is the only thing you'd change if you want to run this on a different CSV. The scaler instance is stored on the object so you can call `self.scaler.transform()` later if needed.
