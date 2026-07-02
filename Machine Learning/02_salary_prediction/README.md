# Salary Prediction — Simple Linear Regression

A straightforward regression on years of experience vs salary. The dataset is small (30 rows) but it's a clean, near-linear relationship so simple linear regression fits well here.

## The dataset

`Salary_Data.csv` has two columns: years of experience and salary. That's it. The goal is to predict salary from experience, which makes for an easy-to-visualise result.

## Results

Expect an R² around **0.97** on the test set — the relationship really is almost perfectly linear in this data. RMSE will be in the range of $5,000–7,000 depending on the random split.

## How to run

```bash
python simple_linear_regression.py
```

Two PNG plots will be saved to a `plots/` folder (created automatically):
- `salary_vs_experience_training.png`
- `salary_vs_experience_test.png`

Both show the scatter of actual values with the regression line overlaid.

## Code structure

```
SalaryPredictor
├── load_data()      → reads CSV, returns X (experience) and y (salary)
├── train()          → fits a LinearRegression model
├── evaluate()       → returns R² and RMSE as a dict
└── save_plots()     → saves training and test scatter plots with regression line
```

`main()` wires everything together: load → split (67/33) → train → print metrics → save plots.

## Notes

The 1/3 test split is intentional — the original dataset uses it so the model only sees 20 training samples. If you use a more standard 80/20 split the R² goes up slightly but the point of the project is to show the method, not squeeze out every last decimal.

## Sample output

![Training set fit](plots/salary_vs_experience_training.png)
![Test set fit](plots/salary_vs_experience_test.png)
