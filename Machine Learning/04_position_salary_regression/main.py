import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


class RegressionBenchmark:
    DATA_FILE: str = "Position_Salaries.csv"
    QUERY_LEVEL: float = 6.5

    def __init__(self, data_path: str = DATA_FILE) -> None:
        self.data_path = Path(data_path)
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self._predictors: dict[str, callable] = {}

    def load_data(self) -> None:
        df = pd.read_csv(self.data_path)
        self.X = df.iloc[:, 1:-1].values
        self.y = df.iloc[:, -1].values

    def fit_all(self) -> None:
        # Polynomial Regression
        poly = PolynomialFeatures(degree=4)
        lr = LinearRegression().fit(poly.fit_transform(self.X), self.y)
        self._predictors["Polynomial Regression"] = (
            lambda x, _poly=poly, _lr=lr: _lr.predict(_poly.transform(x))
        )

        # Support Vector Regression (requires feature scaling)
        sc_X, sc_y = StandardScaler(), StandardScaler()
        svr = SVR(kernel="rbf").fit(
            sc_X.fit_transform(self.X),
            sc_y.fit_transform(self.y.reshape(-1, 1)).ravel(),
        )
        self._predictors["SVR"] = (
            lambda x, _scX=sc_X, _scY=sc_y, _svr=svr: _scY.inverse_transform(
                _svr.predict(_scX.transform(x)).reshape(-1, 1)
            ).ravel()
        )

        # Decision Tree
        dt = DecisionTreeRegressor(random_state=0).fit(self.X, self.y)
        self._predictors["Decision Tree"] = dt.predict

        # Random Forest
        rf = RandomForestRegressor(n_estimators=10, random_state=0).fit(
            self.X, self.y
        )
        self._predictors["Random Forest"] = rf.predict

    def compare_predictions(self) -> dict[str, float]:
        return {
            name: round(float(predict([[self.QUERY_LEVEL]])[0]), 2)
            for name, predict in self._predictors.items()
        }

    def save_plots(self, output_dir: Path = Path("plots")) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        X_grid = np.arange(float(self.X.min()), float(self.X.max()), 0.01).reshape(
            -1, 1
        )
        for name, predict in self._predictors.items():
            fig, ax = plt.subplots()
            ax.scatter(self.X, self.y, color="red", label="Data")
            ax.plot(X_grid, predict(X_grid), color="blue", label=name)
            ax.set_title(name)
            ax.set_xlabel("Position Level")
            ax.set_ylabel("Salary ($)")
            ax.legend()
            slug = name.lower().replace(" ", "_")
            fig.savefig(output_dir / f"{slug}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    benchmark = RegressionBenchmark()
    print("[1/4] Loading data...")
    benchmark.load_data()
    print("[2/4] Fitting all models (Polynomial, SVR, Decision Tree, Random Forest)...")
    benchmark.fit_all()
    print("[3/4] Comparing predictions...")
    predictions = benchmark.compare_predictions()
    print(f"\nPredicted salary for position level {RegressionBenchmark.QUERY_LEVEL}:")
    for model_name, pred in predictions.items():
        print(f"  {model_name:<25}: ${pred:,.2f}")
    print("[4/4] Saving plots...")
    benchmark.save_plots()
    print("\nPlots saved to plots/")


if __name__ == "__main__":
    main()
