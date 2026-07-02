import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class SalaryPredictor:
    DATA_FILE: str = "Salary_Data.csv"

    def __init__(self, data_path: str = DATA_FILE) -> None:
        self.data_path = Path(data_path)
        self.model: LinearRegression | None = None

    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(self.data_path)
        return df.iloc[:, :-1].values, df.iloc[:, -1].values

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> dict[str, float]:
        y_pred = self.model.predict(X_test)
        return {
            "r2": round(float(r2_score(y_test, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2),
        }

    def save_plots(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        output_dir: Path = Path("plots"),
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        regression_line = self.model.predict(X_train)
        for X_set, y_set, split_name in [
            (X_train, y_train, "training"),
            (X_test, y_test, "test"),
        ]:
            fig, ax = plt.subplots()
            ax.scatter(X_set, y_set, color="red", label="Actual")
            ax.plot(X_train, regression_line, color="blue", label="Regression line")
            ax.set_title(f"Salary vs Experience ({split_name.capitalize()} set)")
            ax.set_xlabel("Years of Experience")
            ax.set_ylabel("Salary ($)")
            ax.legend()
            fig.savefig(
                output_dir / f"salary_vs_experience_{split_name}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)


def main() -> None:
    predictor = SalaryPredictor()
    print("[1/4] Loading data...")
    X, y = predictor.load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 / 3, random_state=0
    )
    print("[2/4] Training linear regression...")
    predictor.train(X_train, y_train)
    print("[3/4] Evaluating model...")
    metrics = predictor.evaluate(X_test, y_test)
    print(f"R²  : {metrics['r2']}")
    print(f"RMSE: {metrics['rmse']}")
    print("[4/4] Saving plots...")
    predictor.save_plots(X_train, y_train, X_test, y_test)
    print("Plots saved to plots/")


if __name__ == "__main__":
    main()
