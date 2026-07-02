import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


class StartupProfitPredictor:
    DATA_FILE: str = "50_Startups.csv"

    def __init__(self, data_path: str = DATA_FILE) -> None:
        self.data_path = Path(data_path)
        self.model: LinearRegression | None = None
        self._ct: ColumnTransformer | None = None

    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(self.data_path)
        return df.iloc[:, :-1].values, df.iloc[:, -1].values

    def preprocess(self, X: np.ndarray) -> np.ndarray:
        self._ct = ColumnTransformer(
            transformers=[("encoder", OneHotEncoder(), [3])], remainder="passthrough"
        )
        return np.array(self._ct.fit_transform(X))

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
        y_test: np.ndarray,
        y_pred: np.ndarray,
        output_dir: Path = Path("plots"),
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, color="steelblue", alpha=0.7)
        lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lim, lim, "r--", label="Perfect fit")
        ax.set_title("Actual vs Predicted Profit")
        ax.set_xlabel("Actual Profit ($)")
        ax.set_ylabel("Predicted Profit ($)")
        ax.legend()
        fig.savefig(output_dir / "actual_vs_predicted.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    predictor = StartupProfitPredictor()
    print("[1/5] Loading data...")
    X, y = predictor.load_data()
    print("[2/5] Encoding categorical features...")
    X = predictor.preprocess(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    print("[3/5] Training multiple linear regression...")
    predictor.train(X_train, y_train)
    print("[4/5] Evaluating model...")
    y_pred = predictor.model.predict(X_test)
    metrics = predictor.evaluate(X_test, y_test)
    print(f"R²  : {metrics['r2']}")
    print(f"RMSE: {metrics['rmse']}")
    print("[5/5] Saving plots...")
    predictor.save_plots(y_test, y_pred)
    print("Plots saved to plots/")


if __name__ == "__main__":
    main()
