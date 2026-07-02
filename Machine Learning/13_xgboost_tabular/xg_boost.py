import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


class TabularClassifier:
    DATA_FILE: str = "Data.csv"
    CV_FOLDS: int = 10

    def __init__(self, data_path: str = DATA_FILE) -> None:
        self.data_path = Path(data_path)
        self.model: XGBClassifier | None = None

    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(self.data_path)
        return df.iloc[:, :-1].values, df.iloc[:, -1].values

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model = XGBClassifier()
        self.model.fit(X_train, y_train)

    def evaluate(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        y_pred = self.model.predict(X_test)
        cv_scores = cross_val_score(
            self.model, X_train, y_train, cv=self.CV_FOLDS
        )
        return {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "cv_mean": round(float(cv_scores.mean()), 4),
            "cv_std": round(float(cv_scores.std()), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }

    def save_plots(
        self, cm: np.ndarray, output_dir: Path = Path("plots")
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)
        ax.set_title("XGBoost — Confusion Matrix")
        fig.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    clf = TabularClassifier()
    print("[1/4] Loading data...")
    X, y = clf.load_data()
    y = LabelEncoder().fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    print("[2/4] Training XGBoost...")
    clf.train(X_train, y_train)
    print(f"[3/4] Evaluating (test set + {TabularClassifier.CV_FOLDS}-fold CV on train set)...")
    metrics = clf.evaluate(X_train, X_test, y_train, y_test)
    print(f"Test Accuracy : {metrics['accuracy']}")
    print(f"CV Accuracy   : {metrics['cv_mean']} \u00b1 {metrics['cv_std']}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
    print("[4/4] Saving plots...")
    clf.save_plots(metrics["confusion_matrix"])
    print("Plots saved to plots/")


if __name__ == "__main__":
    main()
