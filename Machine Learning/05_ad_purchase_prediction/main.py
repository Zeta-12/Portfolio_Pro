import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm


class ClassificationBenchmark:
    DATA_FILE: str = "Social_Network_Ads.csv"
    MODELS: dict = {
        "Logistic Regression": LogisticRegression(random_state=0),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2),
        "Linear SVM": SVC(kernel="linear", random_state=0),
        "Kernel SVM": SVC(kernel="rbf", random_state=0),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(criterion="gini", random_state=0),
        "Random Forest": RandomForestClassifier(n_estimators=10, criterion="gini", random_state=0),
    }

    def __init__(self, data_path: str = DATA_FILE) -> None:
        self.data_path = Path(data_path)
        self.scaler = StandardScaler()
        self.X_train: np.ndarray | None = None
        self.X_test: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.y_test: np.ndarray | None = None

    def load_data(self) -> None:
        df = pd.read_csv(self.data_path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=0
        )
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

    def run_all(self) -> pd.DataFrame:
        records = []
        for name, clf in tqdm(self.MODELS.items(), desc="Training classifiers", unit="model"):
            clf.fit(self.X_train, self.y_train)
            y_pred = clf.predict(self.X_test)
            cv_scores = cross_val_score(clf, self.X_train, self.y_train, cv=10)
            records.append(
                {
                    "Model": name,
                    "Test Accuracy": round(float(accuracy_score(self.y_test, y_pred)), 4),
                    "CV Mean": round(float(cv_scores.mean()), 4),
                    "CV Std": round(float(cv_scores.std()), 4),
                }
            )
        return pd.DataFrame(records).sort_values("CV Mean", ascending=False)

    def tune_best_model(self) -> dict:
        param_grid = [
            {"C": [0.25, 0.5, 0.75, 1], "kernel": ["linear"]},
            {
                "C": [0.25, 0.5, 0.75, 1],
                "kernel": ["rbf"],
                "gamma": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            },
        ]
        grid = GridSearchCV(SVC(), param_grid, refit=True, cv=10, scoring="accuracy")
        grid.fit(self.X_train, self.y_train)
        return {
            "best_params": grid.best_params_,
            "best_cv_accuracy": round(float(grid.best_score_), 4),
        }

    def _plot_decision_boundary(
        self,
        ax: plt.Axes,
        clf,
        X: np.ndarray,
        y: np.ndarray,
        title: str,
    ) -> None:
        X_orig = self.scaler.inverse_transform(X)
        X1, X2 = np.meshgrid(
            np.arange(X_orig[:, 0].min() - 10, X_orig[:, 0].max() + 10, 1.0),
            np.arange(X_orig[:, 1].min() - 1000, X_orig[:, 1].max() + 1000, 1.0),
        )
        Z = clf.predict(
            self.scaler.transform(np.c_[X1.ravel(), X2.ravel()])
        ).reshape(X1.shape)
        ax.contourf(X1, X2, Z, alpha=0.6, cmap=ListedColormap(["#FFAAAA", "#AAFFAA"]))
        for label, color in zip(np.unique(y), ["red", "green"]):
            ax.scatter(
                X_orig[y == label, 0], X_orig[y == label, 1],
                c=color, label=str(label), s=20,
            )
        ax.set_title(title)
        ax.set_xlabel("Age")
        ax.set_ylabel("Estimated Salary")
        ax.legend()

    def save_results(
        self, results: pd.DataFrame, output_dir: Path = Path("plots")
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Comparison bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(results["Model"], results["CV Mean"], xerr=results["CV Std"], color="steelblue")
        ax.set_xlabel("CV Accuracy")
        ax.set_title("Classifier Comparison — 10-Fold CV Accuracy")
        ax.set_xlim(0, 1)
        fig.tight_layout()
        fig.savefig(output_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Per-model confusion matrices
        for name, clf in tqdm(self.MODELS.items(), desc="Saving plots", unit="model"):
            clf.fit(self.X_train, self.y_train)
            y_pred = clf.predict(self.X_test)
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            self._plot_decision_boundary(axes[0], clf, self.X_train, self.y_train, f"{name} (Train)")
            self._plot_decision_boundary(axes[1], clf, self.X_test, self.y_test, f"{name} (Test)")
            fig.tight_layout()
            slug = name.lower().replace(" ", "_")
            fig.savefig(output_dir / f"boundary_{slug}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            cm_fig, cm_ax = plt.subplots()
            ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(self.y_test, y_pred)).plot(ax=cm_ax)
            cm_ax.set_title(f"{name} — Confusion Matrix")
            cm_fig.savefig(output_dir / f"cm_{slug}.png", dpi=150, bbox_inches="tight")
            plt.close(cm_fig)


def main() -> None:
    benchmark = ClassificationBenchmark()
    print("[1/4] Loading and scaling data...")
    benchmark.load_data()
    print("[2/4] Training and evaluating 7 classifiers (this may take a minute)...")
    results = benchmark.run_all()
    print("\nClassifier Comparison:")
    print(results.to_string(index=False))
    print("[3/4] Tuning best model with GridSearchCV (this may take a few minutes)...")
    tuning = benchmark.tune_best_model()
    print(f"\nBest SVC params (GridSearchCV): {tuning['best_params']}")
    print(f"Best CV Accuracy              : {tuning['best_cv_accuracy']}")
    print("[4/4] Saving plots (decision boundaries + confusion matrices)...")
    benchmark.save_results(results)
    print("\nPlots saved to plots/")


if __name__ == "__main__":
    main()
