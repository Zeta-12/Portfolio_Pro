import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
from sklearn.decomposition import KernelPCA, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DimensionalityReductionBenchmark:
    DATA_FILE: str = "Wine.csv"
    REDUCERS: dict = {
        "PCA": PCA(n_components=2),
        "LDA": LDA(n_components=2),
        "Kernel PCA": KernelPCA(n_components=2, kernel="rbf"),
    }

    def __init__(self, data_path: str = DATA_FILE) -> None:
        self.data_path = Path(data_path)
        self.scaler = StandardScaler()

    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(self.data_path)
        return df.iloc[:, :-1].values, df.iloc[:, -1].values

    def preprocess(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        return X_train, X_test, y_train, y_test

    def run_all(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> pd.DataFrame:
        records = []
        for name, reducer in self.REDUCERS.items():
            Xt_train = reducer.fit_transform(X_train, y_train)
            Xt_test = reducer.transform(X_test)
            clf = LogisticRegression(random_state=0, max_iter=200)
            clf.fit(Xt_train, y_train)
            acc = accuracy_score(y_test, clf.predict(Xt_test))
            records.append({"Reducer": name, "Accuracy": round(float(acc), 4)})
        return pd.DataFrame(records)

    def _plot_decision_regions(
        self,
        ax: plt.Axes,
        X: np.ndarray,
        y: np.ndarray,
        clf: LogisticRegression,
        title: str,
    ) -> None:
        X1_min, X1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        X2_min, X2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        XX1, XX2 = np.meshgrid(
            np.arange(X1_min, X1_max, 0.02),
            np.arange(X2_min, X2_max, 0.02),
        )
        Z = clf.predict(np.c_[XX1.ravel(), XX2.ravel()]).reshape(XX1.shape)
        ax.contourf(
            XX1, XX2, Z, alpha=0.6,
            cmap=ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"]),
        )
        for label, color in zip(np.unique(y), ["red", "green", "blue"]):
            ax.scatter(X[y == label, 0], X[y == label, 1], c=color, label=str(label), s=20)
        ax.set_title(title)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend()

    def save_plots(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        output_dir: Path = Path("plots"),
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, reducer in self.REDUCERS.items():
            Xt_train = reducer.fit_transform(X_train, y_train)
            Xt_test = reducer.transform(X_test)
            clf = LogisticRegression(random_state=0, max_iter=200)
            clf.fit(Xt_train, y_train)
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            self._plot_decision_regions(axes[0], Xt_train, y_train, clf, f"{name} (Train)")
            self._plot_decision_regions(axes[1], Xt_test, y_test, clf, f"{name} (Test)")
            fig.tight_layout()
            slug = name.lower().replace(" ", "_")
            fig.savefig(
                output_dir / f"{slug}_decision_regions.png", dpi=150, bbox_inches="tight"
            )
            plt.close(fig)


def main() -> None:
    bench = DimensionalityReductionBenchmark()
    print("[1/4] Loading Wine dataset...")
    X, y = bench.load_data()
    print("[2/4] Preprocessing (split + scaling)...")
    X_train, X_test, y_train, y_test = bench.preprocess(X, y)
    print("[3/4] Running PCA, LDA and Kernel PCA + Logistic Regression...")
    results = bench.run_all(X_train, X_test, y_train, y_test)
    print("\nDimensionality Reduction Comparison:")
    print(results.to_string(index=False))
    print("[4/4] Saving decision region plots...")
    bench.save_plots(X_train, X_test, y_train, y_test)
    print("\nPlots saved to plots/")


if __name__ == "__main__":
    main()
