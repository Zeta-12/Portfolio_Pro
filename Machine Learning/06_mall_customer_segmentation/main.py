import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering, KMeans
from tqdm import tqdm


class CustomerSegmentation:
    DATA_FILE: str = "Mall_Customers.csv"
    N_CLUSTERS: int = 5

    def __init__(self, data_path: str = DATA_FILE) -> None:
        self.data_path = Path(data_path)
        self.X: np.ndarray | None = None

    def load_data(self) -> None:
        df = pd.read_csv(self.data_path)
        self.X = df.iloc[:, [3, 4]].values

    def run_kmeans(self) -> tuple[np.ndarray, KMeans]:
        kmeans = KMeans(n_clusters=self.N_CLUSTERS, init="k-means++", random_state=42)
        return kmeans.fit_predict(self.X), kmeans

    def run_hierarchical(self) -> np.ndarray:
        hc = AgglomerativeClustering(
            n_clusters=self.N_CLUSTERS, metric="euclidean", linkage="ward"
        )
        return hc.fit_predict(self.X)

    def save_plots(self, output_dir: Path = Path("plots")) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        colors = ["red", "blue", "green", "cyan", "magenta"]

        # Elbow curve
        wcss = [
            KMeans(n_clusters=k, init="k-means++", random_state=42)
            .fit(self.X)
            .inertia_
            for k in tqdm(range(1, 11), desc="Elbow method", unit="k", leave=False)
        ]
        fig, ax = plt.subplots()
        ax.plot(range(1, 11), wcss, marker="o")
        ax.set_title("Elbow Method — Optimal Number of Clusters")
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("WCSS")
        fig.savefig(output_dir / "elbow_method.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Dendrogram
        fig, ax = plt.subplots(figsize=(10, 5))
        sch.dendrogram(sch.linkage(self.X, method="ward"), ax=ax)
        ax.set_title("Hierarchical Clustering Dendrogram")
        ax.set_xlabel("Customers")
        ax.set_ylabel("Euclidean Distance")
        fig.savefig(output_dir / "dendrogram.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # K-Means clusters
        y_kmeans, kmeans = self.run_kmeans()
        fig, ax = plt.subplots()
        for i, color in enumerate(colors):
            ax.scatter(
                self.X[y_kmeans == i, 0], self.X[y_kmeans == i, 1],
                s=100, c=color, label=f"Cluster {i + 1}",
            )
        ax.scatter(
            kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c="yellow", label="Centroids", edgecolors="black",
        )
        ax.set_title("K-Means Customer Clusters")
        ax.set_xlabel("Annual Income (k$)")
        ax.set_ylabel("Spending Score (1–100)")
        ax.legend()
        fig.savefig(output_dir / "kmeans_clusters.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Hierarchical clusters
        y_hc = self.run_hierarchical()
        fig, ax = plt.subplots()
        for i, color in enumerate(colors):
            ax.scatter(
                self.X[y_hc == i, 0], self.X[y_hc == i, 1],
                s=100, c=color, label=f"Cluster {i + 1}",
            )
        ax.set_title("Hierarchical Customer Clusters")
        ax.set_xlabel("Annual Income (k$)")
        ax.set_ylabel("Spending Score (1–100)")
        ax.legend()
        fig.savefig(
            output_dir / "hierarchical_clusters.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)


def main() -> None:
    seg = CustomerSegmentation()
    print("[1/4] Loading data...")
    seg.load_data()
    print("[2/4] Running K-Means clustering...")
    y_kmeans, _ = seg.run_kmeans()
    print("[3/4] Running hierarchical clustering...")
    y_hc = seg.run_hierarchical()
    print(f"K-Means cluster sizes      : {np.bincount(y_kmeans)}")
    print(f"Hierarchical cluster sizes : {np.bincount(y_hc)}")
    print("[4/4] Saving plots (elbow, dendrogram, cluster scatter)...")
    seg.save_plots()
    print("Plots saved to plots/")


if __name__ == "__main__":
    main()
