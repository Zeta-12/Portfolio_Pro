import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


class _ChurnNet(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 6),
            nn.ReLU(),
            nn.Linear(6, 6),
            nn.ReLU(),
            nn.Linear(6, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ChurnPredictor:
    DATA_FILE: str = "Churn_Modelling.csv"
    BATCH_SIZE: int = 32
    EPOCHS: int = 100
    LR: float = 0.001

    def __init__(self, data_path: str = DATA_FILE) -> None:
        self.data_path = Path(data_path)
        self.scaler = StandardScaler()
        self.model: _ChurnNet | None = None
        self._ct: ColumnTransformer | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(self.data_path)
        return df.iloc[:, 3:-1].values, df.iloc[:, -1].values

    def preprocess(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X[:, 2] = LabelEncoder().fit_transform(X[:, 2])
        self._ct = ColumnTransformer(
            transformers=[("encoder", OneHotEncoder(), [1])], remainder="passthrough"
        )
        X = np.array(self._ct.fit_transform(X))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        return X_train, X_test, y_train, y_test

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model = _ChurnNet(X_train.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LR)
        criterion = nn.BCELoss()

        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.BATCH_SIZE, shuffle=True)

        self.model.train()
        for _ in tqdm(range(self.EPOCHS), desc="Training", unit="epoch"):
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                criterion(self.model(X_batch), y_batch).backward()
                optimizer.step()

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            preds = (self.model(X_t).cpu().numpy().ravel() > 0.5).astype(int)
        return {
            "accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "confusion_matrix": confusion_matrix(y_test, preds),
        }

    def predict_single(self, features: list) -> bool:
        self.model.eval()
        X = self.scaler.transform(np.array([features]))
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            return bool(self.model(X_t).item() > 0.5)

    def save_model(self, path: str = "churn_model.pt") -> None:
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str = "churn_model.pt") -> None:
        state = torch.load(path, map_location=self.device, weights_only=True)
        input_dim = state["net.0.weight"].shape[1]
        self.model = _ChurnNet(input_dim).to(self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def save_plots(
        self, cm: np.ndarray, output_dir: Path = Path("plots")
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Retained", "Churned"]
        ).plot(ax=ax)
        ax.set_title("Churn Prediction — Confusion Matrix")
        fig.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bank churn prediction — ANN (PyTorch)")
    parser.add_argument("--epochs", type=int, default=ChurnPredictor.EPOCHS,
                        help=f"training epochs (default: {ChurnPredictor.EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=ChurnPredictor.BATCH_SIZE,
                        help=f"batch size (default: {ChurnPredictor.BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=ChurnPredictor.LR,
                        help=f"learning rate (default: {ChurnPredictor.LR})")
    parser.add_argument("--save-model", type=str, default="churn_model.pt",
                        help="path to save trained weights (default: churn_model.pt)")
    args = parser.parse_args()

    predictor = ChurnPredictor()
    predictor.EPOCHS = args.epochs
    predictor.BATCH_SIZE = args.batch_size
    predictor.LR = args.lr

    print("[1/5] Loading data (10,000 customers)...")
    X, y = predictor.load_data()
    print("[2/5] Preprocessing (encoding, scaling, split)...")
    X_train, X_test, y_train, y_test = predictor.preprocess(X, y)
    print(f"[3/5] Training ANN for {args.epochs} epochs on {predictor.device}...")
    predictor.train(X_train, y_train)
    predictor.save_model(args.save_model)
    print("[4/5] Evaluating...")
    metrics = predictor.evaluate(X_test, y_test)
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
    print("[5/5] Saving plots...")
    predictor.save_plots(metrics["confusion_matrix"])
    print(f"Plots saved to plots/  |  Model saved to {args.save_model}")


if __name__ == "__main__":
    main()
