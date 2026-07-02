import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


class DataPreprocessor:
    DATA_FILE: str = "Data.csv"

    def __init__(self, data_path: str = DATA_FILE) -> None:
        self.data_path = Path(data_path)
        self.scaler = StandardScaler()

    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(self.data_path)
        return df.iloc[:, :-1].values, df.iloc[:, -1].values

    def handle_missing(self, X: np.ndarray) -> np.ndarray:
        imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        X[:, 1:3] = imputer.fit_transform(X[:, 1:3])
        return X

    def encode_features(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        ct = ColumnTransformer(
            transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough"
        )
        X = np.array(ct.fit_transform(X))
        y = LabelEncoder().fit_transform(y)
        return X, y

    def split_and_scale(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        X_train[:, 3:] = self.scaler.fit_transform(X_train[:, 3:])
        X_test[:, 3:] = self.scaler.transform(X_test[:, 3:])
        return X_train, X_test, y_train, y_test

    def run(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X, y = self.load_data()
        X = self.handle_missing(X)
        X, y = self.encode_features(X, y)
        return self.split_and_scale(X, y)


def main() -> None:
    preprocessor = DataPreprocessor()
    print("[1/4] Loading data...")
    X, y = preprocessor.load_data()
    print("[2/4] Handling missing values...")
    X = preprocessor.handle_missing(X)
    print("[3/4] Encoding features...")
    X, y = preprocessor.encode_features(X, y)
    print("[4/4] Splitting and scaling...")
    X_train, X_test, y_train, y_test = preprocessor.split_and_scale(X, y)
    print(f"Training set shape : {X_train.shape}")
    print(f"Test set shape     : {X_test.shape}")
    print("Done.")


if __name__ == "__main__":
    main()
