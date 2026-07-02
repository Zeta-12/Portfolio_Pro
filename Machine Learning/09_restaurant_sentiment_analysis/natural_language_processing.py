import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


class SentimentAnalyzer:
    DATA_FILE: str = "Restaurant_Reviews.tsv"
    MAX_FEATURES: int = 1500

    def __init__(self, data_path: str = DATA_FILE) -> None:
        self.data_path = Path(data_path)
        self.vectorizer: CountVectorizer | None = None
        self.model: GaussianNB | None = None

    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.data_path, delimiter="\t", quoting=3)

    def clean_corpus(self, reviews: pd.Series) -> list[str]:
        nltk.download("stopwords", quiet=True)
        ps = PorterStemmer()
        stop_words = set(stopwords.words("english")) - {"not"}
        corpus = []
        for review in reviews:
            text = re.sub("[^a-zA-Z]", " ", review).lower().split()
            text = " ".join(ps.stem(w) for w in text if w not in stop_words)
            corpus.append(text)
        return corpus

    def vectorize(self, corpus: list[str]) -> np.ndarray:
        self.vectorizer = CountVectorizer(max_features=self.MAX_FEATURES)
        return self.vectorizer.fit_transform(corpus).toarray()

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model = GaussianNB()
        self.model.fit(X_train, y_train)

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> dict:
        y_pred = self.model.predict(X_test)
        return {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }

    def save_plots(
        self, cm: np.ndarray, output_dir: Path = Path("plots")
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Negative", "Positive"]
        ).plot(ax=ax)
        ax.set_title("Sentiment Analysis — Confusion Matrix")
        fig.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    analyzer = SentimentAnalyzer()
    print("[1/5] Loading reviews...")
    df = analyzer.load_data()
    print(f"[2/5] Cleaning corpus ({len(df)} reviews)...")
    corpus = analyzer.clean_corpus(df["Review"])
    print("[3/5] Vectorizing (Bag of Words)...")
    X = analyzer.vectorize(corpus)
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    print("[4/5] Training Naive Bayes classifier...")
    analyzer.train(X_train, y_train)
    metrics = analyzer.evaluate(X_test, y_test)
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
    print("[5/5] Saving plots...")
    analyzer.save_plots(metrics["confusion_matrix"])
    print("Plots saved to plots/")


if __name__ == "__main__":
    main()
