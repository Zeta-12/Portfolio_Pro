import numpy as np
import pandas as pd
from pathlib import Path
from apyori import apriori


class BasketAnalyzer:
    DATA_FILE: str = "Market_Basket_Optimisation.csv"
    N_TRANSACTIONS: int = 7501
    N_ITEMS: int = 20

    def __init__(self, data_path: str = DATA_FILE) -> None:
        self.data_path = Path(data_path)
        self.transactions: list[list[str]] = []

    def load_transactions(self) -> None:
        df = pd.read_csv(self.data_path, header=None)
        self.transactions = [
            [str(df.values[i, j]) for j in range(self.N_ITEMS)]
            for i in range(self.N_TRANSACTIONS)
        ]

    @staticmethod
    def _parse_rules(results: list) -> pd.DataFrame:
        records = []
        for r in results:
            for stat in r.ordered_statistics:
                lhs = list(stat.items_base)
                rhs = list(stat.items_add)
                if not lhs or not rhs:
                    continue
                records.append(
                    {
                        "antecedent": lhs[0],
                        "consequent": rhs[0],
                        "support": round(r.support, 4),
                        "confidence": round(stat.confidence, 4),
                        "lift": round(stat.lift, 4),
                    }
                )
        return pd.DataFrame(records)

    def run_apriori(
        self,
        min_support: float = 0.003,
        min_confidence: float = 0.2,
        min_lift: float = 3.0,
    ) -> pd.DataFrame:
        rules = apriori(
            transactions=self.transactions,
            min_support=min_support,
            min_confidence=min_confidence,
            min_lift=min_lift,
            min_length=2,
            max_length=2,
        )
        return self._parse_rules(list(rules)).sort_values("lift", ascending=False)

    def run_eclat(
        self,
        min_support: float = 0.003,
        min_confidence: float = 0.2,
        min_lift: float = 3.0,
    ) -> pd.DataFrame:
        rules = apriori(
            transactions=self.transactions,
            min_support=min_support,
            min_confidence=min_confidence,
            min_lift=min_lift,
            min_length=2,
            max_length=2,
        )
        df = self._parse_rules(list(rules))
        return df[["antecedent", "consequent", "support"]].sort_values(
            "support", ascending=False
        )

    def save_results(self, output_dir: Path = Path("results")) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.run_apriori().head(10).to_csv(
            output_dir / "apriori_top10.csv", index=False
        )
        self.run_eclat().head(10).to_csv(
            output_dir / "eclat_top10.csv", index=False
        )


def main() -> None:
    analyzer = BasketAnalyzer()
    print("[1/4] Loading 7,501 transactions...")
    analyzer.load_transactions()
    print("[2/4] Running Apriori algorithm...")
    apriori_df = analyzer.run_apriori()
    print("=== Apriori — Top 10 Rules by Lift ===")
    print(apriori_df.head(10).to_string(index=False))
    print("[3/4] Running ECLAT...")
    eclat_df = analyzer.run_eclat()
    print("\n=== ECLAT — Top 10 Item Sets by Support ===")
    print(eclat_df.head(10).to_string(index=False))
    print("[4/4] Saving results to CSV...")
    analyzer.save_results()
    print("\nResults saved to results/")


if __name__ == "__main__":
    main()
