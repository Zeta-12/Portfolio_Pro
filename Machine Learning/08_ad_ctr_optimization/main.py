import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class BanditOptimizer:
    DATA_FILE: str = "Ads_CTR_Optimisation.csv"

    def __init__(self, data_path: str = DATA_FILE) -> None:
        self.data_path = Path(data_path)
        self.dataset: pd.DataFrame | None = None

    def load_data(self) -> None:
        self.dataset = pd.read_csv(self.data_path)

    def run_ucb(self) -> tuple[list[int], int]:
        N, d = self.dataset.shape
        selections = [0] * d
        rewards_sum = [0] * d
        ads_selected: list[int] = []
        total_reward = 0

        for n in range(N):
            ad = max(
                range(d),
                key=lambda i: (
                    rewards_sum[i] / selections[i]
                    + math.sqrt(1.5 * math.log(n + 1) / selections[i])
                    if selections[i] > 0
                    else float("inf")
                ),
            )
            ads_selected.append(ad)
            selections[ad] += 1
            reward = int(self.dataset.values[n, ad])
            rewards_sum[ad] += reward
            total_reward += reward

        return ads_selected, total_reward

    def run_thompson(self) -> tuple[list[int], int]:
        N, d = self.dataset.shape
        wins = [0] * d
        losses = [0] * d
        ads_selected: list[int] = []
        total_reward = 0

        for n in range(N):
            ad = max(
                range(d),
                key=lambda i: random.betavariate(wins[i] + 1, losses[i] + 1),
            )
            ads_selected.append(ad)
            reward = int(self.dataset.values[n, ad])
            if reward == 1:
                wins[ad] += 1
            else:
                losses[ad] += 1
            total_reward += reward

        return ads_selected, total_reward

    def save_plots(self, output_dir: Path = Path("plots")) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        ucb_selections, ucb_reward = self.run_ucb()
        ts_selections, ts_reward = self.run_thompson()
        n_ads = self.dataset.shape[1]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, selections, label, reward in [
            (axes[0], ucb_selections, "UCB", ucb_reward),
            (axes[1], ts_selections, "Thompson Sampling", ts_reward),
        ]:
            ax.hist(selections, bins=n_ads, edgecolor="black", rwidth=0.8)
            ax.set_title(f"{label}  (total reward: {reward:,})")
            ax.set_xlabel("Ad Index")
            ax.set_ylabel("Times Selected")
            ax.set_xticks(range(n_ads))
        fig.tight_layout()
        fig.savefig(
            output_dir / "bandit_comparison.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)


def main() -> None:
    optimizer = BanditOptimizer()
    print("[1/4] Loading data (10,000 rounds, 10 ads)...")
    optimizer.load_data()
    print("[2/4] Running UCB algorithm...")
    ucb_selections, ucb_reward = optimizer.run_ucb()
    print("[3/4] Running Thompson Sampling...")
    ts_selections, ts_reward = optimizer.run_thompson()
    best_ad_ucb = max(range(len(set(ucb_selections))), key=ucb_selections.count)
    best_ad_ts = max(range(len(set(ts_selections))), key=ts_selections.count)
    print(f"UCB               — total reward: {ucb_reward:,}  |  best ad: #{best_ad_ucb + 1}")
    print(f"Thompson Sampling — total reward: {ts_reward:,}  |  best ad: #{best_ad_ts + 1}")
    print("[4/4] Saving plots...")
    optimizer.save_plots()
    print("Plots saved to plots/")


if __name__ == "__main__":
    main()
