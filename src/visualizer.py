import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    from src.generator import SyntheticDataGenerator
except ImportError:  # pragma: no cover
    from generator import SyntheticDataGenerator


class ABVisualizer:
    """Класс для визуализации распределений и результатов A/B-теста."""

    def __init__(self, data: pd.DataFrame):
        self.data = data
        sns.set_theme(style="whitegrid")

    def plot_distributions(self, save_path: str = "reports/distributions.png"):
        """Строит графики плотности распределения трат во время теста для групп A и B."""
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=self.data,
            x="post_spends",
            hue="group",
            element="step",
            stat="density",
            common_norm=False,
            palette={"A": "#4C72B0", "B": "#DD8452"},
            bins=50,
            alpha=0.4,
        )
        plt.title("Распределение трат во время теста (Post-spends)", fontsize=14, fontweight="bold")
        plt.xlabel("Траты в период теста", fontsize=12)
        plt.ylabel("Плотность вероятности", fontsize=12)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[Успех] График распределения сохранен в: {save_path}")

    def plot_pre_vs_post(self, save_path: str = "reports/pre_vs_post.png"):
        """Строит scatter plot зависимости исторических трат от трат в тесте (для CUPED)."""
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=self.data,
            x="pre_spends",
            y="post_spends",
            hue="group",
            palette={"A": "#4C72B0", "B": "#DD8452"},
            alpha=0.5,
            s=35,
        )
        plt.title("Связь между историческими и тестовыми тратами (CUPED Base)", fontsize=14, fontweight="bold")
        plt.xlabel("Траты до эксперимента (pre_spends)", fontsize=12)
        plt.ylabel("Траты во время эксперимента (post_spends)", fontsize=12)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[Успех] График связи pre/post сохранен в: {save_path}")


if __name__ == "__main__":
    generator = SyntheticDataGenerator(seed=42)
    df = generator.generate_user_data(num_users=10000, effect_size=0.05)

    visualizer = ABVisualizer(df)
    visualizer.plot_distributions()
    visualizer.plot_pre_vs_post()