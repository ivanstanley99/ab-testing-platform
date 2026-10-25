import numpy as np
import pandas as pd


class SyntheticDataGenerator:
    """Генератор синтетических данных для моделирования A/B-тестов.

    Генерирует базовые траты пользователей и их реакцию в контрольной и
    тестовой группах с воспроизводимым сидом.
    """

    def __init__(self, seed: int):
        """Фиксация seed для воспроизводимости результатов."""
        if not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an integer")
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def generate_user_data(
        self,
        num_users: int = 10000,
        effect_size: float = 0.05,
        noise_level: float = 0.2,
    ) -> pd.DataFrame:
        """
        :param num_users: количество пользователей в тесте
        :param effect_size: относительный прирост метрики в группе B (0.05 =
            +5%)
        :param noise_level: уровень случайного шума в данных
        :return: DataFrame user_id, group, pre_spends, post_spends
        """
        if not isinstance(num_users, (int, np.integer)) or num_users <= 0:
            raise ValueError("num_users must be a positive integer")

        for name, value in {"effect_size": effect_size, "noise_level": noise_level}.items():
            if not isinstance(value, (int, float, np.integer, np.floating)):
                raise TypeError(f"{name} must be numeric")

        if noise_level < 0:
            raise ValueError("noise_level must be non-negative")

        groups = self.rng.choice(["A", "B"], size=num_users, p=[0.5, 0.5])

        shape, scale = 0.5, 500
        pre_spends = self.rng.lognormal(mean=np.log(scale), sigma=shape, size=num_users)

        correlation_factor = 0.85
        random_noise = self.rng.normal(0, noise_level * scale, size=num_users)

        post_spends = (
            correlation_factor * pre_spends
            + (1 - correlation_factor) * scale
            + random_noise
        )

        treatment_mask = groups == "B"
        post_spends[treatment_mask] *= 1 + effect_size

        post_spends = np.maximum(post_spends, 0)
        pre_spends = np.maximum(pre_spends, 0)

        return pd.DataFrame(
            {
                "user_id": np.arange(1, num_users + 1),
                "group": groups,
                "pre_spends": np.round(pre_spends, 2),
                "post_spends": np.round(post_spends, 2),
            }
        )


if __name__ == "__main__":
    try:
        from src.generator import SyntheticDataGenerator
    except ImportError:  # pragma: no cover
        from generator import SyntheticDataGenerator

    generator = SyntheticDataGenerator(seed=42)
    data = generator.generate_user_data(num_users=1000, effect_size=0.05)
    print(data.head())
    print("\n", data.groupby("group")["post_spends"].mean())