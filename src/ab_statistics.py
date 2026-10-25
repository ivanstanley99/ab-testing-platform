import numpy as np
import pandas as pd
from scipy import stats
from generator import SyntheticDataGenerator


class ABStatistics:
    """Класс для проведения статистического анализа результатов A/B-теста."""

    def __init__(self, data: pd.DataFrame):
        """Принимает на вход DataFrame с колонками group и post_spends."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        required_columns = {"group", "post_spends"}
        missing = required_columns - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        self.data = data.copy()
        self.control = pd.to_numeric(
            self.data.loc[self.data["group"] == "A", "post_spends"], errors="coerce"
        ).dropna()
        self.treatment = pd.to_numeric(
            self.data.loc[self.data["group"] == "B", "post_spends"], errors="coerce"
        ).dropna()

        if self.control.empty or self.treatment.empty:
            raise ValueError("DataFrame must contain both control (A) and treatment (B) groups")

    def t_test(self) -> dict:
        """Классический t-test Стьюдента для проверки равенства средних."""
        if len(self.control) < 2 or len(self.treatment) < 2:
            raise ValueError("Each group must contain at least two observations for a t-test")

        t_stat, p_value = stats.ttest_ind(self.treatment, self.control, equal_var=False)
        return {
            "method": "Welch's t-test",
            "t_stat": float(round(float(t_stat), 4)),
            "p_value": float(round(float(p_value), 4)),
            "significant_05": bool(p_value < 0.05),
        }

    def bootstrap(self, num_samples: int = 2000) -> dict:
        """Бутстрап для оценки разницы средних и построения доверительного интервала."""
        if not isinstance(num_samples, (int, np.integer)) or num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")

        diffs = []
        n_ctrl = len(self.control)
        n_trt = len(self.treatment)

        for _ in range(num_samples):
            sample_ctrl = np.random.choice(self.control, size=n_ctrl, replace=True)
            sample_trt = np.random.choice(self.treatment, size=n_trt, replace=True)
            diffs.append(sample_trt.mean() - sample_ctrl.mean())

        diffs = np.array(diffs, dtype=float)
        ci_lower = np.percentile(diffs, 2.5)
        ci_upper = np.percentile(diffs, 97.5)

        p_value = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
        p_value = min(float(p_value), 1.0)

        return {
            "method": "Bootstrap",
            "mean_diff": float(round(float(np.mean(diffs)), 4)),
            "ci_lower": float(round(float(ci_lower), 4)),
            "ci_upper": float(round(float(ci_upper), 4)),
            "p_value": float(round(p_value, 4)),
            "significant_05": bool(p_value < 0.05),
        }
    
def apply_cuped(df: pd.DataFrame, y_col: str = 'post_spends', x_col: str = 'pre_spends') -> pd.DataFrame:
    """
    Применяет метод CUPED для снижения дисперсии метрики.
    """
    df_cuped = df.copy()
    y = df_cuped[y_col]
    x = df_cuped[x_col]
    
    # Считаем коэффициент theta = Cov(y, x) / Var(x)
    cov = np.cov(y, x)[0, 1]
    var_x = np.var(x, ddof=1)
    theta = cov / var_x
    
    # Создаем скорректированную метрику
    df_cuped[f'{y_col}_cuped'] = y - theta * (x - x.mean())
    
    return df_cuped
        
    
if __name__ == "__main__":
    # 1. Генерируем синтетические данные
    generator = SyntheticDataGenerator(seed=42)
    df = generator.generate_user_data(num_users=10000, effect_size=0.05)
    
    print("=== СРАВНЕНИЕ ДО И ПОСЛЕ CUPED ===\n")
    
    # 2. Анализ сырых данных с помощью нашего класса ABStatistics
    print("--- Сырые данные (Raw) ---")
    stats_raw = ABStatistics(df)
    print("t-test:", stats_raw.t_test())
    print("Bootstrap:", stats_raw.bootstrap())
    
    # 3. Применяем CUPED
    df_cuped = apply_cuped(df, y_col='post_spends', x_col='pre_spends')
    
    # Создаем датасет для нашего класса, подменяя post_spends на CUPED-версию
    df_cuped_for_stats = df_cuped[['group']].copy()
    df_cuped_for_stats['post_spends'] = df_cuped['post_spends_cuped']
    
    print("\n--- Данные после CUPED ---")
    stats_cuped = ABStatistics(df_cuped_for_stats)
    print("t-test:", stats_cuped.t_test())
    print("Bootstrap:", stats_cuped.bootstrap())
    
    # 4. Сравнение дисперсий
    var_raw = df['post_spends'].var()
    var_cuped = df_cuped['post_spends_cuped'].var()
    variance_reduction = (1 - var_cuped / var_raw) * 100
    print(f"\n📉 Снижение дисперсии: {variance_reduction:.2f}%")