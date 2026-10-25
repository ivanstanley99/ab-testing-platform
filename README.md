# ab-testing-platform

Simple A/B test data generator and analytics package.

## Quick start

```python
from src.generator import SyntheticDataGenerator
from src.statistics import ABStatistics

rng = SyntheticDataGenerator(seed=42)
df = rng.generate_user_data(num_users=10000, effect_size=0.05)

stats = ABStatistics(df)
print(stats.t_test())
print(stats.bootstrap())
```

The generator creates a synthetic dataset with users split into control (`A`) and treatment (`B`) groups, base spend history (`pre_spends`), and post-experiment spend (`post_spends`). The statistics class provides a Welch t-test and bootstrap confidence interval for the mean lift.
