## 2024-05-18 - Pandas `where` vs `.clip()`
**Learning:** In Pandas, using `.clip(lower=0)` and `.clip(upper=0)` combined with inversion is significantly faster than using `.where(delta > 0, 0)` for calculating gains and losses in RSI features due to avoiding element-wise conditional evaluation overhead. Furthermore, `.diff().fillna(0)` should be used to perfectly mimic `.where(delta > 0, 0)`'s behavior on initial NaNs.
**Action:** Replace `delta.where(delta > 0, 0)` with `delta.fillna(0).clip(lower=0)` for performance optimizations when separating positive and negative returns.

## 2024-05-18 - Memoizing rolling evaluations
**Learning:** Calling `.rolling()` multiple times with the same window and min_periods on the same column creates significant overhead.
**Action:** Always memoize `.rolling()` calls into a variable (e.g. `vol_roll = vol.rolling(20, min_periods=5)`) before applying aggregation methods like `.mean()` and `.std()`.
