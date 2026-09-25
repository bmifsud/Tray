
## 2024-05-18 - Fast Pandas DataFrame Creation from Pydantic Models
**Learning:** `t.model_dump()` in Pydantic v2 introduces significant overhead (~3x slower) compared to `t.__dict__` when converting large lists of flat Pydantic models to Pandas DataFrames. This is due to recursive serialization checks within `model_dump()`.
**Action:** For performance-critical bulk operations where Pydantic models contain only flat data types (like strings, floats, ints), use `__dict__` or explicitly map fields instead of using `model_dump()` to bypass serialization overhead.
