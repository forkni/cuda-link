# Python Style Guide - Quick Reference

**Purpose**: Quick reference for Python coding standards. See `PYTHON_STYLE_GUIDE.md` for comprehensive documentation.

**Standards**: PEP 8 + Black (88 chars) + Google/NumPy docstrings + Type hints required

---

## Golden Rules (Top 20)

1. **Specify exception types**: `except ValueError:` not `except:`
2. **Use type hints**: All public functions need parameter/return types
3. **No mutable defaults**: `def func(items=None)` not `def func(items=[])`
4. **Use context managers**: `with open(...)` for all resources
5. **Check None explicitly**: `if x is None:` not `if not x:`
6. **No wildcard imports**: `from x import y` not `from x import *`
7. **Use isinstance()**: Not `type(x) == list`
8. **Document array shapes**: In ML code (shape, dtype)
9. **Use f-strings**: `f"Hello {name}"` not `"Hello " + name`
10. **Separate I/O from logic**: Pure functions for testing
11. **snake_case functions**: `def process_data()`
12. **CamelCase classes**: `class DataProcessor`
13. **UPPER_CASE constants**: `MAX_SIZE = 100`
14. **Double quotes**: `"string"` for consistency (Black)
15. **88 char lines**: Black standard (not 79)
16. **Group imports**: stdlib → third-party → local
17. **Trailing commas**: In multi-line structures
18. **Chain exceptions**: `raise ... from e`
19. **Await coroutines**: Always `await` in async
20. **Parameterized queries**: Never concatenate SQL

**⚠️ When to Break the Rules** (PEP 8 "Foolish Consistency"):

- Matching existing code style (backwards compatibility)
- Compliance reduces readability
- The code predates the rule

**Remember**: Readability FIRST, compliance SECOND.

---

## Naming At-a-Glance

| Entity | Convention | Example |
|--------|------------|---------|
| **Class** | CamelCase | `class UserAccount` |
| **Function** | snake_case | `def calculate_total()` |
| **Variable** | snake_case | `user_count = 10` |
| **Constant** | UPPER_CASE | `MAX_RETRIES = 3` |
| **Protected** | _leading | `self._internal_state` |
| **Private** | __double | `self.__private_data` |
| **Module** | lowercase | `data_utils.py` |
| **Package** | lowercase | `mypackage/` |

---

## Import Template

```python
"""Module docstring explaining purpose."""
# 1. Standard library (alphabetical)
import os
import sys
from pathlib import Path

# 2. Third-party libraries (alphabetical)
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score

# 3. Local application (alphabetical)
from myproject.models import BaseModel
from myproject.utils import helper_func

# Module-level constants
MAX_ITERATIONS = 100
DEFAULT_TIMEOUT = 30
```

---

## Docstring Templates

### Google Style (General Python)

```python
def fetch_data(url: str, timeout: int = 30) -> dict:
    """Fetches data from a URL.

    Args:
        url: The URL to fetch. Must be HTTP/HTTPS.
        timeout: Max wait time in seconds. Defaults to 30.

    Returns:
        Dictionary with parsed JSON response.

    Raises:
        requests.ConnectionError: If connection fails.
        requests.Timeout: If request times out.

    Examples:
        >>> data = fetch_data("https://api.example.com")
        >>> print(data["status"])
        "success"
    """
    pass
```

### NumPy Style (ML/Scientific)

```python
def matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiplies two matrices.

    Parameters
    ----------
    a : ndarray, shape (n, m)
        First matrix with dtype float32.
    b : ndarray, shape (m, k)
        Second matrix with dtype float32.

    Returns
    -------
    result : ndarray, shape (n, k)
        Product of a and b.

    Raises
    ------
    ValueError
        If dimensions incompatible.

    See Also
    --------
    numpy.matmul : Underlying implementation

    Examples
    --------
    >>> a = np.array([[1, 2]], dtype=np.float32)
    >>> b = np.array([[3], [4]], dtype=np.float32)
    >>> result = matrix_multiply(a, b)
    >>> result.shape
    (1, 1)
    """
    pass
```

---

## Type Hint Cheat Sheet

```python
# Basic types
def greet(name: str, age: int) -> str:
    pass

# Collections (Python 3.9+)
names: list[str] = ["Alice", "Bob"]
scores: dict[str, int] = {"Alice": 95}
unique_ids: set[int] = {1, 2, 3}
coords: tuple[float, float] = (10.0, 20.0)

# Optional (can be None)
def find(key: str) -> str | None:
    pass

# Union (multiple types)
def process(value: int | float | str) -> str:
    pass

# Callable (function type)
from typing import Callable
def apply(func: Callable[[int], int], x: int) -> int:
    pass

# Generic
from typing import TypeVar
T = TypeVar("T")

def first(items: list[T]) -> T | None:
    return items[0] if items else None

# ML/NumPy
import numpy as np
from numpy.typing import NDArray

def transform(data: NDArray[np.float32]) -> NDArray[np.float32]:
    pass

# PyTorch
import torch
from torch import Tensor

def forward(x: Tensor, weight: Tensor) -> Tensor:
    pass
```

---

## Common Anti-pattern Fixes

| ❌ Anti-pattern | ✅ Fix |
|----------------|--------|
| `except:` | `except ValueError as e:` |
| `def func(items=[])` | `def func(items=None)` then `items = items or []` |
| `if not value:` (for None) | `if value is None:` |
| `file = open(...); file.close()` | `with open(...) as file:` |
| `"Hello " + name` | `f"Hello {name}"` |
| `from x import *` | `from x import specific_func` |
| `type(x) == list` | `isinstance(x, list)` |
| `raise ValueError("msg")` (in except) | `raise ValueError("msg") from e` |
| Manual GPU cleanup | `torch.cuda.empty_cache()` every N batches |
| Tight I/O coupling | Separate `load_data()` from `process_data()` |
| `[print(x) for x in items]` | Use `for x in items: print(x)` |
| Implicit None return | Explicit `return None` in all branches |
| `if x: return y` | Separate lines: `if x:\n    return y` |
| `getattr(obj, field)` loops | Explicit if/elif for known fields |
| `from . import utils` | `from myproject import utils` (absolute) |

---

## Pre-flight Checklist

Before running Python code:

- [ ] All public functions have type hints
- [ ] No bare `except:` clauses
- [ ] No mutable default arguments (`[]`, `{}`)
- [ ] All files opened with `with` statement
- [ ] None checks use `is None`, not truthiness
- [ ] No wildcard imports (`from x import *`)
- [ ] Variables use `snake_case`
- [ ] Classes use `CamelCase`
- [ ] Constants use `UPPER_CASE`
- [ ] Line length ≤ 88 characters
- [ ] Imports grouped: stdlib → third-party → local
- [ ] Docstrings present for all public functions
- [ ] Array shapes documented (for ML code)
- [ ] f-strings used for formatting
- [ ] Resources managed with context managers

---

## Error Quick Reference

| Error | Likely Cause | Quick Fix |
|-------|--------------|-----------|
| **NameError** | Variable not defined | Check spelling, add import |
| **TypeError** | Wrong type | Convert type: `int(x)`, `str(x)` |
| **AttributeError** | Method on None | Add `if obj is not None:` |
| **KeyError** | Missing dict key | Use `dict.get(key, default)` |
| **IndexError** | List out of range | Check `if idx < len(list):` |
| **ValueError** | Invalid value | Validate input first |
| **ImportError** | Module not found | `pip install package_name` |
| **IndentationError** | Mixed tabs/spaces | Use 4 spaces everywhere |
| **SyntaxError** | Invalid syntax | Check matching `()`, `[]`, `{}` |

---

## ML/Data Science Quick Patterns

### Standard Imports

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
```

### Array Shape Documentation

```python
def process_batch(images: np.ndarray) -> np.ndarray:
    """Process images.

    Parameters
    ----------
    images : ndarray, shape (batch_size, height, width, channels)
        Input images with dtype uint8.

    Returns
    -------
    processed : ndarray, shape (batch_size, height, width, channels)
        Processed images with dtype float32.
    """
    pass
```

### GPU Memory Management

```python
# In training loop
for batch_idx, (data, target) in enumerate(dataloader):
    data, target = data.to(device), target.to(device)

    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    # Clear cache periodically
    if batch_idx % 10 == 0:
        torch.cuda.empty_cache()

    # Delete large tensors if memory constrained
    del data, target, output, loss
```

### Configuration Dataclass

```python
from dataclasses import dataclass, field

@dataclass
class Config:
    """Training configuration."""
    model_name: str
    dataset_path: str
    batch_size: int = 32
    learning_rate: float = 1e-3
    device: str = "cuda"
    optimizer_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
```

---

## Ruff Configuration Snippet

**pyproject.toml**:

```toml
[tool.black]
line-length = 88
target-version = ["py310"]

[tool.ruff]
line-length = 88
target-version = "py310"
select = ["E", "W", "F", "I", "N", "UP", "B", "C4", "SIM"]
ignore = ["E501"]  # Line length (Black handles)

[tool.ruff.per-file-ignores]
"__init__.py" = ["F401"]  # Unused imports OK

[tool.mypy]
python_version = "3.10"
disallow_untyped_defs = true
warn_return_any = true
```

**Run checks**:

```bash
# Format
black script.py

# Lint
ruff check script.py

# Auto-fix
ruff check --fix script.py

# Type check
mypy script.py
```

---

## Critical Safety Checklist

**Exception Handling**:

```python
# ✅ Always specify
try:
    risky_operation()
except (ValueError, KeyError) as e:
    logger.error(f"Failed: {e}")
    raise
```

**Resource Management**:

```python
# ✅ Always use with
with open("file.txt") as f:
    data = f.read()
```

**Mutable Defaults**:

```python
# ✅ Use None pattern
def add_item(item: int, items: list[int] | None = None) -> list[int]:
    if items is None:
        items = []
    items.append(item)
    return items
```

**None Checking**:

```python
# ✅ Explicit identity check
if result is None:
    result = fetch_default()
```

**Import Organization**:

```python
# ✅ Explicit, grouped
from typing import Optional
from mymodule import specific_function
```

---

## Code Layout Quick Rules

- **Indentation**: 4 spaces (no tabs)
- **Line length**: 120 characters maximum (this project override; see PYTHON_STYLE_GUIDE.md Section 2.2)
- **Blank lines**: 2 before classes/functions, 1 between methods
- **Quotes**: Double quotes `"..."` (Black standard)
- **Trailing commas**: Use in multi-line structures
- **Continuation**: Use parentheses, not backslash

```python
# ✅ Good layout
def long_function_name(
    parameter_one: str,
    parameter_two: int,
    parameter_three: bool = False,
) -> dict[str, any]:
    """Function with good layout."""
    result = {
        "key1": "value1",
        "key2": "value2",
    }
    return result
```

---

## See Also

- **Comprehensive Guide**: `PYTHON_STYLE_GUIDE.md` - Full documentation with examples
- **Other Guides**: `BASH_STYLE_GUIDE.md`, `BATCH_STYLE_GUIDE.md`
- **Project Standards**: `CLAUDE.md` - Project-specific conventions

---

**Last Updated**: 2026-01-05
**Character Count**: ~10,500 characters
**For Details**: See `PYTHON_STYLE_GUIDE.md`
