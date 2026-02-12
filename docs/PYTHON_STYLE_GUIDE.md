# Python Coding Safety Guide for Claude Code

**Purpose**: Comprehensive Python coding patterns optimized for error prevention, maintainability, and ML/data science workflows.

**Target Environment**: Python 3.10+, Black/Ruff formatting, type hints required.

**Philosophy**: Safety through explicitness. Write code that's hard to misuse and easy to verify.

**Sources**:

- [PEP 8 - Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [The Hitchhiker's Guide to Python](https://docs.python-guide.org/writing/style/)
- [NumPy Documentation Standard](https://numpydoc.readthedocs.io/en/latest/format.html)
- [The Black Code Style](https://black.readthedocs.io/en/stable/the_black_code_style/)

---

## Table of Contents

1. [Critical Safety Patterns](#1-critical-safety-patterns)
2. [Code Layout & Formatting](#2-code-layout--formatting)
3. [Naming Conventions](#3-naming-conventions)
4. [Docstrings (Hybrid Format)](#4-docstrings-hybrid-format)
5. [Type Hints](#5-type-hints)
6. [ML/Data Science Patterns](#6-mldata-science-patterns)
7. [Common Anti-patterns & Fixes](#7-common-anti-patterns--fixes)
8. [Error Recovery Reference](#8-error-recovery-reference)
9. [Quick Reference Tables](#9-quick-reference-tables)
10. [Integration with Claude Code](#10-integration-with-claude-code)
11. [Tooling Configuration](#11-tooling-configuration)
12. [Summary: Golden Rules](#12-summary-golden-rules)

---

## 1. Critical Safety Patterns

### 1.1 Error Handling - Specific Exception Types

**❌ DANGEROUS (catches everything)**:

```python
try:
    result = risky_operation()
except:  # Bare except catches KeyboardInterrupt, SystemExit!
    print("Something went wrong")
```

**✅ SAFE (specific exceptions)**:

```python
try:
    result = risky_operation()
except (ValueError, KeyError) as e:
    logger.error(f"Operation failed: {e}")
    raise
```

**Rule**: Never use bare `except:`. Always specify exception types.

**When to catch what**:

```python
# File operations
try:
    with open(file_path) as f:
        data = f.read()
except FileNotFoundError:
    # Handle missing file
    data = default_data()
except PermissionError:
    # Handle permission denied
    logger.error(f"Cannot read {file_path}")

# Network operations
try:
    response = requests.get(url)
except requests.ConnectionError:
    # Handle network failure
    retry_with_backoff()
except requests.Timeout:
    # Handle timeout
    logger.warning(f"Request to {url} timed out")
```

### 1.2 Type Hints - Required for Function Signatures

**❌ MISSING TYPE HINTS**:

```python
def process_data(items, threshold):
    return [x for x in items if x > threshold]
```

**✅ WITH TYPE HINTS**:

```python
def process_data(items: list[float], threshold: float) -> list[float]:
    return [x for x in items if x > threshold]
```

**Rule**: All public functions MUST have type hints for parameters and return values.

**Generic types**:

```python
from typing import TypeVar, Generic

T = TypeVar("T")

def first_element(items: list[T]) -> T | None:
    """Returns first element or None if empty."""
    return items[0] if items else None

# For complex types
from collections.abc import Callable

def apply_transform(
    data: list[float],
    transform: Callable[[float], float]
) -> list[float]:
    return [transform(x) for x in data]
```

### 1.3 Mutable Default Arguments - The Classic Gotcha

**❌ DANGEROUS (shared mutable default)**:

```python
def add_item(item, target_list=[]):  # BUG: List created once!
    target_list.append(item)
    return target_list

# Unexpected behavior
print(add_item(1))  # [1]
print(add_item(2))  # [1, 2] - WRONG! Expected [2]
```

**✅ SAFE (None pattern)**:

```python
def add_item(item: int, target_list: list[int] | None = None) -> list[int]:
    if target_list is None:
        target_list = []
    target_list.append(item)
    return target_list

# Correct behavior
print(add_item(1))  # [1]
print(add_item(2))  # [2] - Correct!
```

**Rule**: Never use mutable defaults (`[]`, `{}`, `set()`). Use `None` and create inside function.

### 1.4 Resource Management - Always Use Context Managers

**❌ FRAGILE (manual cleanup)**:

```python
file = open("data.txt")
try:
    data = file.read()
    process(data)
finally:
    file.close()  # Easy to forget!
```

**✅ SAFE (context manager)**:

```python
with open("data.txt") as file:
    data = file.read()
    process(data)
# File automatically closed, even if exception occurs
```

**Rule**: Use `with` statement for files, locks, network connections, database cursors.

**Creating custom context managers**:

```python
from contextlib import contextmanager
import time

@contextmanager
def timer(name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{name} took {elapsed:.3f}s")

# Usage
with timer("Data processing"):
    process_large_dataset()
```

### 1.5 None Checking - `is None` vs Truthiness

**❌ WRONG (confuses None with empty)**:

```python
def get_config(key):
    value = config.get(key)
    if not value:  # BUG: False if value is 0, "", []
        return default_config[key]
    return value
```

**✅ CORRECT (explicit None check)**:

```python
def get_config(key: str) -> str | int | None:
    value = config.get(key)
    if value is None:  # Only checks None, not falsiness
        return default_config[key]
    return value
```

**Rule**: Use `is None` and `is not None` for None checks. Don't rely on truthiness.

**When truthiness is OK**:

```python
# Checking if collection is empty (OK)
if not items:
    print("No items to process")

# Checking if string is empty (OK)
if not user_input:
    print("Input required")

# But for None vs empty distinction:
if items is None:
    items = fetch_from_database()
elif not items:
    print("Empty list returned")
```

### 1.6 Import Safety - No Wildcards, Explicit Imports

**❌ DANGEROUS (namespace pollution)**:

```python
from math import *  # Imports everything, shadows builtins
from utils import *  # Where does this function come from?

result = pow(2, 3)  # math.pow or builtins.pow?
```

**✅ SAFE (explicit imports)**:

```python
from math import sqrt, pi
from utils import process_data, validate_input

result = sqrt(16)  # Clear origin
```

**Rule**: Never use wildcard imports (`from x import *`). Import specific names.

**Import organization** (PEP 8):

```python
"""Module docstring."""
# 1. Standard library imports
import os
import sys
from pathlib import Path

# 2. Third-party imports
import numpy as np
import torch
from sklearn.metrics import accuracy_score

# 3. Local application imports
from myproject.utils import helper_function
from myproject.models import BaseModel

# Constants
MAX_ITERATIONS = 100
```

### 1.7 Global State - Avoid Mutable Globals

**❌ DANGEROUS (mutable global state)**:

```python
# Global mutable variable
cache = {}  # Multiple functions modify this

def add_to_cache(key, value):
    cache[key] = value  # Side effect!

def clear_cache():
    cache.clear()  # Another side effect!
```

**✅ SAFE (encapsulated state)**:

```python
class Cache:
    def __init__(self):
        self._data: dict[str, any] = {}

    def add(self, key: str, value: any) -> None:
        self._data[key] = value

    def clear(self) -> None:
        self._data.clear()

# Usage
cache = Cache()  # Explicit instance
```

**Rule**: Avoid mutable global variables. Use classes or pass state explicitly.

**When globals are OK**:

```python
# Constants (immutable)
API_KEY = os.getenv("API_KEY")
MAX_RETRIES = 3

# Loggers
logger = logging.getLogger(__name__)

# Compiled regexes
EMAIL_PATTERN = re.compile(r"[^@]+@[^@]+\.[^@]+")
```

### 1.8 Exception Chaining - Preserve Original Context

**❌ LOSES CONTEXT**:

```python
try:
    result = json.loads(data)
except json.JSONDecodeError:
    raise ValueError("Invalid data format")  # Original error lost!
```

**✅ PRESERVES CONTEXT**:

```python
try:
    result = json.loads(data)
except json.JSONDecodeError as e:
    raise ValueError("Invalid data format") from e
    # Stack trace shows both errors
```

**Rule**: Use `raise ... from ...` to chain exceptions and preserve context.

**Suppressing context** (rare cases):

```python
try:
    result = parse_data(data)
except ValueError:
    # Intentionally suppress original exception
    raise RuntimeError("Data parsing failed") from None
```

### 1.9 Async Safety - Proper Await Patterns

**❌ BROKEN (not awaited)**:

```python
async def fetch_data(url):
    response = requests.get(url)  # Blocking! Defeats async
    return response.json()

async def main():
    result = fetch_data(url)  # Returns coroutine, not data!
    print(result)  # Prints <coroutine object>
```

**✅ CORRECT (properly awaited)**:

```python
import aiohttp

async def fetch_data(url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

async def main():
    result = await fetch_data(url)  # Actually executes
    print(result)
```

**Rule**: Always `await` coroutines. Use async libraries in async code.

### 1.10 Security - Injection Prevention

**❌ DANGEROUS (SQL injection)**:

```python
query = f"SELECT * FROM users WHERE name = '{user_input}'"
cursor.execute(query)  # user_input = "'; DROP TABLE users; --"
```

**✅ SAFE (parameterized query)**:

```python
query = "SELECT * FROM users WHERE name = ?"
cursor.execute(query, (user_input,))  # Safely escaped
```

**Rule**: Never concatenate user input into queries. Use parameterized queries.

**Command injection**:

```python
# ❌ DANGEROUS
os.system(f"ls {user_path}")  # user_path = "; rm -rf /"

# ✅ SAFE
import subprocess
subprocess.run(["ls", user_path], check=True)  # Argument array
```

### 1.11 The Foolish Consistency Principle - When to Break the Rules

**Critical**: Rules are tools, not absolutes. Sometimes compliance hurts readability.

**PEP 8 states**:
> "A Foolish Consistency is the Hobgoblin of Little Minds"

**When to break the rules**:

```python
# ❌ Following rules blindly (ugly alignment)
result = some_function(
    very_long_parameter_name_one,
    very_long_parameter_name_two,
    very_long_parameter_name_three,
)  # Forced break at 88 chars looks awkward

# ✅ Breaking the rule for readability (>88 chars OK here)
result = some_function(very_long_parameter_name_one, very_long_parameter_name_two)
```

**Break the rules when**:

1. **Matching existing code style** - Backwards compatibility in legacy codebases
2. **Compliance reduces readability** - The code becomes harder to understand
3. **The code predates the rule** - Don't reformat stable, working code
4. **Team has different conventions** - Project-specific style overrides general guides

**Rule**: Use judgment. Code should be readable FIRST, compliant SECOND.

### 1.12 Power Features to Avoid (Google Guidance)

**For large codebases**: Restrict "magic" Python features that obscure control flow.

**❌ AVOID (dynamic attribute access)**:

```python
# Hard to trace with static analysis
for field in fields:
    value = getattr(obj, field)  # Where is 'field' defined?
    setattr(obj, field, transform(value))

# Direct __dict__ manipulation
obj.__dict__['dynamic_attr'] = value  # Bypasses class definition
```

**✅ PREFER (explicit access)**:

```python
# Clear and traceable
for field in fields:
    if field == "name":
        obj.name = transform(obj.name)
    elif field == "age":
        obj.age = transform(obj.age)
```

**❌ AVOID (metaclasses)**:

```python
class Meta(type):
    def __new__(cls, name, bases, dct):
        # Magic happens here - hard to debug
        return super().__new__(cls, name, bases, dct)

class MyClass(metaclass=Meta):  # Obfuscates control flow
    pass
```

**✅ PREFER (explicit patterns)**:

```python
class MyClass:
    """Explicit initialization is clearer."""

    def __init__(self):
        self.setup()  # Clear method call

    def setup(self):
        # Explicit setup logic
        pass
```

**Rule**: Avoid features that make code unreadable to developers who don't know deep Python internals.

**Rationale**: Code should be understandable by:

- Junior engineers new to Python
- Engineers from other languages (C++, Java)
- Anyone without memorizing Python's magic methods

**When magic is OK**:

- Standard protocols: `__init__`, `__str__`, `__repr__`
- Context managers: `__enter__`, `__exit__`
- Well-documented frameworks (Django ORM metaclasses)

---

## 2. Code Layout & Formatting

### 2.1 Indentation - 4 Spaces, No Tabs

**Rule**: Use 4 spaces per indentation level. Never mix tabs and spaces.

```python
def example_function():
    if condition:
        for item in items:
            process(item)
```

**Continuation lines** (aligned):

```python
# Aligned with opening delimiter
result = some_function(argument1, argument2,
                       argument3, argument4)

# Hanging indent (preferred by Black)
result = some_function(
    argument1,
    argument2,
    argument3,
)
```

### 2.2 Line Length - 88 Characters (Black Standard)

**Rule**: Limit lines to 88 characters maximum.

**Historical Context - Why Different Limits?**

| Standard | Line Length | Rationale |
|----------|-------------|-----------|
| **PEP 8** | 79 chars | Historical: 80-column terminals (reserve 1 for newline) |
| **Google** | 80 chars | Aligns with most editors, allows side-by-side diffs |
| **NumPy/Pandas** | 79 chars | Follows PEP 8 for scientific code |
| **Black** (Modern) | 88 chars | Reduces wrapping by ~10%, fits 99% of modern code |

**Why Black chose 88**:

- 10% longer than 79 allows more code on one line
- Reduces unnecessary line breaks in nested code
- Calculated to minimize total lines across large codebases
- Still readable in standard diffs (< 100 chars)

**Docstring Exception**: PEP 8 recommends **72 characters for docstrings** (flowing text) to ensure clean wrapping in documentation tools.

```python
def example():
    """Short summary fits on one line.

    This longer description should wrap at 72 characters to ensure
    it displays cleanly in various documentation tools and terminals.
    """
    pass
```

---

> **Project Override (CUDA_IPC)**: This project uses `line-length = 120` in `pyproject.toml`.
> CUDA/IPC code has long ctypes signatures, struct operations, and f-string diagnostics
> that would require excessive wrapping at 88 chars. Per PEP 8's "Foolish Consistency"
> principle (Section 1.11), this project-specific override improves readability for
> low-level systems code. The 120-char limit balances modern screen widths with
> maintainability.

---

**Line breaking**:

```python
# Long function call
result = very_long_function_name(
    first_argument,
    second_argument,
    third_argument,
    keyword_arg="value",
)

# Long string
message = (
    "This is a very long message that needs to be split "
    "across multiple lines for readability purposes."
)

# Or use implicit string concatenation
sql_query = (
    "SELECT user_id, user_name, email "
    "FROM users "
    "WHERE active = TRUE AND created_at > %s"
)
```

### 2.3 Blank Lines - Semantic Rhythm

**Two blank lines**:

- Between top-level classes
- Between top-level functions

**One blank line**:

- Between methods inside a class
- Between logical sections inside functions (sparingly)

```python
"""Module for data processing."""
import numpy as np


class DataProcessor:
    """Processes data."""

    def __init__(self):
        self.data = []

    def add_data(self, item):
        """Adds an item."""
        self.data.append(item)

    def process(self):
        """Processes all data."""
        return [self._transform(x) for x in self.data]


def standalone_function():
    """A standalone function."""
    pass
```

### 2.4 Imports - Grouped and Sorted

**Import order** (PEP 8 + isort):

1. Standard library
2. Third-party libraries
3. Local application

**Within each group**: Alphabetically sorted.

```python
# Standard library
import os
import sys
from pathlib import Path

# Third-party
import numpy as np
import torch
from sklearn.metrics import accuracy_score

# Local
from myproject.models import BaseModel
from myproject.utils import helper_func
```

**One import per line** (classes are exception):

```python
# ✅ Correct
import os
import sys

# ✅ Also correct (multiple classes from same module)
from typing import Dict, List, Optional

# ❌ Wrong
import os, sys
```

**Relative vs Absolute Imports**:

**For large codebases (Google guidance)**: Prefer absolute imports.

```python
# ✅ Preferred (absolute import - clear package path)
from myproject.utils.helpers import process_data
from myproject.models.base import BaseModel

# ⚠️ Allowed but discouraged (relative import)
from .helpers import process_data  # Where is this module?
from ..models.base import BaseModel  # Hard to trace
```

**When relative imports are OK**:

- Small projects with simple structure
- Internal package organization (within a single package)
- Test files importing from parent directories

**When to use absolute imports**:

- Large codebases (easier to trace with grep/IDE)
- Shared libraries used across projects
- Code review (clear origin of every import)

**Rule**: In doubt, use absolute imports. They're more explicit and easier to maintain.

### 2.5 String Quotes - Double Quotes (Black Standard)

**Rule**: Use double quotes for strings, unless the string contains double quotes.

```python
# ✅ Preferred
message = "Hello, world!"
name = "Alice"

# ✅ OK (avoids escaping)
phrase = 'He said "Hello"'

# ❌ Inconsistent (but Black will fix)
mixed = 'single' + "double"  # Black makes both double
```

**Docstrings**: Always use triple double quotes.

```python
def function():
    """This is a docstring."""
    pass
```

### 2.6 Trailing Commas - The Magic Comma

**Rule**: Use trailing commas in multi-line structures (Black feature).

**Why?** Cleaner diffs when adding items.

```python
# ✅ With trailing comma
items = [
    "first",
    "second",
    "third",  # Trailing comma
]

# Without trailing comma, adding "fourth" changes TWO lines:
# - "third",
# + "third",
# + "fourth"

# With trailing comma, only ONE line changes:
# + "fourth",
```

**Applies to**:

```python
# Lists
data = [
    1,
    2,
    3,
]

# Dictionaries
config = {
    "key1": "value1",
    "key2": "value2",
}

# Function arguments
result = function_call(
    arg1,
    arg2,
    kwarg1="value",
)

# Function definitions
def my_function(
    param1: str,
    param2: int,
    param3: bool = False,
) -> None:
    pass
```

### 2.7 Line Continuation - Parentheses Over Backslash

**❌ AVOID (backslash)**:

```python
result = some_long_computation(arg1, arg2) + \
         another_computation(arg3)
```

**✅ PREFER (parentheses)**:

```python
result = (
    some_long_computation(arg1, arg2)
    + another_computation(arg3)
)
```

**Rule**: Use implicit line continuation (parentheses, brackets, braces) instead of backslash.

**Fluent Interface Formatting** (Method Chaining):

Black formats chained method calls vertically for readability (common in Pandas, ORMs):

```python
# ✅ Black style (vertical stacking)
result = (
    df.filter(col("age") > 18)
    .select("name", "email")
    .sort("name")
    .limit(100)
)

# Also good for SQL-like operations
query = (
    User.query
    .filter_by(active=True)
    .order_by(User.created_at.desc())
    .limit(10)
)

# ❌ Harder to read (all on one line)
result = df.filter(col("age") > 18).select("name", "email").sort("name").limit(100)
```

**Rule**: For method chains > 2 calls, use vertical stacking for readability.

### 2.8 Function/Class Spacing

**Whitespace inside**:

```python
# ✅ Correct
spam(ham[1], {eggs: 2})

# ❌ Wrong
spam( ham[ 1 ], { eggs: 2 } )
```

**Around operators**:

```python
# ✅ Correct
x = 1
y = x + 2
result = x * y + 3

# ❌ Wrong (inconsistent spacing)
x=1
y = x+2
result=x*y + 3
```

---

## 3. Naming Conventions

### 3.1 Classes - CamelCase (CapWords)

**Rule**: Class names use CamelCase with no underscores.

```python
class DataProcessor:
    pass

class NeuralNetwork:
    pass

class HTTPSConnection:  # Acronyms stay uppercase
    pass
```

**Exceptions**: Type variables (see 3.6).

### 3.2 Functions and Variables - snake_case

**Rule**: Functions and variables use lowercase with underscores.

```python
def process_data(input_data):
    result_value = compute_result(input_data)
    return result_value

user_name = "Alice"
max_iterations = 100
```

### 3.3 Constants - UPPER_CASE

**Rule**: Module-level constants use all uppercase with underscores.

```python
MAX_CONNECTIONS = 10
DEFAULT_TIMEOUT = 30
API_BASE_URL = "https://api.example.com"
```

### 3.4 Protected Members - Single Leading Underscore

**Rule**: Use `_name` to indicate "internal use" (weak convention).

```python
class MyClass:
    def __init__(self):
        self._internal_state = 0  # Protected

    def _helper_method(self):  # Protected method
        pass
```

**Note**: Not enforced by Python, just a convention for users.

### 3.5 Private Members - Double Leading Underscore

**Rule**: Use `__name` to trigger name mangling (prevents subclass collision).

```python
class Base:
    def __init__(self):
        self.__private_var = 42  # Mangled to _Base__private_var

class Derived(Base):
    def __init__(self):
        super().__init__()
        self.__private_var = 99  # Mangled to _Derived__private_var
        # These don't collide!
```

**Use sparingly**: Only when you need true name collision prevention.

### 3.6 Type Variables - Single Capital Letters

**Rule**: Type variables use single capital letters or CamelCase.

```python
from typing import TypeVar, Generic

T = TypeVar("T")  # Generic type
K = TypeVar("K")  # Key type
V = TypeVar("V")  # Value type

# Or descriptive names
InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")
```

### 3.7 Module and Package Names

**Rule**: Use short, lowercase names. Avoid underscores if possible.

```python
# ✅ Preferred
import utils
import mypackage

# ✅ OK (if needed for readability)
import data_processing

# ❌ Avoid
import MyPackage
import my-package  # Hyphens not allowed
```

---

## 4. Docstrings (Hybrid Format)

### 4.1 Google Style Base - Args/Returns/Raises

**Standard format** for general Python code:

```python
def fetch_data(url: str, timeout: int = 30) -> dict:
    """Fetches data from a URL.

    Args:
        url: The URL to fetch data from. Must be HTTP or HTTPS.
        timeout: Maximum time to wait in seconds. Defaults to 30.

    Returns:
        A dictionary containing the parsed JSON response.

    Raises:
        requests.ConnectionError: If the connection fails.
        requests.Timeout: If the request times out.
        ValueError: If the response is not valid JSON.

    Examples:
        >>> data = fetch_data("https://api.example.com/data")
        >>> print(data["status"])
        "success"
    """
    response = requests.get(url, timeout=timeout)
    return response.json()
```

**Sections** (in order):

1. **Summary line**: One-line description ending with period
2. **Extended description**: Optional detailed explanation (blank line after summary)
3. **Args**: Parameter descriptions
4. **Returns**: What the function returns
5. **Raises**: What exceptions can be raised
6. **Examples**: Optional usage examples

### 4.2 NumPy Extensions - Parameters with Shape/Dtype

**For ML/scientific code** with arrays:

```python
import numpy as np
from numpy.typing import NDArray

def matrix_multiply(
    a: NDArray[np.float32],
    b: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Multiplies two matrices.

    Parameters
    ----------
    a : ndarray, shape (n, m)
        The first matrix with dtype float32.
    b : ndarray, shape (m, k)
        The second matrix with dtype float32.

    Returns
    -------
    result : ndarray, shape (n, k)
        The product of matrices a and b.

    Raises
    ------
    ValueError
        If matrix dimensions are incompatible (a.shape[1] != b.shape[0]).

    Notes
    -----
    This function uses NumPy's matmul for computation.

    See Also
    --------
    numpy.matmul : Underlying implementation
    torch.mm : PyTorch equivalent

    Examples
    --------
    >>> a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    >>> b = np.array([[5, 6], [7, 8]], dtype=np.float32)
    >>> result = matrix_multiply(a, b)
    >>> result.shape
    (2, 2)
    """
    if a.shape[1] != b.shape[0]:
        raise ValueError("Incompatible matrix dimensions")
    return np.matmul(a, b)
```

**Key differences**:

- `Parameters` instead of `Args` (use underline `---`)
- Include **shape** and **dtype** information
- `Notes` and `See Also` sections for scientific context

### 4.3 Module Docstrings

**Rule**: Every module should have a docstring at the top.

```python
"""Data processing utilities for machine learning pipelines.

This module provides functions for loading, cleaning, and transforming
data for ML models. It includes support for:
- CSV and JSON loading
- Missing value imputation
- Feature scaling and normalization

Example:
    >>> from myproject import data_utils
    >>> data = data_utils.load_csv("dataset.csv")
    >>> cleaned = data_utils.clean_data(data)
"""
import pandas as pd
import numpy as np
```

### 4.4 Class Docstrings

```python
class DataLoader:
    """Loads and preprocesses data for training.

    This class handles data loading from various sources and applies
    preprocessing transformations.

    Attributes:
        data_path: Path to the data file.
        batch_size: Number of samples per batch.
        transform: Optional transformation function.

    Examples:
        >>> loader = DataLoader("data.csv", batch_size=32)
        >>> for batch in loader:
        ...     process(batch)
    """

    def __init__(self, data_path: str, batch_size: int = 32):
        """Initializes the DataLoader.

        Args:
            data_path: Path to the data file.
            batch_size: Number of samples per batch.
        """
        self.data_path = data_path
        self.batch_size = batch_size
```

### 4.5 Function Docstrings

See sections 4.1 and 4.2 for complete examples.

**Minimal example** (simple functions):

```python
def add(a: int, b: int) -> int:
    """Returns the sum of a and b."""
    return a + b
```

### 4.6 Property Docstrings

```python
class Model:
    @property
    def device(self) -> str:
        """The device this model is on ('cpu' or 'cuda')."""
        return self._device

    @device.setter
    def device(self, value: str) -> None:
        """Sets the device.

        Args:
            value: Either 'cpu' or 'cuda'.

        Raises:
            ValueError: If value is not 'cpu' or 'cuda'.
        """
        if value not in ("cpu", "cuda"):
            raise ValueError(f"Invalid device: {value}")
        self._device = value
```

---

## 5. Type Hints

### 5.1 Basic Types

```python
def greet(name: str, age: int) -> str:
    return f"Hello {name}, you are {age} years old"

# Built-in types
count: int = 42
price: float = 19.99
is_active: bool = True
message: str = "Hello"
data: bytes = b"binary data"
```

### 5.2 Container Types

```python
from typing import Dict, List, Set, Tuple

# Use built-in types (Python 3.9+)
names: list[str] = ["Alice", "Bob"]
scores: dict[str, int] = {"Alice": 95, "Bob": 87}
unique_ids: set[int] = {1, 2, 3}
coordinates: tuple[float, float] = (10.0, 20.0)

# For older Python (3.8 and below)
from typing import List, Dict, Set, Tuple

names: List[str] = ["Alice", "Bob"]
scores: Dict[str, int] = {"Alice": 95}
```

### 5.3 Optional and Union

```python
from typing import Optional

# Optional[X] means X | None
def find_user(user_id: int) -> Optional[str]:
    return users.get(user_id)  # Returns str or None

# Modern syntax (Python 3.10+)
def find_user(user_id: int) -> str | None:
    return users.get(user_id)

# Union for multiple types
def process(value: int | float | str) -> str:
    return str(value)
```

### 5.4 Callable and TypeVar

```python
from typing import Callable, TypeVar

# Function that takes a function
def apply_twice(
    func: Callable[[int], int],
    value: int
) -> int:
    return func(func(value))

# Generic function
T = TypeVar("T")

def first(items: list[T]) -> T | None:
    return items[0] if items else None

# Now works with any type
numbers: list[int] = [1, 2, 3]
first_num = first(numbers)  # Type inferred as int | None

strings: list[str] = ["a", "b"]
first_str = first(strings)  # Type inferred as str | None
```

### 5.5 Generic Classes

```python
from typing import Generic, TypeVar

T = TypeVar("T")

class Stack(Generic[T]):
    """A generic stack data structure."""

    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T | None:
        return self._items.pop() if self._items else None

# Usage
int_stack: Stack[int] = Stack()
int_stack.push(1)
int_stack.push(2)

str_stack: Stack[str] = Stack()
str_stack.push("hello")
```

### 5.6 Protocol and ABC

```python
from typing import Protocol
from abc import ABC, abstractmethod

# Protocol (structural subtyping)
class Drawable(Protocol):
    """Anything with a draw method."""

    def draw(self) -> None:
        ...

def render(obj: Drawable) -> None:
    obj.draw()  # Type checker ensures draw() exists

# Abstract base class (nominal subtyping)
class Animal(ABC):
    @abstractmethod
    def make_sound(self) -> str:
        """Returns the sound this animal makes."""
        pass

class Dog(Animal):
    def make_sound(self) -> str:
        return "Woof!"
```

---

## 6. ML/Data Science Patterns

### 6.1 NumPy/Pandas Imports - Standard Abbreviations

**Rule**: Use standard abbreviations for scientific libraries.

```python
# ✅ Standard (universally recognized)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

# ❌ Non-standard
import numpy  # No abbreviation
import pandas as p  # Wrong abbreviation
```

### 6.2 Array-like Duck Typing

**Rule**: Check for array-like behavior, not specific types.

```python
# ❌ Too restrictive
def process(data: np.ndarray) -> np.ndarray:
    return data * 2  # Only works with NumPy arrays

# ✅ Duck typing (works with NumPy, PyTorch, Dask, CuPy)
from numpy.typing import ArrayLike

def process(data: ArrayLike) -> ArrayLike:
    """Works with any array-like object."""
    return data * 2  # Works with np.ndarray, torch.Tensor, etc.
```

### 6.3 Shape Documentation

**Rule**: Document array shapes in docstrings.

```python
def convolve_2d(
    image: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    """Applies 2D convolution to an image.

    Parameters
    ----------
    image : ndarray, shape (H, W, C)
        Input image with height H, width W, and C channels.
    kernel : ndarray, shape (K, K)
        Square convolution kernel of size K x K.

    Returns
    -------
    output : ndarray, shape (H', W', C)
        Convolved image. H' and W' depend on padding/stride.
    """
    pass
```

### 6.4 GPU Memory Management - PyTorch

**Rule**: Explicitly manage GPU memory in training loops.

```python
import torch

def train_epoch(model, dataloader, optimizer, device):
    """Trains model for one epoch."""
    model.train()

    for batch_idx, (data, target) in enumerate(dataloader):
        # Move to device
        data = data.to(device)
        target = target.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Clear GPU cache periodically (every 10 batches)
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()

        # Delete intermediate tensors if memory constrained
        del data, target, output, loss
```

**Memory-efficient patterns**:

```python
# Use gradient checkpointing for large models
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    # Trades compute for memory
    return checkpoint(self.expensive_layer, x)

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(data)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 6.5 I/O Separation - Keep Logic Pure

**Rule**: Separate computation from I/O for testability and reusability.

```python
# ❌ Tightly coupled
def analyze_csv(filepath: str) -> float:
    df = pd.read_csv(filepath)  # I/O mixed with logic
    return df["value"].mean()

# ✅ Separated concerns
def load_data(filepath: str) -> pd.DataFrame:
    """Handles I/O."""
    return pd.read_csv(filepath)

def compute_mean(df: pd.DataFrame, column: str) -> float:
    """Pure computation (easy to test)."""
    return df[column].mean()

# Usage
df = load_data("data.csv")
result = compute_mean(df, "value")
```

### 6.6 Tensor Type Hints

```python
import torch
from torch import Tensor

def linear_layer(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
) -> Tensor:
    """Applies linear transformation.

    Args:
        x: Input tensor, shape (batch_size, in_features).
        weight: Weight matrix, shape (out_features, in_features).
        bias: Optional bias vector, shape (out_features,).

    Returns:
        Output tensor, shape (batch_size, out_features).
    """
    output = x @ weight.T
    if bias is not None:
        output = output + bias
    return output
```

### 6.7 Configuration Dataclasses

**Rule**: Use dataclasses for configuration objects.

```python
from dataclasses import dataclass, field

@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Required fields
    model_name: str
    dataset_path: str

    # Optional fields with defaults
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 10
    device: str = "cuda"

    # Field with factory (for mutable defaults)
    optimizer_params: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validates configuration after initialization."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'")

# Usage
config = TrainingConfig(
    model_name="resnet50",
    dataset_path="data/train",
    batch_size=64,
)
```

---

## 7. Common Anti-patterns & Fixes

### 7.1 Bare Except → Specific Exceptions

```python
# ❌ Catches everything (even Ctrl+C!)
try:
    process_data()
except:
    print("Error")

# ✅ Specific exception
try:
    process_data()
except ValueError as e:
    logger.error(f"Invalid data: {e}")
```

### 7.2 Mutable Default → None Pattern

```python
# ❌ Shared mutable default
def func(items=[]):
    items.append(1)
    return items

# ✅ None with initialization
def func(items: list[int] | None = None) -> list[int]:
    if items is None:
        items = []
    items.append(1)
    return items
```

### 7.3 `== None` → `is None`

```python
# ❌ Equality check (can be overridden)
if value == None:
    pass

# ✅ Identity check
if value is None:
    pass
```

### 7.4 Manual Cleanup → Context Manager

```python
# ❌ Manual resource management
file = open("data.txt")
data = file.read()
file.close()

# ✅ Context manager
with open("data.txt") as file:
    data = file.read()
```

### 7.5 String Concatenation → f-strings

```python
# ❌ Slow and ugly
message = "Hello " + name + ", you have " + str(count) + " messages"

# ✅ Fast and readable
message = f"Hello {name}, you have {count} messages"
```

### 7.6 Nested Comprehensions → Loops

```python
# ❌ Unreadable nested comprehension
result = [
    item.upper()
    for sublist in data
    for item in sublist
    if item.startswith("a")
    if len(item) > 3
]

# ✅ Clear explicit loops
result = []
for sublist in data:
    for item in sublist:
        if item.startswith("a") and len(item) > 3:
            result.append(item.upper())
```

### 7.7 Checking Type with `type()` → isinstance()

```python
# ❌ Doesn't work with inheritance
if type(obj) == list:
    pass

# ✅ Works with subclasses
if isinstance(obj, list):
    pass

# ✅ Check multiple types
if isinstance(obj, (list, tuple)):
    pass
```

### 7.8 Comprehensions for Side Effects → Explicit Loops

```python
# ❌ Wastes memory building list of Nones
[print(x) for x in items]  # Comprehension returns list
result = [file.close() for file in files]  # Bad!

# ✅ Use explicit loop for side effects
for x in items:
    print(x)

for file in files:
    file.close()
```

**Why it's bad**: Comprehensions allocate memory for a result list. Using them for side effects (print, close, modify) wastes memory and obscures intent.

**Rule**: Use comprehensions only for building new collections. Use loops for side effects.

### 7.9 Return Value Symmetry → Explicit Returns

```python
# ❌ Implicit None return (asymmetric)
def get_value(key: str):
    if key in cache:
        return cache[key]
    # Implicitly returns None (confusing)

# ✅ Explicit None return (symmetric)
def get_value(key: str) -> str | None:
    if key in cache:
        return cache[key]
    return None  # Clear intent

# ❌ Inconsistent return types
def process(value: int):
    if value > 0:
        return value * 2
    elif value < 0:
        return str(value)  # Different type!
    # Implicit None

# ✅ Consistent return types
def process(value: int) -> int:
    if value > 0:
        return value * 2
    elif value < 0:
        return -value  # Same type
    return 0  # Explicit default
```

**Rule**: If a function returns a value in one branch, explicitly return a value (or None) in ALL branches.

### 7.10 Compound Statements → One Statement Per Line

```python
# ❌ Compound statement (hard to debug)
if condition: return value

# ❌ Multiple statements on one line
x = 1; y = 2; z = 3

# ✅ One statement per line
if condition:
    return value

# ✅ Clear and debuggable
x = 1
y = 2
z = 3
```

**Why it's bad**:

- Debuggers can't set breakpoints on individual statements
- Harder to read in diffs
- Violates "one logical unit per line" principle

**Rule**: Write one statement per line. Don't compress code for brevity.

---

## 8. Error Recovery Reference

### 8.1 Common Error Patterns

#### Error: "NameError: name 'X' is not defined"

**Likely Causes**:

1. Variable used before assignment
2. Typo in variable name
3. Missing import

```python
# Fix: Check spelling and order
result = process_data(input_value)  # Ensure input_value exists

# Fix: Add import
import numpy as np  # If np was undefined
```

#### Error: "TypeError: unsupported operand type(s)"

```python
# ❌ Mixing incompatible types
result = "5" + 10

# ✅ Convert types
result = int("5") + 10
```

#### Error: "AttributeError: 'NoneType' object has no attribute"

```python
# ❌ Calling method on None
result = get_data()
print(result.shape)  # result is None!

# ✅ Check for None
result = get_data()
if result is not None:
    print(result.shape)
```

### 8.2 Diagnostic Commands

**Check object type**:

```python
print(type(obj))
print(isinstance(obj, list))
```

**Inspect object attributes**:

```python
print(dir(obj))  # List all attributes
print(vars(obj))  # Show __dict__
```

**Check if attribute exists**:

```python
if hasattr(obj, "method_name"):
    obj.method_name()
```

### 8.3 Recovery Workflows

#### Workflow: Fix Import Error

1. Check if package installed: `pip list | grep package_name`
2. Install if missing: `pip install package_name`
3. Check import path: `from package.submodule import function`
4. Verify Python version compatibility

#### Workflow: Fix Type Error

1. Print types: `print(type(var1), type(var2))`
2. Add type conversion: `int(var)`, `str(var)`, `float(var)`
3. Add type hints for clarity
4. Use mypy to catch type errors: `mypy script.py`

---

## 9. Quick Reference Tables

### 9.1 Safe vs Dangerous Patterns

| Situation | ❌ Dangerous | ✅ Safe |
|-----------|-------------|---------|
| **Exception handling** | `except:` | `except ValueError:` |
| **Default arguments** | `def func(items=[])` | `def func(items=None)` |
| **None check** | `if not value:` | `if value is None:` |
| **Resource management** | `file.close()` | `with open(...) as file:` |
| **String formatting** | `"Hello " + name` | `f"Hello {name}"` |
| **Imports** | `from x import *` | `from x import specific` |
| **Type checking** | `type(x) == list` | `isinstance(x, list)` |

### 9.2 Pre-flight Checklist

Before running Python code, verify:

- [ ] All functions have type hints
- [ ] No bare `except:` clauses
- [ ] No mutable default arguments (`[]`, `{}`)
- [ ] Files opened with `with` statement
- [ ] None checks use `is None`
- [ ] Imports are explicit (no wildcards)
- [ ] Variables named with `snake_case`
- [ ] Classes named with `CamelCase`
- [ ] Constants named with `UPPER_CASE`
- [ ] Line length ≤ 88 characters

### 9.3 Error Message → Likely Cause

| Error | Likely Cause | Quick Fix |
|-------|--------------|-----------|
| NameError | Variable not defined | Check spelling, add import |
| TypeError | Wrong type | Convert type or add check |
| AttributeError | Accessing None | Add None check |
| KeyError | Missing dict key | Use `.get()` or check `if key in dict` |
| IndexError | List index out of range | Check list length |
| ValueError | Invalid value | Validate input |
| ImportError | Module not found | Install package with pip |

### 9.4 PEP 8 Naming Cheat Sheet

| Entity | Convention | Example |
|--------|------------|---------|
| **Class** | CamelCase | `class DataProcessor` |
| **Function** | snake_case | `def process_data()` |
| **Variable** | snake_case | `user_name = "Alice"` |
| **Constant** | UPPER_CASE | `MAX_SIZE = 100` |
| **Protected** | _leading | `self._internal` |
| **Private** | __double | `self.__private` |
| **Module** | lowercase | `utils.py` |

### 9.5 Type Hint Cheat Sheet

```python
# Basic types
name: str
age: int
price: float
is_active: bool

# Collections (Python 3.9+)
names: list[str]
scores: dict[str, int]
unique: set[int]
coords: tuple[float, float]

# Optional (can be None)
result: str | None
result: Optional[str]  # Older syntax

# Union (multiple types)
value: int | float | str

# Callable (function)
func: Callable[[int, int], int]  # Takes 2 ints, returns int

# Generic
from typing import TypeVar
T = TypeVar("T")
def first(items: list[T]) -> T | None: ...
```

---

## 10. Integration with Claude Code

### 10.1 When to Consult This Guide

**Pre-flight (before writing Python)**:

- Writing new functions or classes
- Handling errors or exceptions
- Working with files or resources
- Defining data structures
- Writing ML/data science code

**Error recovery (after Python failure)**:

- Exception raised during execution
- Type errors from mypy or IDE
- Import errors
- Runtime AttributeError or NameError

### 10.2 Automatic Validation Triggers

Claude should automatically reference this guide when:

**Critical patterns**:

- Bare `except:` detected
- Mutable default argument (`def func(items=[])`)
- Using `type()` instead of `isinstance()`
- File operations without context manager
- Missing type hints on public functions

**Formatting issues**:

- Line length > 88 characters
- Inconsistent import ordering
- Wrong naming convention (camelCase function, snake_case class)

### 10.3 Quick Lookup by Error Type

| Error Type | Section | Key Fix |
|------------|---------|---------|
| Bare except | 1.1, 7.1 | Specify exception type |
| Mutable default | 1.3, 7.2 | Use None pattern |
| None check | 1.5, 7.3 | Use `is None` |
| Resource leak | 1.4, 7.4 | Use context manager |
| Import error | 1.6, 8.3 | Install package, fix path |
| Type error | 5.1-5.6, 8.3 | Add type hints, convert |

### 10.4 Ruff/Black Configuration

See Section 11 for full configuration.

Quick check:

```bash
# Format with black
black script.py

# Lint with ruff
ruff check script.py

# Auto-fix with ruff
ruff check --fix script.py
```

---

## 11. Tooling Configuration

### 11.1 pyproject.toml - Ruff + Black

**Recommended configuration**:

```toml
[tool.black]
line-length = 88
target-version = ["py310"]
include = '\.pyi?$'

[tool.ruff]
line-length = 88
target-version = "py310"

# Enable rules
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort (import sorting)
    "N",   # pep8-naming
    "UP",  # pyupgrade
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "SIM", # flake8-simplify
]

# Ignore specific rules
ignore = [
    "E501",  # Line too long (Black handles this)
]

[tool.ruff.per-file-ignores]
"__init__.py" = ["F401"]  # Allow unused imports

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

### 11.2 Pre-commit Hooks

**`.pre-commit-config.yaml`**:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3.10

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
        args: [--fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

Install: `pre-commit install`

### 11.3 VS Code Settings

**`.vscode/settings.json`**:

```json
{
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.linting.mypyEnabled": true,
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "[python]": {
    "editor.rulers": [88],
    "editor.tabSize": 4
  }
}
```

---

## 12. Summary: Golden Rules

1. **ALWAYS specify exception types**: `except ValueError:` not `except:`
2. **ALWAYS use type hints**: For all public function parameters and returns
3. **NEVER use mutable defaults**: Use `None` pattern instead of `def func(items=[])`
4. **ALWAYS use context managers**: `with open(...)` for files and resources
5. **ALWAYS check None explicitly**: `if value is None:` not `if not value:`
6. **NEVER use wildcard imports**: `from x import specific` not `from x import *`
7. **ALWAYS use isinstance()**: Not `type() ==` for type checking
8. **ALWAYS document array shapes**: In ML code docstrings
9. **ALWAYS use f-strings**: Not string concatenation or `%` formatting
10. **ALWAYS separate I/O from logic**: For testability and reusability
11. **ALWAYS name correctly**: `snake_case` functions, `CamelCase` classes, `UPPER_CASE` constants
12. **ALWAYS use double quotes**: For consistency (Black standard)
13. **ALWAYS limit lines to 88 chars**: Black standard
14. **ALWAYS group imports**: stdlib → third-party → local
15. **ALWAYS use trailing commas**: In multi-line structures
16. **ALWAYS chain exceptions**: `raise ... from ...` to preserve context
17. **ALWAYS await coroutines**: In async code
18. **ALWAYS use parameterized queries**: Never concatenate SQL
19. **ALWAYS clear GPU memory**: In PyTorch training loops
20. **ALWAYS validate configuration**: In `__post_init__` or properties

---

**Last Updated**: 2026-01-05
**Character Count**: ~42,000 characters
**License**: MIT (Compiled from PEP 8, Google, Black, NumPy documentation standards)
**Related Guides**: See `BASH_STYLE_GUIDE.md`, `BATCH_STYLE_GUIDE.md`, `SHELL_STYLE_GUIDE.md`
