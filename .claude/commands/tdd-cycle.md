---
model: claude-opus-4-1
---

# Practical Test-Driven Development (TDD) Workflow

**Purpose**: Disciplined RED-GREEN-REFACTOR cycle for high-quality, well-tested code.
**Time**: 30-60 minutes per feature
**Focus**: Test-first development, minimal implementation, continuous refactoring

## When to Use TDD

✅ **Good use cases:**

- Complex business logic (algorithms, state machines)
- Critical functionality (authentication, payment processing)
- Bug fixes (reproduce bug with test first)
- Public APIs (define contracts with tests)
- Refactoring existing code (safety net)

❌ **Don't use TDD for:**

- UI/UX experimentation (design unclear)
- Prototyping/spikes (exploring solutions)
- Trivial getters/setters
- Glue code (simple wiring)
- One-off scripts

## Feature to Implement

$ARGUMENTS

---

## The TDD Cycle

```
RED → GREEN → REFACTOR → Repeat
 ↓       ↓         ↓
Test   Code    Improve
Fails  Works   Quality
```

---

## Phase 1: RED - Write a Failing Test

### Step 1: Write ONE Failing Test

**Focus**: Write the simplest test that specifies desired behavior.

```python
import pytest

def test_search_returns_relevant_results():
    """Test that search returns results matching query"""
    # Arrange
    engine = SearchEngine()
    query = "python decorators"

    # Act
    results = engine.search(query)

    # Assert
    assert len(results) > 0
    assert all("python" in r.lower() or "decorator" in r.lower()
               for r in results)
```

### Step 2: Verify Test Fails Correctly

**Run the test and ensure it fails:**

```bash
pytest tests/test_search.py::test_search_returns_relevant_results -v
```

**Expected failure reasons:**

- `ImportError`: SearchEngine doesn't exist yet ✅ Good
- `AttributeError`: search() method doesn't exist ✅ Good
- `AssertionError`: search() returns wrong results ✅ Good

**Bad failure reasons:**

- `SyntaxError` in test ❌ Fix test
- `TypeError` from test setup ❌ Fix test
- Test passes accidentally ❌ Test is wrong

### TDD Rule #1: Never Write Production Code Without a Failing Test

---

## Phase 2: GREEN - Make Test Pass (Minimal Code)

### Step 3: Write Minimal Implementation

**Goal**: Make the test pass with the simplest possible code.

```python
class SearchEngine:
    def search(self, query):
        # Minimal implementation to pass test
        return ["python decorator tutorial", "decorators in python"]
```

**Yes, this is allowed!** Hard-coding is fine initially. More tests will force better implementation.

### Step 4: Run Tests and Verify They Pass

```bash
pytest tests/test_search.py -v
```

**All tests must pass.** If not, fix implementation (not tests).

### TDD Rule #2: Write Only Enough Code to Make Tests Pass

---

## Phase 3: REFACTOR - Improve Code Quality

### Step 5: Refactor While Keeping Tests Green

**Now improve the code:**

```python
class SearchEngine:
    def __init__(self):
        self.documents = self._load_documents()

    def search(self, query):
        query_terms = query.lower().split()
        results = []
        for doc in self.documents:
            if any(term in doc.lower() for term in query_terms):
                results.append(doc)
        return results

    def _load_documents(self):
        # Load from database/files
        pass
```

### Step 6: Run Tests After Each Change

```bash
# Run tests after EVERY refactoring change
pytest tests/test_search.py -v
```

**If tests break, revert immediately.** Never commit broken tests.

### TDD Rule #3: Refactor Mercilessly, But Keep Tests Green

---

## Incremental TDD: One Test at a Time

**Recommended approach for most features:**

```
1. Write ONE test (fails)
2. Make ONLY that test pass
3. Refactor
4. Write NEXT test (fails)
5. Make it pass
6. Refactor
7. Repeat...
```

**Example progression:**

```python
# Test 1: Basic functionality
def test_search_returns_results():
    results = engine.search("python")
    assert len(results) > 0

# Test 2: Empty query handling
def test_search_empty_query_returns_empty():
    results = engine.search("")
    assert results == []

# Test 3: Case insensitivity
def test_search_is_case_insensitive():
    results1 = engine.search("Python")
    results2 = engine.search("python")
    assert results1 == results2

# Test 4: Multiple terms
def test_search_multiple_terms():
    results = engine.search("python decorators")
    assert len(results) > 0
```

**Each test forces implementation improvements.**

---

## Coverage Targets

**Aim for these thresholds:**

| Metric | Target | Critical Code |
|--------|--------|---------------|
| Line coverage | 80%+ | 100% |
| Branch coverage | 75%+ | 100% |
| Function coverage | 90%+ | 100% |

```bash
# Generate coverage report
pytest --cov=src --cov-report=html tests/

# View report
open htmlcov/index.html
```

---

## Test Organization

### Arrange-Act-Assert Pattern

```python
def test_example():
    # Arrange: Set up test data
    engine = SearchEngine()
    query = "test query"

    # Act: Execute the code being tested
    result = engine.search(query)

    # Assert: Verify the results
    assert result is not None
```

### Fixture Pattern (Shared Setup)

```python
import pytest

@pytest.fixture
def search_engine():
    """Shared search engine instance for tests"""
    return SearchEngine()

@pytest.fixture
def sample_documents():
    """Sample test data"""
    return [
        "Python decorators tutorial",
        "JavaScript async patterns",
        "Python generators guide"
    ]

def test_with_fixtures(search_engine, sample_documents):
    search_engine.add_documents(sample_documents)
    results = search_engine.search("python")
    assert len(results) == 2
```

---

## Common TDD Patterns

### Pattern 1: Start with Simplest Case

```python
# ❌ Don't start with complex case
def test_search_with_regex_and_fuzzy_matching_and_ranking():
    # Too complex for first test!
    pass

# ✅ Start simple
def test_search_returns_exact_match():
    results = engine.search("python")
    assert "python" in results[0].lower()
```

### Pattern 2: One Assert Per Test (Usually)

```python
# ❌ Multiple unrelated assertions
def test_search():
    assert engine.search("python")
    assert engine.search("") == []
    assert len(engine.search("test")) > 0

# ✅ Separate tests for each concern
def test_search_returns_results():
    assert engine.search("python")

def test_search_empty_query():
    assert engine.search("") == []

def test_search_returns_multiple():
    assert len(engine.search("test")) > 0
```

### Pattern 3: Test Behavior, Not Implementation

```python
# ❌ Testing implementation details
def test_search_calls_internal_method():
    engine.search("test")
    assert engine._internal_method_called  # Brittle!

# ✅ Testing behavior
def test_search_returns_relevant_results():
    results = engine.search("test")
    assert all("test" in r.lower() for r in results)
```

---

## TDD Anti-Patterns

❌ **Writing tests after code** → Defeats the purpose
❌ **Skipping refactor phase** → Code quality degrades
❌ **Testing implementation details** → Brittle tests
❌ **Large test suites with slow tests** → Nobody runs them
❌ **Modifying tests to make them pass** → Breaking the cycle
❌ **100% coverage obsession** → Diminishing returns

---

## Quick TDD Checklist

**Before starting:**

- [ ] Feature clearly defined
- [ ] Test framework set up
- [ ] Example test cases identified

**During development:**

- [ ] Test written first (RED)
- [ ] Test fails for right reason
- [ ] Minimal code to pass (GREEN)
- [ ] All tests pass
- [ ] Code refactored (REFACTOR)
- [ ] Tests still pass after refactoring

**After feature complete:**

- [ ] Coverage meets targets
- [ ] Tests run fast (<5s for unit tests)
- [ ] No skipped/ignored tests
- [ ] Test names clearly describe behavior

---

## Pytest Tips for TDD

### Useful Pytest Commands

```bash
# Run specific test
pytest tests/test_search.py::test_search_returns_results -v

# Run tests matching pattern
pytest -k "search" -v

# Stop on first failure
pytest -x

# Show print statements
pytest -s

# Run with coverage
pytest --cov=src --cov-report=term-missing tests/

# Watch mode (re-run on file changes)
pytest-watch
```

### Useful Pytest Fixtures

```python
# Setup and teardown
@pytest.fixture
def database():
    db = setup_test_database()
    yield db
    cleanup_test_database(db)

# Parametrized tests
@pytest.mark.parametrize("query,expected_count", [
    ("python", 2),
    ("javascript", 1),
    ("rust", 0),
])
def test_search_counts(query, expected_count):
    results = engine.search(query)
    assert len(results) == expected_count
```

---

## When TDD is Hard

**If tests are difficult to write, it's a design smell:**

- Hard to instantiate classes? → Too many dependencies
- Hard to test functions? → Functions doing too much
- Tests need lots of mocking? → Tight coupling
- Tests are slow? → Integration tests, not unit tests

**Solution**: Improve design, then tests become easy.

---

## TDD Benefits

**Short-term:**

- Fewer bugs (catch issues immediately)
- Better design (testability forces good design)
- Living documentation (tests show how code works)

**Long-term:**

- Fearless refactoring (tests catch regressions)
- Faster debugging (failing test shows exact problem)
- Confidence in changes (comprehensive test suite)

---

## Summary

**TDD in 3 Rules:**

1. **RED**: Write a failing test
2. **GREEN**: Make it pass with minimal code
3. **REFACTOR**: Improve code while keeping tests green

**Time per cycle:** 5-15 minutes
**Cycles per feature:** 5-10 cycles
**Total time:** 30-60 minutes per feature

**Remember:**

- Test first, code second
- Simplest code that works
- Refactor constantly
- Keep tests fast

Focus on the discipline, not perfection. TDD is a skill that improves with practice.
