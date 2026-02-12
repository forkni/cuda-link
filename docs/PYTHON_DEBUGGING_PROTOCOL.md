# Python Debugging Protocol

**Version:** 1.0
**Last Updated:** 2026-01-14
**Based on:** The Art of Debugging Open Book by Stas Bekman

---

## Table of Contents

1. [Debug Cycle Optimization](#1-debug-cycle-optimization)
2. [Print Debugging Patterns](#2-print-debugging-patterns)
3. [Package and Import Debugging](#3-package-and-import-debugging)
4. [Process Inspection](#4-process-inspection)
5. [Test Suite Configuration](#5-test-suite-configuration)
6. [Debugger Usage](#6-debugger-usage)
7. [Shell Efficiency](#7-shell-efficiency)
8. [Finding Breaking Changes](#8-finding-breaking-changes)

---

## 1. Debug Cycle Optimization

### 1.1 Quick Iteration Principles

**Goal:** Debug cycles should take **seconds**, not minutes.

**Checklist:**

- [ ] **Minimize payload size**
  - Use tiny datasets (2x2 matrices instead of 1000x1000)
  - Use small models (2 layers instead of 48)
  - Reduce batch size to 1

- [ ] **Use synthetic data for functional debugging**
  ```python
  # Good: Easy to spot issues
  x = torch.tensor([1.0, 2.0, 3.0, 4.0])
  y = torch.tensor([10.0, 20.0, 30.0, 40.0])

  # Bad: Hard to debug
  x = torch.randn(1000)
  ```

- [ ] **Run locally before distributed**
  - Single process > Multi-process
  - Single GPU > Multi-GPU
  - Local desktop > Remote cluster

- [ ] **Skip expensive operations during debugging**
  ```python
  DEBUG_MODE = True

  if not DEBUG_MODE:
      results = expensive_computation(data)
  else:
      results = mock_results  # Use cached/mocked data
  ```

**StreamDiffusionTD-Custom Validation:**
```python
# From main_sdtd.py:44-47
ENABLE_TOME_OSC_STATS = False  # Disable stats for performance
TOME_SUMMARY_THROTTLE_FRAMES = 60  # Reduce frequency
```

---

### 1.2 Atomic Debug Cycles

**Problem:** Forgetting steps in multi-command debugging

**Solution:** Combine commands with `&&` or `;`

```bash
# Atomic: Always clean before running
rm -r data && ./launch.sh

# Atomic: Multi-step setup
source venv/bin/activate && pip install -e . && python test.py

# Atomic: Test and report
pytest tests/ && echo "Tests passed!" || echo "Tests failed!"
```

**Bash Script Best Practices:**

```bash
#!/bin/bash
set -euo pipefail  # Exit on error, undefined vars, pipe failures
set -x             # Print commands (comment out for production)

# Your script here
```

**StreamDiffusionTD-Custom Validation:**
```python
# From main_sdtd.py - Combined initialization
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

---

### 1.3 One-Liner Programs for Isolation

**Purpose:** Test specific functionality without full application

```bash
# Test import speed
python -c "import torch"

# Test model loading
python -c 'from transformers import AutoModel; AutoModel.from_pretrained("t5-small")'

# Check package version
python -c "import torch; print(torch.__version__)"

# Verify CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 2. Print Debugging Patterns

### 2.1 Python 3.8+ Auto-Print F-String

**Old Way (Error-Prone):**
```python
x = 5
y = 6
print(f"x+y={x+y}")  # Easy to mistype
```

**New Way (Self-Describing):**
```python
x = 5
y = 6
print(f"{x=}, {y=}, {x+y=}")
# Output: x=5, y=6, x+y=11
```

**Advanced Usage:**
```python
# Multiple expressions
print(f"{len(items)=}, {sum(items)=}, {max(items)=}")

# Method calls
print(f"{model.training=}, {optimizer.state_dict().keys()=}")

# Nested attributes
print(f"{config.model.num_layers=}")
```

---

### 2.2 Conditional Debug Flags

**Environment Variable Pattern:**

```python
import os

DEBUG = os.environ.get("DEBUG", "0") == "1"
VERBOSE = os.environ.get("VERBOSE", "0") == "1"

if DEBUG:
    print(f"Processing batch: {batch_idx=}, {batch.shape=}")

if VERBOSE:
    print(f"Full data: {batch}")
```

**Usage:**
```bash
# Enable debugging
DEBUG=1 python train.py

# Enable multiple flags
DEBUG=1 VERBOSE=1 python train.py
```

**StreamDiffusionTD-Custom Validation:**
```python
# From debug_utils.py:25-26
MEMORY_DEBUG = os.environ.get("SD_MEMORY_DEBUG", "0") == "1"
DETECT_ANOMALIES = os.environ.get("SD_DETECT_ANOMALIES", "0") == "1"
```

---

### 2.3 Boolean Parameter Debug Flags

```python
def process_data(
    data,
    debug: bool = False,
    verbose: bool = False,
    log_memory: bool = False
):
    if debug:
        print(f"[DEBUG] Input shape: {data.shape}")

    result = compute(data)

    if verbose:
        print(f"[VERBOSE] Result: {result}")

    if log_memory:
        import psutil
        mem = psutil.virtual_memory()
        print(f"[MEMORY] Used: {mem.percent}%")

    return result
```

**StreamDiffusionTD-Custom Validation:**
```python
# From main_sdtd.py - Parameter-based debug flags
def prepare(
    self,
    ...,
    controlnetdebug: bool = False,
    performancedebug: bool = False,
    queuedebug: bool = False,
):
```

---

### 2.4 Colored Output for Visibility

```python
# ANSI color codes
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"

print(f"{RED}Error: Connection failed{RESET}")
print(f"{GREEN}Success: Model loaded{RESET}")
print(f"{YELLOW}Warning: Using fallback method{RESET}")
```

**StreamDiffusionTD-Custom Validation:**
```python
# From main_sdtd.py:76
print(f"\033[33mWarning: Requested GPU {gpu_id} not available...\033[0m")
```

---

## 3. Package and Import Debugging

### 3.1 Verifying Correct Package is Loaded

**Problem:** Multiple installations of same package (system, venv, editable)

**Solution 1: Check import path**
```python
import mypackage
print(f"Package location: {mypackage.__file__}")
print(f"Package version: {mypackage.__version__}")
```

**Solution 2: Inspect sys.path**
```python
import sys
print("Python path order:")
print("\n".join(sys.path))
```

**Solution 3: Intentionally break the file**
```python
# In the file you think is being imported
def main():
    die  # NameError will confirm this file is loaded
    x = 5
```

---

### 3.2 Editable Installs

**Install package in development mode:**
```bash
# Navigate to package root (where pyproject.toml or setup.py is)
pip install -e .

# With extras
pip install -e ".[dev,test]"
```

**Verify editable install:**
```bash
pip list | grep mypackage
# Output should show: mypackage @ file:///path/to/repo
```

---

### 3.3 PYTHONPATH Configuration

**Temporary override:**
```bash
# Single directory
PYTHONPATH=/path/to/my/repo/src python script.py

# Multiple directories (Unix)
PYTHONPATH=/path/one:/path/two python script.py

# Multiple directories (Windows)
set PYTHONPATH=C:\path\one;C:\path\two
python script.py
```

**Permanent configuration (Unix):**
```bash
# Add to ~/.bashrc or ~/.zshrc
export PYTHONPATH="/path/to/repo/src:$PYTHONPATH"
```

---

### 3.4 Import Fallback Pattern

```python
# Try importing optimized version, fall back to standard
try:
    from mypackage_cuda import fast_function
    print("[INFO] Using CUDA-accelerated version")
except ImportError:
    from mypackage import fast_function
    print("[WARN] CUDA version not available, using CPU fallback")
```

**StreamDiffusionTD-Custom Validation:**
```python
# From attention_processor.py:23-32
try:
    from v2v_utils_hybrid import TOMESD_AVAILABLE, create_merge_functions
    print("[INFO] Using hybrid v2v_utils with tomesd support")
except ImportError:
    from v2v_utils import get_nn_feats, random_bipartite_soft_matching
    TOMESD_AVAILABLE = False
    print("[WARN] Using original v2v_utils")
```

---

## 4. Process Inspection

### 4.1 py-spy: Process Profiling

**Installation:**
```bash
pip install py-spy
```

**Use Case 1: Hanging Process**
```bash
# Get process ID
ps aux | grep python

# Dump stack trace
py-spy dump --pid PID

# Include C/C++ extensions
py-spy dump -n --pid PID
```

**Use Case 2: Multi-Process Debugging**
```bash
# Dump all Python processes
pgrep python | xargs -I {} py-spy dump --pid {}

# Skip launcher process (get child processes only)
pgrep -P $(pgrep -o python) | xargs -I {} py-spy dump --pid {}
```

**Use Case 3: Live Profiling**
```bash
# Top-like interface
py-spy top --pid PID

# Record to file
py-spy record -o profile.svg --pid PID
# Open profile.svg in browser
```

---

### 4.2 Detecting Deadlocks

**Pattern 1: Print before/after locks**
```python
import threading

lock = threading.Lock()

print("[LOCK] Attempting to acquire")
with lock:
    print("[LOCK] Acquired successfully")
    # Critical section
print("[LOCK] Released")
```

**Pattern 2: Timeout on locks**
```python
if lock.acquire(timeout=5.0):
    try:
        # Critical section
        pass
    finally:
        lock.release()
else:
    print("[ERROR] Failed to acquire lock after 5 seconds - deadlock?")
```

---

## 5. Test Suite Configuration

### 5.1 Test Path Resolution with conftest.py

**Problem:** Tests can't find local package

**Solution:** Create `tests/conftest.py`

```python
import sys
from pathlib import Path

# Add git repo root to sys.path
git_repo_path = str(Path(__file__).resolve().parents[1])
sys.path.insert(1, git_repo_path)

print(f"[TEST] Using package from: {git_repo_path}")
```

---

### 5.2 pytest Aliases

**Add to `~/.bashrc` or `~/.zshrc`:**

```bash
# Basic pytest with useful flags
alias pyt="pytest --disable-warnings --instafail -rA"

# Show test collection without running
alias pytc="pytest --disable-warnings --collect-only -q"

# Run specific test verbosely
alias pytv="pytest --disable-warnings -vv -s"

# Stop on first failure
alias pytx="pytest --disable-warnings -x"
```

**Usage:**
```bash
pyt tests/                     # Run all tests
pyt tests/test_model.py        # Run specific file
pyt tests/test_model.py::test_forward  # Run specific test
pytv tests/ -k "cuda"          # Run tests matching "cuda"
```

---

### 5.3 pytest Debugging Tips

**Drop into debugger on failure:**
```bash
pytest --pdb tests/
```

**Drop into debugger on error (not assertion):**
```bash
pytest --pdbcls=IPython.terminal.debugger:TerminalPdb --pdb tests/
```

**Print output even on success:**
```bash
pytest -s tests/
```

---

## 6. Debugger Usage

### 6.1 pdb Configuration

**Create `~/.pdbrc`:**
```
# Aliases for common operations
alias nl n;;l   # Next + List
alias sl s;;l   # Step + List

# Pretty-print dictionary
alias pd for k in sorted(%1.keys()): print "%s%-15s= %-80.80s" % ("%2",k,repr(%1[k]))

# Show all local variables
alias ll pp locals()

# Continue until line N
alias tl tbreak %1;;c
```

**Basic pdb commands:**
```python
import pdb; pdb.set_trace()  # Python 3.6 and older
breakpoint()                 # Python 3.7+
```

**Interactive commands:**
```
l        # List code
n        # Next line
s        # Step into function
c        # Continue until breakpoint
p var    # Print variable
pp var   # Pretty-print variable
w        # Where (stack trace)
q        # Quit
```

---

### 6.2 IPython embed for Interactive Debugging

**Usage:**
```python
a = 5
b = [1, 2, 3]

from IPython import embed; embed()
# Interactive shell opens here - inspect variables
# Press Ctrl-D to exit and continue execution

print("Execution continues after embed()")
```

**Benefits:**
- Full IPython features (tab completion, magic commands)
- Can modify variables and see effect
- Can test fixes interactively before editing code

---

### 6.3 Post-Mortem Debugging

**Automatically debug on exception:**
```python
import sys
import pdb

def main():
    # Your code that might fail
    result = 10 / 0  # ZeroDivisionError

if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        pdb.post_mortem()
```

---

### 6.4 gdb for C Extension Segfaults

**Enable core dumps:**
```bash
ulimit -c unlimited
```

**Run Python under gdb:**
```bash
gdb -ex r --args python script.py

# When segfault occurs:
# 1. Press 'c' + Enter
# 2. Type 'bt' (backtrace) + Enter
# 3. Type 'c' + Enter to continue
```

**Run pytest under gdb:**
```bash
gdb -ex r --args python -m pytest -sv tests/test_failing.py
```

---

## 7. Shell Efficiency

### 7.1 Command Aliases

**Add to `~/.bashrc` or `~/.zshrc`:**

```bash
# Python shortcuts
alias py="python"
alias py3="python3"
alias ipy="ipython"

# Virtual environment
alias va="source venv/bin/activate"
alias vd="deactivate"

# Package management
alias pipi="pip install"
alias pipu="pip install --upgrade"
alias pipdev="pip install -e '.[dev]'"

# Git shortcuts
alias gs="git status"
alias gd="git diff"
alias gl="git log --oneline -10"

# List with details
alias l="ls -lah"
alias lt="ls -lahtr"  # Sort by time
```

---

### 7.2 Bash History Search

**Add to `~/.inputrc`:**
```
"\e[A": history-search-backward
"\e[B": history-search-forward
```

**Usage:**
- Type beginning of command (e.g., `pytest tests/`)
- Press Up arrow to search backward through history
- Press Down arrow to search forward

---

### 7.3 Navigation Shortcuts

**Keyboard shortcuts (works in most terminals):**

```
Ctrl+a    # Move to beginning of line
Ctrl+e    # Move to end of line
Alt+f     # Move forward one word
Alt+b     # Move backward one word
Ctrl+u    # Delete from cursor to beginning
Ctrl+k    # Delete from cursor to end
Ctrl+w    # Delete word before cursor
Ctrl+r    # Reverse search history
```

---

### 7.4 Watch Commands for Real-Time Monitoring

**Monitor GPU usage:**
```bash
watch -n 1 nvidia-smi
```

**Monitor disk space:**
```bash
watch -n 5 'df -h | grep /tmp'
```

**Monitor Python processes:**
```bash
watch -n 2 'ps aux | grep python'
```

**Monitor specific file:**
```bash
watch -n 1 'cat /tmp/training_log.txt | tail -20'
```

**Using htop for process monitoring:**
```bash
# Install htop if not available
sudo apt install htop  # Ubuntu/Debian
brew install htop      # macOS

# Monitor Python processes sorted by memory
htop -F python -s M_RESIDENT -u $(whoami)
```

---

## 8. Finding Breaking Changes

### 8.1 Git Bisect

**Purpose:** Find which commit introduced a bug

**Full workflow:**

```bash
# 1. Start bisect
git bisect start

# 2. Mark current commit as bad
git bisect bad HEAD

# 3. Mark known good commit
git bisect good 5a4f340d  # Use commit hash or tag

# 4. Test the current code
python test.py  # or run your test

# 5. Mark as good or bad
git bisect good   # if test passes
# OR
git bisect bad    # if test fails

# 6. Repeat step 4-5 until bisect finds the culprit
# Git will automatically checkout different commits

# 7. When done, reset
git bisect reset
```

**Automated bisect with test script:**
```bash
git bisect start HEAD v1.0.0
git bisect run pytest tests/test_feature.py
git bisect reset
```

---

### 8.2 Comparing Package Versions

**Check installed version:**
```bash
pip list | grep torch
python -c "import torch; print(torch.__version__)"
```

**Install specific version:**
```bash
pip install torch==2.0.0
```

**Test multiple versions:**
```bash
# Create separate environments
python -m venv venv_old
source venv_old/bin/activate
pip install torch==1.13.0
python test.py  # Does it work?
deactivate

python -m venv venv_new
source venv_new/bin/activate
pip install torch==2.0.0
python test.py  # Does it work?
deactivate
```

---

## Checklist: General Debugging Workflow

### When encountering a bug:

- [ ] **Step 1: Reproduce reliably**
  - [ ] Create minimal reproduction case
  - [ ] Use small data/models
  - [ ] Document exact steps to reproduce

- [ ] **Step 2: Isolate the problem**
  - [ ] Remove unrelated code
  - [ ] Test individual components
  - [ ] Use one-liner programs

- [ ] **Step 3: Add strategic prints**
  - [ ] Before/after problematic code
  - [ ] Use `f"{var=}"` syntax
  - [ ] Print shapes, types, values

- [ ] **Step 4: Check assumptions**
  - [ ] Verify correct file is loaded
  - [ ] Check package versions
  - [ ] Validate input data

- [ ] **Step 5: Use appropriate tools**
  - [ ] py-spy for hanging processes
  - [ ] pdb for logic errors
  - [ ] git bisect for regressions
  - [ ] gdb for segfaults

- [ ] **Step 6: Document the fix**
  - [ ] Comment why the fix works
  - [ ] Add test to prevent regression
  - [ ] Update documentation if needed

---

## Quick Reference: Command Cheatsheet

```bash
# Process inspection
ps aux | grep python              # Find Python processes
py-spy dump -n --pid PID          # Stack trace with C extensions
kill -SIGTERM PID                 # Gracefully stop process
kill -9 PID                       # Force kill process

# Package debugging
pip list | grep package           # Check installed version
pip show package                  # Show package details
python -c "import pkg; print(pkg.__file__)"  # Find package location

# Testing
pytest -v tests/                  # Verbose test output
pytest -s tests/                  # Show print statements
pytest --pdb tests/               # Drop to debugger on failure
pytest -k "keyword" tests/        # Run tests matching keyword

# Git
git status                        # Check working tree
git diff                          # Show uncommitted changes
git log --oneline -10             # Last 10 commits
git bisect start                  # Start bisect

# System monitoring
nvidia-smi                        # GPU usage
htop                              # CPU/memory usage
df -h                             # Disk space
free -h                           # Memory usage
```

---

## Further Reading

- **py-spy documentation:** https://github.com/benfred/py-spy
- **pytest documentation:** https://docs.pytest.org/
- **pdb tutorial:** https://docs.python.org/3/library/pdb.html
- **Git bisect guide:** https://git-scm.com/docs/git-bisect

---

**License:** CC-BY-SA 4.0
**Source:** Based on "The Art of Debugging" by Stas Bekman
