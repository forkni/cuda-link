---
model: claude-opus-4-1
---

# Smart Debugging & Issue Resolution

**Purpose**: Intelligent debugging with automatic routing to appropriate analysis approach.
**Time**: 15-60 minutes depending on issue complexity
**Focus**: Root cause analysis, targeted fixes, verification

## When to Use This Command

✅ **Good use cases:**

- Complex bugs with unclear root cause
- Performance issues needing profiling
- Integration failures across components
- Mysterious errors without obvious cause

❌ **Just debug manually if:**

- Error message clearly shows the problem
- Simple syntax/typo errors
- Stack trace points to exact line
- You already know what's wrong

## Issue to Debug

$ARGUMENTS

---

## Analysis Phase

**Examine the issue and categorize:**

1. **Error type**: Exception, performance, logic, integration?
2. **Scope**: Single function, module, or cross-component?
3. **Reproducibility**: Always, intermittent, or specific conditions?
4. **Impact**: Critical (blocks work) vs minor (inconvenience)?

---

## Common Python/ML Debugging Patterns

### Pattern 1: PyTorch/CUDA Errors

**Symptoms:**

- `RuntimeError: CUDA out of memory`
- `RuntimeError: CUDA error: device-side assert triggered`
- Model returns NaN/Inf values

**Quick diagnosis:**

```python
import torch

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Check memory usage
if torch.cuda.is_available():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"GPU memory cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

# Debug NaN/Inf
torch.autograd.set_detect_anomaly(True)  # Slower but finds NaN source
```

**Common fixes:**

- Reduce batch size
- Clear GPU cache: `torch.cuda.empty_cache()`
- Use gradient accumulation
- Check for division by zero
- Verify input data (no NaN/Inf)

### Pattern 2: FAISS Index Errors

**Symptoms:**

- `RuntimeError: Index not trained`
- Slow search performance
- Inconsistent results

**Quick diagnosis:**

```python
# Check index status
print(f"Index trained: {index.is_trained}")
print(f"Index size: {index.ntotal}")

# Test search
distances, indices = index.search(test_vector, k=5)
print(f"Distances: {distances}")
print(f"Valid indices: {(indices >= 0).all()}")
```

**Common fixes:**

- Train index before adding vectors
- Verify vector dimensionality matches
- Check nprobe settings
- Rebuild index if corrupted

### Pattern 3: Tree-sitter Parser Failures

**Symptoms:**

- Parser returns None
- Missing nodes in AST
- Incorrect syntax detection

**Quick diagnosis:**

```python
import tree_sitter

# Test parser
tree = parser.parse(bytes(code, 'utf8'))
print(f"Has errors: {tree.root_node.has_error}")
print(f"Root type: {tree.root_node.type}")

# Check for error nodes
def find_errors(node):
    if node.type == 'ERROR':
        print(f"Error at {node.start_point}: {node.text}")
    for child in node.children:
        find_errors(child)

find_errors(tree.root_node)
```

**Common fixes:**

- Verify language parser loaded correctly
- Check for incomplete/malformed code
- Update parser to latest version
- Handle syntax errors gracefully

### Pattern 4: Memory Leaks

**Symptoms:**

- Memory usage grows over time
- OOM after many iterations
- Slow garbage collection

**Quick diagnosis:**

```python
import tracemalloc
import gc

# Start tracking
tracemalloc.start()

# Your code here
for i in range(100):
    process_batch(data[i])

# Show top memory consumers
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)

# Force garbage collection
gc.collect()
```

**Common fixes:**

- Delete large objects explicitly
- Clear PyTorch cache
- Use context managers
- Break circular references
- Profile with `memory_profiler`

### Pattern 5: Slow Performance

**Symptoms:**

- Operations take much longer than expected
- High CPU/GPU usage
- Unresponsive application

**Quick diagnosis:**

```python
import cProfile
import pstats

# Profile the slow function
profiler = cProfile.Profile()
profiler.enable()

slow_function(args)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

**Common fixes:**

- Use `/performance-optimization` command
- Batch operations
- Add caching
- Use better algorithms
- Parallelize if possible

---

## Debugging Decision Tree

```
Issue reported
      ↓
Can you reproduce it?
  NO → Add logging, get repro steps
  YES ↓
      ↓
Is error message clear?
  YES → Fix directly (don't use this command)
  NO ↓
      ↓
Is it performance-related?
  YES → Use /performance-optimization
  NO ↓
      ↓
Is it ML/PyTorch/FAISS specific?
  YES → Use patterns above
  NO ↓
      ↓
Complex multi-component issue?
  YES → Use this command (smart routing)
  NO → Debug manually with prints/pdb
```

---

## Agent Routing (for Complex Issues)

### For Python Code Errors

**When to use:**

- Stack traces span multiple files
- Error message is cryptic
- Intermittent failures

**Approach:**

- Use Task tool with subagent_type="debugger"
- Provide full stack trace
- Include recent code changes
- Share relevant context

### For ML Model Issues

**When to use:**

- Model not converging
- Unexpected predictions
- Training instability

**Approach:**

- Use Task tool with subagent_type="performance-engineer"
- Include model architecture
- Share training logs
- Provide data samples

### For Integration Issues

**When to use:**

- Components work separately but fail together
- API contract violations
- Data format mismatches

**Approach:**

- Use Task tool with subagent_type="debugger"
- Document component interfaces
- Show data flow
- Include integration points

---

## Quick Debugging Checklist

**Before deep debugging:**

- [ ] Can you reproduce the issue consistently?
- [ ] Have you checked the error message carefully?
- [ ] Have you looked at recent code changes?
- [ ] Have you checked logs for clues?
- [ ] Have you tried the simplest fix first?

**During debugging:**

- [ ] Add strategic print statements / logging
- [ ] Verify assumptions with assertions
- [ ] Test in isolation (unit test)
- [ ] Check edge cases
- [ ] Review recent changes

**After fixing:**

- [ ] Add test to prevent regression
- [ ] Document the fix
- [ ] Check for similar issues elsewhere
- [ ] Update error messages if needed

---

## Python Debugging Tools

### Built-in Debugger (pdb)

```python
import pdb

def buggy_function(data):
    pdb.set_trace()  # Debugger stops here
    # Step through code interactively
    result = process(data)
    return result

# Commands:
# n - next line
# s - step into function
# c - continue execution
# p variable - print variable
# q - quit debugger
```

### IPython Debugger (ipdb)

```bash
pip install ipdb

# Use ipdb instead of pdb for better experience
import ipdb; ipdb.set_trace()
```

### Post-mortem Debugging

```python
import pdb
import sys

def main():
    try:
        buggy_function()
    except Exception:
        # Drop into debugger at exception point
        pdb.post_mortem(sys.exc_info()[2])
```

### Logging for Debugging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def function():
    logger.debug(f"Input: {input}")
    logger.info(f"Processing...")
    logger.warning(f"Unexpected value: {value}")
    logger.error(f"Failed: {error}")
```

---

## When to Stop Debugging

**Stop and rethink if:**

- Debugging for >2 hours without progress
- Adding more print statements than code
- Issue only happens in production (need better logging)
- Root cause still unclear after extensive investigation

**Better approaches:**

- Add comprehensive logging first
- Write tests to isolate issue
- Simplify code to find minimal repro
- Pair program with someone else
- Take a break and return fresh

---

## Summary

**Smart debugging workflow:**

1. **Categorize** → What type of issue?
2. **Reproduce** → Can you trigger it consistently?
3. **Isolate** → Narrow down to smallest repro
4. **Diagnose** → Use appropriate tools/patterns
5. **Fix** → Apply targeted solution
6. **Verify** → Confirm fix works
7. **Prevent** → Add tests

**Time per issue:**

- Simple bugs: 5-15 minutes
- Complex bugs: 30-60 minutes
- Mysterious issues: 1-4 hours (consider getting help)

**Remember:** The best debugging is preventing bugs with tests (`/tdd-cycle`) and good error handling.
