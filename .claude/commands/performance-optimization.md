---
model: claude-opus-4-1
---

# Python/ML Performance Optimization

**Purpose**: Profile and optimize Python/ML applications with data-driven improvements.
**Time**: 60-120 minutes for comprehensive optimization
**Focus**: Python profiling, PyTorch/FAISS optimization, algorithmic improvements

## When to Use This Command

✅ **Good use cases:**

- Slow response times (>2-3 seconds for typical operations)
- High memory usage (OOM errors, swap thrashing)
- CPU bottlenecks (100% CPU, slow batch processing)
- Before production release (proactive optimization)

❌ **Don't use when:**

- Performance is already acceptable for use case
- Premature optimization (no profiling data yet)
- Optimizing cold paths (rarely executed code)
- Sub-millisecond optimizations without proven need

## Performance Optimization Target

$ARGUMENTS

---

## Phase 1: Profile First (Know Before You Optimize)

### Rule #1: Never Optimize Without Profiling

**Why profiling matters:**

- Intuition is usually wrong about bottlenecks
- 90% of time spent in 10% of code (optimize the 10%)
- Micro-optimizations often have zero real-world impact

### Python Profiling Tools

#### 1. Quick CPU Profiling (cProfile)

```python
# Profile specific function
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
result = slow_function(args)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

```bash
# Profile entire script
python -m cProfile -o profile.stats script.py

# Analyze with snakeviz (visualization)
pip install snakeviz
snakeviz profile.stats
```

**What to look for:**

- `cumtime`: Total time including subcalls (find hotspots)
- `tottime`: Time excluding subcalls (find slow functions)
- `ncalls`: Number of calls (find loops calling expensive functions)

#### 2. Line-by-Line Profiling (line_profiler)

```bash
# Install
pip install line-profiler

# Add @profile decorator to functions
@profile
def slow_function():
    # Code here...

# Run profiler
kernprof -l -v script.py
```

**Example output:**

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    45        100      10000.0    100.0     50.0      embeddings = model.encode(texts)
    46        100       9000.0     90.0     45.0      results = faiss_search(embeddings)
    47        100       1000.0     10.0      5.0      return process_results(results)
```

#### 3. Memory Profiling (memory_profiler)

```bash
# Install
pip install memory-profiler

# Add @profile decorator
@profile
def memory_intensive_function():
    large_data = load_huge_dataset()
    processed = transform(large_data)
    return processed

# Run profiler
python -m memory_profiler script.py
```

### PyTorch/ML Specific Profiling

#### PyTorch Profiler

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True) as prof:
    with record_function("model_inference"):
        output = model(input_tensor)

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export for Chrome trace viewer
prof.export_chrome_trace("trace.json")
```

#### FAISS Profiling

```python
import time

# Measure search performance
start = time.perf_counter()
distances, indices = index.search(query_vectors, k=10)
search_time = time.perf_counter() - start

print(f"FAISS search: {search_time*1000:.2f}ms for {len(query_vectors)} queries")
print(f"Per-query time: {search_time/len(query_vectors)*1000:.2f}ms")
```

---

## Phase 2: Optimize Based on Profiling Data

### A. Python Code Optimization

#### Common Python Bottlenecks

**1. Slow Loops**

```python
# ❌ BAD: String concatenation in loop
result = ""
for item in large_list:
    result += process(item)  # Creates new string each time

# ✅ GOOD: Use list + join
result_list = []
for item in large_list:
    result_list.append(process(item))
result = "".join(result_list)

# ✅ BETTER: List comprehension
result = "".join([process(item) for item in large_list])
```

**2. Inefficient Data Structures**

```python
# ❌ BAD: List membership testing
items = [1, 2, 3, 4, 5, ...1000]
if x in items:  # O(n) lookup

# ✅ GOOD: Set membership testing
items = {1, 2, 3, 4, 5, ...1000}
if x in items:  # O(1) lookup
```

**3. Repeated Function Calls**

```python
# ❌ BAD: Call expensive function repeatedly
for item in items:
    result = expensive_function()  # Same result every time!
    process(item, result)

# ✅ GOOD: Call once, reuse
cached_result = expensive_function()
for item in items:
    process(item, cached_result)
```

### B. PyTorch Optimization

#### 1. GPU Utilization

```python
# Move data to GPU once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ❌ BAD: Moving data in loop
for batch in dataloader:
    inputs = batch['text'].to(device)  # Slow!
    outputs = model(inputs)

# ✅ GOOD: Use pin_memory + non_blocking
dataloader = DataLoader(dataset, pin_memory=True, num_workers=4)
for batch in dataloader:
    inputs = batch['text'].to(device, non_blocking=True)
    outputs = model(inputs)
```

#### 2. Batch Processing

```python
# ❌ BAD: Process one at a time
results = []
for text in texts:
    embedding = model.encode([text])  # Slow!
    results.append(embedding)

# ✅ GOOD: Batch processing
batch_size = 32
results = []
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    embeddings = model.encode(batch)
    results.extend(embeddings)
```

#### 3. Model Optimization

```python
# Use torch.compile (PyTorch 2.0+)
model = torch.compile(model)

# Enable inference mode
with torch.inference_mode():  # Faster than torch.no_grad()
    outputs = model(inputs)

# Use mixed precision (FP16)
from torch.cuda.amp import autocast
with autocast():
    outputs = model(inputs)
```

### C. FAISS Optimization

#### 1. Index Selection

```python
# For < 1M vectors
index = faiss.IndexFlatL2(dimension)  # Exact search, fast enough

# For 1M-10M vectors
index = faiss.IndexIVFFlat(quantizer, dimension, nlist=100)
index.train(training_vectors)

# For > 10M vectors
index = faiss.IndexIVFPQ(quantizer, dimension, nlist=1000, m=8, nbits=8)
index.train(training_vectors)
```

#### 2. Search Optimization

```python
# Adjust nprobe for speed/accuracy tradeoff
index.nprobe = 10  # Default: search 10 clusters
# Higher = more accurate, slower
# Lower = less accurate, faster

# Batch queries
distances, indices = index.search(query_batch, k=10)  # Faster than one-by-one
```

### D. Caching Strategies

#### 1. Function Results (LRU Cache)

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def expensive_computation(input_value):
    # Heavy computation
    return result

# Automatically caches up to 1000 most recent results
```

#### 2. Database/File Results

```python
import pickle
from pathlib import Path

def load_or_compute(cache_path, compute_fn, *args):
    if Path(cache_path).exists():
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    result = compute_fn(*args)
    with open(cache_path, 'wb') as f:
        pickle.dump(result, f)
    return result

# Usage
embeddings = load_or_compute('embeddings.pkl', model.encode, texts)
```

---

## Phase 3: Validate & Benchmark

### Measure Performance Improvement

```python
import time

def benchmark(func, *args, runs=100):
    """Benchmark function with multiple runs"""
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = func(*args)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"Average: {avg*1000:.2f}ms")
    print(f"Min: {min_time*1000:.2f}ms")
    print(f"Max: {max_time*1000:.2f}ms")
    return result

# Compare before/after
print("Before optimization:")
benchmark(slow_version, input_data)

print("\nAfter optimization:")
benchmark(fast_version, input_data)
```

### Performance Testing

```python
import pytest

def test_search_performance():
    """Ensure search completes within time budget"""
    query = generate_test_query()

    start = time.perf_counter()
    results = search_function(query)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.5, f"Search took {elapsed:.2f}s, expected < 0.5s"
    assert len(results) > 0, "No results returned"
```

### Memory Usage Testing

```python
import tracemalloc

def measure_memory(func, *args):
    """Measure peak memory usage"""
    tracemalloc.start()

    result = func(*args)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")
    return result
```

---

## Common Optimization Patterns

### Pattern 1: Lazy Loading

```python
class EmbeddingModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self._model = None

    @property
    def model(self):
        """Load model only when first accessed"""
        if self._model is None:
            self._model = load_model(self.model_path)
        return self._model
```

### Pattern 2: Pre-computation

```python
# ❌ BAD: Compute on every query
def search(query):
    normalized_query = normalize(query)
    embedding = model.encode(normalized_query)
    results = index.search(embedding)
    return results

# ✅ GOOD: Pre-compute index
class SearchEngine:
    def __init__(self, documents):
        self.documents = documents
        # Pre-compute all embeddings once
        self.embeddings = model.encode(documents)
        self.index = build_index(self.embeddings)

    def search(self, query):
        query_embedding = model.encode(query)
        return self.index.search(query_embedding)
```

### Pattern 3: Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# For I/O-bound tasks (file reading, API calls)
with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(io_bound_function, items))

# For CPU-bound tasks (computation)
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(cpu_bound_function, items))
```

---

## ML-Specific Quick Wins

### 1. Model Quantization

```python
# Reduce model size and increase speed
import torch

# Dynamic quantization (easiest)
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Can reduce size by 4x and increase speed by 2-4x
```

### 2. Reduce Batch Size (Memory)

```python
# If running out of memory
# ❌ batch_size = 64  # OOM!
# ✅ batch_size = 16  # Fits in memory
# ✅ gradient_accumulation_steps = 4  # Effective batch size still 64
```

### 3. Use Smaller Models

```python
# ❌ model = "BAAI/bge-large-en-v1.5"  # 1.34B params, slow
# ✅ model = "BAAI/bge-base-en-v1.5"   # 435M params, 3x faster, 95% accuracy
```

---

## Performance Checklist

**Before optimizing:**

- [ ] Profile code to identify real bottlenecks
- [ ] Measure baseline performance
- [ ] Set performance targets (e.g., "< 100ms per query")

**During optimization:**

- [ ] Optimize hot paths (highest cumulative time)
- [ ] Batch processing where possible
- [ ] Cache expensive computations
- [ ] Use appropriate data structures

**After optimizing:**

- [ ] Benchmark improvements (before/after)
- [ ] Verify correctness (results unchanged)
- [ ] Add performance tests
- [ ] Document optimizations

---

## Anti-Patterns to Avoid

❌ **Optimizing cold paths** → Focus on hot paths (90% of runtime)
❌ **Premature optimization** → Profile first, optimize second
❌ **Micro-optimizations** → Focus on algorithmic improvements first
❌ **Breaking correctness** → Never sacrifice correctness for speed
❌ **Optimizing without measuring** → Always benchmark before/after

---

## Tools Quick Reference

```bash
# CPU profiling
python -m cProfile -o profile.stats script.py
pip install snakeviz && snakeviz profile.stats

# Line profiling
pip install line-profiler
kernprof -l -v script.py

# Memory profiling
pip install memory-profiler
python -m memory_profiler script.py

# PyTorch profiling
# (Built into PyTorch, see examples above)

# Benchmarking
pip install pytest-benchmark
pytest tests/test_performance.py --benchmark-only
```

---

## Expected Performance Improvements

**Typical improvements by optimization type:**

| Optimization | Speedup | Effort | Priority |
|--------------|---------|--------|----------|
| Algorithmic improvement (O(n²) → O(n log n)) | 10-100x | High | 🔴 Critical |
| Batch processing | 5-20x | Medium | 🔴 Critical |
| Caching | 2-10x | Low | 🟡 High |
| Vectorization (NumPy) | 2-5x | Medium | 🟡 High |
| Better data structures | 2-5x | Low | 🟡 High |
| Code-level optimization | 1.2-2x | Medium | 🟢 Low |
| Micro-optimizations | 1.0-1.1x | High | ⚪ Skip |

**Focus on high-impact optimizations first** (algorithmic, batching, caching).

---

## Summary

**Quick Performance Optimization Workflow:**

1. **Profile** → Find bottlenecks (cProfile, line_profiler)
2. **Optimize** → Fix hot paths (batching, caching, better algorithms)
3. **Validate** → Benchmark improvements, ensure correctness
4. **Test** → Add performance tests to prevent regressions

**Time allocation:**

- Profiling: 30% of time (essential!)
- Optimization: 50% of time (focus on hot paths)
- Validation: 20% of time (ensure improvements are real)

**Remember:** Profile first, optimize what matters, measure everything.
