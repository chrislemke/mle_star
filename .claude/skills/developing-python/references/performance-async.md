# Performance & Async — Python Reference

Profiling, optimization, and async/concurrency patterns.

## Contents

- [Profiling](#profiling)
- [Common Optimizations](#common-optimizations)
- [Async/Await Patterns](#asyncawait-patterns)
- [Concurrency Models](#concurrency-models)
- [Data Structure Selection](#data-structure-selection)

## Profiling

**Always profile before optimizing.** Intuition about bottlenecks is unreliable.

```python
# Quick function timing
import cProfile
cProfile.run("main()", sort="cumulative")

# Line-by-line profiling (pip install line-profiler)
# Add @profile decorator, then run:
# kernprof -l -v script.py

# Memory profiling (pip install memory-profiler)
# @profile decorator, then: python -m memory_profiler script.py

# Production profiling with py-spy (no code changes needed)
# py-spy record -o profile.svg -- python script.py
```

**Rules**:
- Profile with realistic data sizes, not toy examples
- Measure wall time and CPU time separately
- Profile before AND after changes to verify improvement
- Focus on the hottest path first — 80/20 rule applies

## Common Optimizations

### Use Generators for Single-Pass Processing

```python
# Bad: builds a full intermediate list
total = sum([compute(x) for x in large_dataset])

# Good: generator expression, constant memory
total = sum(compute(x) for x in large_dataset)

# Good: generator pipeline
def read_records(path: Path) -> Generator[Record]:
    with path.open() as f:
        for line in f:
            yield parse_record(line)

def filter_active(records: Iterable[Record]) -> Generator[Record]:
    yield from (r for r in records if r.active)

active = filter_active(read_records(data_path))
```

### Choose Efficient Data Structures

```python
# Membership testing: set > list
valid_ids: set[str] = {u.id for u in users}
if user_id in valid_ids: ...  # O(1) vs O(n) with list

# Counting: Counter > manual dict
from collections import Counter
word_freq = Counter(words)

# Sorted data: bisect for binary search
import bisect
index = bisect.bisect_left(sorted_list, target)

# Named access: dataclass > dict (faster attribute access, type-safe)
```

### Avoid Common Performance Traps

```python
# Bad: string concatenation in loop — O(n^2)
result = ""
for item in items:
    result += str(item)

# Good: join — O(n)
result = "".join(str(item) for item in items)

# Bad: repeated dict/attr lookups in tight loop
for item in items:
    self.config.settings.threshold  # 3 lookups per iteration

# Good: local variable binding
threshold = self.config.settings.threshold
for item in items:
    ... threshold ...

# Use __slots__ on frequently instantiated classes
@dataclass(slots=True)
class Point:
    x: float
    y: float
```

### Use `functools` Effectively

```python
from functools import lru_cache, cache

# Cache expensive computations (bounded)
@lru_cache(maxsize=256)
def expensive_lookup(key: str) -> Result: ...

# Unbounded cache for pure functions with hashable args
@cache
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
```

## Async/Await Patterns

### Basic Pattern

```python
import asyncio
import httpx

async def fetch_user(client: httpx.AsyncClient, user_id: str) -> User:
    response = await client.get(f"/users/{user_id}")
    response.raise_for_status()
    return User(**response.json())

async def main() -> None:
    async with httpx.AsyncClient(base_url=API_URL) as client:
        users = await asyncio.gather(
            fetch_user(client, "1"),
            fetch_user(client, "2"),
            fetch_user(client, "3"),
        )
```

### Structured Concurrency with TaskGroups (3.11+)

```python
async def process_batch(items: list[str]) -> list[Result]:
    results: list[Result] = []

    async with asyncio.TaskGroup() as tg:
        for item in items:
            tg.create_task(process_item(item))

    # All tasks complete or all are cancelled on first exception
    return results
```

**Prefer `TaskGroup` over `gather`** — it provides proper cancellation and exception handling.

### Async Context Managers and Iterators

```python
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

@asynccontextmanager
async def db_session() -> AsyncGenerator[Session]:
    session = await create_session()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()

# Async iteration
async def stream_records(query: str) -> AsyncGenerator[Record]:
    async with db_session() as session:
        cursor = await session.execute(query)
        async for row in cursor:
            yield Record(**row)
```

### Semaphore for Rate Limiting

```python
async def fetch_all(urls: list[str], max_concurrent: int = 10) -> list[str]:
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_one(url: str) -> str:
        async with semaphore:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url)
                return resp.text

    return await asyncio.gather(*(fetch_one(url) for url in urls))
```

## Concurrency Models

| Model | Best For | GIL Impact | Example |
|-------|----------|------------|---------|
| `asyncio` | I/O-bound (network, files) | Not affected | HTTP clients, web servers |
| `threading` | Legacy blocking I/O | Limited by GIL | File watchers, simple parallelism |
| `multiprocessing` | CPU-bound computation | Bypasses GIL | Data processing, image transforms |
| `concurrent.futures` | Simple parallel dispatch | Both modes | Map/submit patterns |

```python
# CPU-bound: use ProcessPoolExecutor
from concurrent.futures import ProcessPoolExecutor

def heavy_compute(data: bytes) -> Result:
    ...  # CPU-intensive work

with ProcessPoolExecutor() as pool:
    results = list(pool.map(heavy_compute, data_chunks))

# Mix async + CPU: run_in_executor
async def process(data: bytes) -> Result:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, heavy_compute, data)
```

## Data Structure Selection

| Need | Use | Why |
|------|-----|-----|
| Ordered collection | `list` | Fast append/index |
| Unique elements | `set` | O(1) membership test |
| Key-value mapping | `dict` | O(1) lookup (ordered since 3.7) |
| FIFO queue | `collections.deque` | O(1) both ends |
| Counting | `collections.Counter` | Optimized counting |
| Default values | `collections.defaultdict` | No KeyError handling |
| Immutable sequence | `tuple` | Hashable, lighter than list |
| Sorted container | `sortedcontainers.SortedList` | O(log n) insert + search |
| Named fields | `@dataclass` or `NamedTuple` | Type-safe, readable |
