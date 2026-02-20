# Design Patterns — Python Reference

Practical design patterns using Pythonic idioms. Apply when they solve a real problem.

## Contents

- [Strategy](#strategy)
- [Factory](#factory)
- [Observer](#observer)
- [Decorator](#decorator)
- [Context Manager](#context-manager)
- [Repository](#repository)
- [Builder](#builder)

## Strategy

Swap algorithms at runtime via functions or protocols.

```python
from typing import Protocol
from collections.abc import Callable

# Function-based (simplest)
def process(data: list[float], strategy: Callable[[list[float]], float]) -> float:
    return strategy(data)

result = process(values, strategy=statistics.median)

# Protocol-based (when strategy has state or multiple methods)
class Scorer(Protocol):
    def score(self, text: str) -> float: ...
    def name(self) -> str: ...

class SentimentScorer:
    def score(self, text: str) -> float:
        return analyze_sentiment(text)
    def name(self) -> str:
        return "sentiment"
```

**When to use**: Multiple interchangeable algorithms, configurable behavior, policy injection.

## Factory

Encapsulate complex object creation.

```python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Connection:
    host: str
    port: int
    ssl: bool

    @classmethod
    def from_url(cls, url: str) -> "Connection":
        parsed = urlparse(url)
        return cls(
            host=parsed.hostname or "localhost",
            port=parsed.port or 5432,
            ssl=parsed.scheme == "https",
        )

    @classmethod
    def local(cls) -> "Connection":
        return cls(host="localhost", port=5432, ssl=False)

# Factory function — when creation spans multiple classes
def create_storage(config: dict[str, str]) -> Storage:
    match config["backend"]:
        case "s3":
            return S3Storage(bucket=config["bucket"])
        case "local":
            return LocalStorage(path=Path(config["path"]))
        case backend:
            raise ValueError(f"Unknown backend: {backend}")
```

**When to use**: Complex construction logic, multiple creation paths, configuration-driven instantiation.

## Observer

Decouple event producers from consumers.

```python
from collections.abc import Callable
from dataclasses import dataclass, field

@dataclass
class EventBus:
    _listeners: dict[str, list[Callable]] = field(default_factory=dict)

    def on(self, event: str, callback: Callable) -> None:
        self._listeners.setdefault(event, []).append(callback)

    def emit(self, event: str, **kwargs) -> None:
        for callback in self._listeners.get(event, []):
            callback(**kwargs)

# Usage
bus = EventBus()
bus.on("user.created", lambda user_id: send_welcome_email(user_id))
bus.on("user.created", lambda user_id: init_default_settings(user_id))
bus.emit("user.created", user_id="abc-123")
```

**When to use**: Event-driven architectures, plugin systems, decoupled notifications.

## Decorator

Add behavior without modifying the decorated function.

```python
import functools
import time
import logging
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

def retry(max_attempts: int = 3, delay: float = 1.0) -> Callable:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    logging.warning(
                        "Attempt %d/%d failed for %s: %s",
                        attempt, max_attempts, func.__name__, exc,
                    )
                    if attempt < max_attempts:
                        time.sleep(delay * attempt)
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.5)
def fetch_data(url: str) -> dict:
    return httpx.get(url).json()
```

**Always** use `@functools.wraps` to preserve the original function's metadata.

**When to use**: Logging, timing, caching, authentication, validation, rate limiting.

## Context Manager

Guarantee resource cleanup.

```python
from contextlib import contextmanager
from typing import Generator

# Function-based (simpler)
@contextmanager
def timer(label: str) -> Generator[None]:
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logging.info("%s took %.3fs", label, elapsed)

with timer("data processing"):
    process(data)

# Class-based (when you need state)
class DatabaseTransaction:
    def __init__(self, connection: Connection) -> None:
        self._conn = connection

    def __enter__(self) -> "DatabaseTransaction":
        self._conn.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None:
            self._conn.commit()
        else:
            self._conn.rollback()
        return False  # Don't suppress exceptions
```

**When to use**: File handles, DB connections, locks, temporary state, timing, transactions.

## Repository

Abstract data access behind a clean interface.

```python
from typing import Protocol

class UserRepository(Protocol):
    def get(self, user_id: str) -> User: ...
    def save(self, user: User) -> None: ...
    def delete(self, user_id: str) -> None: ...
    def find_by_email(self, email: str) -> User | None: ...

class PostgresUserRepository:
    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def get(self, user_id: str) -> User:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE id = %s", (user_id,)
            ).fetchone()
            if row is None:
                raise NotFoundError("User", user_id)
            return User(**row)

class InMemoryUserRepository:
    """For testing — no database required."""
    def __init__(self) -> None:
        self._store: dict[str, User] = {}

    def get(self, user_id: str) -> User:
        try:
            return self._store[user_id]
        except KeyError:
            raise NotFoundError("User", user_id)
```

**When to use**: Separating business logic from data access, enabling test doubles, supporting multiple backends.

## Builder

Construct complex objects step by step.

```python
from dataclasses import dataclass, field

@dataclass
class QueryBuilder:
    _table: str = ""
    _columns: list[str] = field(default_factory=lambda: ["*"])
    _conditions: list[str] = field(default_factory=list)
    _params: list = field(default_factory=list)
    _limit: int | None = None

    def table(self, name: str) -> "QueryBuilder":
        self._table = name
        return self

    def select(self, *columns: str) -> "QueryBuilder":
        self._columns = list(columns)
        return self

    def where(self, condition: str, *params) -> "QueryBuilder":
        self._conditions.append(condition)
        self._params.extend(params)
        return self

    def limit(self, n: int) -> "QueryBuilder":
        self._limit = n
        return self

    def build(self) -> tuple[str, list]:
        sql = f"SELECT {', '.join(self._columns)} FROM {self._table}"
        if self._conditions:
            sql += " WHERE " + " AND ".join(self._conditions)
        if self._limit is not None:
            sql += f" LIMIT {self._limit}"
        return sql, self._params

# Usage
query, params = (
    QueryBuilder()
    .table("users")
    .select("id", "name", "email")
    .where("active = %s", True)
    .where("created_at > %s", cutoff_date)
    .limit(100)
    .build()
)
```

**When to use**: Complex object construction with many optional parameters, fluent APIs, query building.
