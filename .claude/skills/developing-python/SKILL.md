---
name: developing-python
description: >-
  Writes production-grade Python code following senior-level best practices.
  Applies SOLID principles, modern type hints (Python 3.12+), proper error
  handling, design patterns, and security-aware coding. Use when writing Python
  code, reviewing Python, refactoring Python, designing Python architecture,
  implementing classes, writing functions, adding type hints, handling errors,
  optimizing performance, or when the user mentions Python, .py files, pytest,
  mypy, async, dataclasses, or asks for "clean code", "best practices",
  "production-ready", "senior-level", or "Pythonic" code.
---

# Developing Python — Senior-Level Practices

Write clean, maintainable, production-grade Python following modern best practices.

## Core Principles

1. **Readability over cleverness** — Code is read far more than written. Favor explicit over implicit (PEP 20)
2. **Type everything** — Use modern type hints everywhere. Catch bugs at lint time, not runtime
3. **Fail fast, fail loud** — Validate at boundaries, raise specific exceptions, never swallow errors silently
4. **Composition over inheritance** — Prefer protocols and composition. Use inheritance only for true "is-a" relationships
5. **Immutability by default** — Use `frozen=True` dataclasses, tuples, `Final`, and `ReadOnly` where possible

## Modern Python Style (3.12+)

### Type Hints

```python
# Use builtin generics — not typing.List, typing.Dict, typing.Optional
def process(items: list[str], config: dict[str, int] | None = None) -> bool: ...

# Use X | Y union syntax — not Union[X, Y] or Optional[X]
def fetch(url: str) -> Response | None: ...

# Use PEP 695 type parameter syntax for generics
type Vector[T] = list[T]

def first[T](items: Sequence[T]) -> T: ...

# Parameters: accept abstract types (Sequence, Mapping, Iterable)
# Returns: use concrete types (list, dict)
def transform(data: Iterable[int]) -> list[int]: ...
```

### Data Modeling

```python
from dataclasses import dataclass, field

@dataclass(frozen=True, slots=True)
class Config:
    host: str
    port: int = 8080
    tags: tuple[str, ...] = ()
```

Use `@dataclass(frozen=True, slots=True)` by default. Use Pydantic `BaseModel` for validation at system boundaries (API input, config files, external data).

### Pattern Matching

```python
match command:
    case {"action": "create", "name": str(name)}:
        create(name)
    case {"action": "delete", "id": int(id_)}:
        delete(id_)
    case _:
        raise ValueError(f"Unknown command: {command}")
```

## SOLID Principles in Python

- **Single Responsibility**: Each class/module has one reason to change. Split file I/O from business logic from formatting
- **Open/Closed**: Extend via protocols and ABC, not modification. Add new behavior through new classes implementing existing protocols
- **Liskov Substitution**: Subclasses must honor the parent's contract — no surprise side effects or narrowed inputs
- **Interface Segregation**: Use `Protocol` classes with small, focused interfaces. A class needing only `read()` should not depend on a `ReadWriteStore`
- **Dependency Inversion**: Depend on abstractions (Protocol, ABC), inject concrete implementations

```python
from typing import Protocol

class Repository(Protocol):
    def get(self, id: str) -> Item: ...
    def save(self, item: Item) -> None: ...

class UserService:
    def __init__(self, repo: Repository) -> None:
        self._repo = repo
```

## Error Handling

```python
# Define domain-specific exceptions
class AppError(Exception):
    """Base for all application errors."""

class NotFoundError(AppError):
    def __init__(self, entity: str, id: str) -> None:
        super().__init__(f"{entity} not found: {id}")

# Catch specific exceptions — never bare except
try:
    user = repo.get(user_id)
except NotFoundError:
    logger.warning("User %s not found", user_id)
    raise
except DatabaseError as exc:
    logger.exception("DB failure fetching user %s", user_id)
    raise ServiceError("Storage unavailable") from exc
```

**Rules**:
- Never use bare `except:` or `except Exception:` unless re-raising
- Chain exceptions with `raise ... from exc` to preserve tracebacks
- Log at the handling site, not at the raising site
- Let unexpected exceptions propagate — don't catch what you can't handle
- Use context managers for resource cleanup, not try/finally

## Design Patterns

Apply when they solve a real problem, not preemptively:

| Pattern | When to Use | Python Idiom |
|---------|-------------|--------------|
| Strategy | Swappable algorithms | Functions as params or Protocol |
| Factory | Complex construction | `@classmethod` or factory function |
| Observer | Event notification | Callbacks, `signal` libs |
| Decorator | Cross-cutting concerns | `@functools.wraps` decorators |
| Iterator | Lazy sequences | Generators with `yield` |
| Context Manager | Resource lifecycle | `@contextmanager` or `__enter__/__exit__` |
| Singleton | Single shared instance | Module-level instance |

For detailed patterns and examples: See [references/design-patterns.md](references/design-patterns.md)

## Performance & Concurrency

- **Profile first** — Use `cProfile` / `py-spy` before optimizing. Never guess bottlenecks
- **Choose the right concurrency model**:
  - `asyncio` for I/O-bound (network, files) — the default choice
  - `multiprocessing` for CPU-bound (computation, data processing)
  - `threading` only for legacy blocking I/O with no async alternative
- **Use efficient data structures** — `set` for membership, `deque` for queues, `defaultdict` to avoid key checks
- **Generator over list** when items are consumed once: `sum(x**2 for x in data)` not `sum([x**2 for x in data])`
- **`__slots__`** on classes instantiated many times

For async patterns and performance details: See [references/performance-async.md](references/performance-async.md)

## Security

- **Never** use `eval()`, `exec()`, or `pickle` on untrusted data
- **Always** parameterize SQL/queries — never string-format user input
- **Validate** all external input at system boundaries using Pydantic or manual checks
- **Hash** passwords with `bcrypt` or `argon2` — never roll your own crypto
- **Pin** dependencies and audit with `pip-audit` or `safety`
- **Secrets** via environment variables or vaults — never in source code
- Use `secrets` module for tokens, not `random`

For the full security checklist: See [references/security-checklist.md](references/security-checklist.md)

## Code Organization

```
src/
  package_name/
    __init__.py          # Public API only
    domain/              # Business logic, no I/O
    services/            # Orchestration, uses domain + adapters
    adapters/            # External systems (DB, API, filesystem)
    models.py            # Data classes, Pydantic models
    exceptions.py        # Domain exceptions hierarchy
    protocols.py         # Abstract interfaces
tests/
  conftest.py            # Shared fixtures
  unit/                  # Fast, no I/O
  integration/           # With real dependencies
```

## Quick Reference

| Topic | Preferred | Avoid |
|-------|-----------|-------|
| Type unions | `str \| None` | `Optional[str]`, `Union[str, None]` |
| Generics | `list[int]` | `List[int]` from typing |
| Data classes | `@dataclass(frozen=True, slots=True)` | Plain dicts for structured data |
| String format | f-strings | `.format()`, `%` |
| Path handling | `pathlib.Path` | `os.path` |
| Iteration | Generator expressions | Building intermediate lists |
| Imports | Absolute imports | Relative imports (except within package) |
| Constants | `Final[int] = 42` at module level | Magic numbers inline |
| Enums | `StrEnum` / `IntEnum` | String constants |
