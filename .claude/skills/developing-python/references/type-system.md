# Type System — Python Reference

Modern type hints, generics, and static analysis patterns for Python 3.12+.

## Contents

- [Syntax Quick Reference](#syntax-quick-reference)
- [Protocols](#protocols)
- [Generics (PEP 695)](#generics-pep-695)
- [TypeGuard and TypeIs](#typeguard-and-typeis)
- [Advanced Patterns](#advanced-patterns)
- [Mypy Configuration](#mypy-configuration)

## Syntax Quick Reference

```python
# Unions — use pipe syntax (3.10+)
x: int | str
y: str | None               # Not Optional[str]

# Builtin generics — no typing imports needed (3.9+)
items: list[int]
mapping: dict[str, list[int]]
coords: tuple[float, float]
var_tuple: tuple[int, ...]   # Variable-length tuple

# Callable
from collections.abc import Callable
handler: Callable[[str, int], bool]

# Type aliases (3.12+)
type UserId = str
type Matrix[T] = list[list[T]]

# Final and constants
from typing import Final
MAX_RETRIES: Final[int] = 3

# Literal types
from typing import Literal
def set_mode(mode: Literal["read", "write", "append"]) -> None: ...

# Self type (3.11+)
from typing import Self
class Builder:
    def set_name(self, name: str) -> Self: ...
```

## Protocols

Structural subtyping — duck typing with static checking.

```python
from typing import Protocol, runtime_checkable

# Basic protocol
class Renderable(Protocol):
    def render(self) -> str: ...

# Any class with a render() -> str method satisfies this
class HtmlWidget:
    def render(self) -> str:
        return "<div>widget</div>"

def display(item: Renderable) -> None:
    print(item.render())  # Type-safe

display(HtmlWidget())  # Works — HtmlWidget has render() -> str

# Protocol with properties
class Sized(Protocol):
    @property
    def size(self) -> int: ...

# Protocol with generics
class Repository[T](Protocol):
    def get(self, id: str) -> T: ...
    def save(self, entity: T) -> None: ...

# Runtime-checkable (for isinstance checks)
@runtime_checkable
class Closable(Protocol):
    def close(self) -> None: ...

if isinstance(resource, Closable):
    resource.close()
```

**When to use Protocol vs ABC**:
- Protocol: third-party classes you can't modify, structural compatibility, loose coupling
- ABC: you control the hierarchy, need shared implementation, want to enforce registration

## Generics (PEP 695)

Python 3.12+ syntax — no more explicit TypeVar declarations.

```python
# Generic function
def first[T](items: Sequence[T]) -> T:
    return items[0]

# Generic class
class Stack[T]:
    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        return self._items.pop()

# Multiple type parameters
def zip_with[T, U, R](
    xs: Iterable[T],
    ys: Iterable[U],
    func: Callable[[T, U], R],
) -> list[R]:
    return [func(x, y) for x, y in zip(xs, ys)]

# Bounded type variables
from typing import SupportsFloat

def average[T: SupportsFloat](values: Sequence[T]) -> float:
    return sum(float(v) for v in values) / len(values)

# Constrained to specific types
def format_id[T: (str, int)](value: T) -> str:
    return str(value)

# Type aliases with generics
type Result[T] = T | Error
type Handler[T] = Callable[[Request], Result[T]]
```

## TypeGuard and TypeIs

Narrow types in conditional branches.

```python
# TypeIs (3.13+ or typing_extensions) — preferred, narrows both branches
from typing import TypeIs

def is_string_list(val: list[object]) -> TypeIs[list[str]]:
    return all(isinstance(x, str) for x in val)

def process(data: list[object]) -> None:
    if is_string_list(data):
        # data is list[str] here
        print(", ".join(data))
    else:
        # data is list[object] here (properly narrowed)
        ...

# TypeGuard — narrows only the positive branch
from typing import TypeGuard

def is_not_none[T](val: T | None) -> TypeGuard[T]:
    return val is not None
```

**Prefer `TypeIs` over `TypeGuard`** when your Python version supports it.

## Advanced Patterns

### Overload

Define multiple signatures for a single function:

```python
from typing import overload

@overload
def fetch(id: str) -> User: ...
@overload
def fetch(id: list[str]) -> list[User]: ...

def fetch(id: str | list[str]) -> User | list[User]:
    if isinstance(id, str):
        return _fetch_one(id)
    return [_fetch_one(i) for i in id]
```

### ParamSpec for Decorator Typing

```python
from typing import ParamSpec, TypeVar
from collections.abc import Callable

P = ParamSpec("P")
R = TypeVar("R")

def logged(func: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        logging.info("Calling %s", func.__name__)
        return func(*args, **kwargs)
    return wrapper

@logged
def greet(name: str, greeting: str = "Hello") -> str:
    return f"{greeting}, {name}!"

# Type checker knows: greet(name: str, greeting: str = "Hello") -> str
```

### Typed Dictionaries

```python
from typing import TypedDict, Required, NotRequired

class UserData(TypedDict):
    name: str                    # Required by default
    email: str
    age: NotRequired[int]        # Optional key

# With total=False, all keys are optional by default
class UpdateData(TypedDict, total=False):
    name: str
    email: str
    role: Required[str]          # Required even when total=False

def update_user(user_id: str, data: UpdateData) -> None: ...
```

### NewType for Domain Primitives

```python
from typing import NewType

UserId = NewType("UserId", str)
OrderId = NewType("OrderId", str)

def get_user(user_id: UserId) -> User: ...
def get_order(order_id: OrderId) -> Order: ...

uid = UserId("u-123")
oid = OrderId("o-456")

get_user(uid)   # OK
get_user(oid)   # Type error — OrderId is not UserId
```

## Mypy Configuration

```toml
# pyproject.toml
[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_any_generics = true
check_untyped_defs = true
no_implicit_reexport = true

# Per-module overrides for third-party libs without stubs
[[tool.mypy.overrides]]
module = ["some_untyped_lib.*"]
ignore_missing_imports = true
```

**Rules**:
- Always use `strict = true` for new projects
- Fix type errors instead of adding `# type: ignore`
- If `# type: ignore` is truly needed, add error code: `# type: ignore[assignment]`
- Run mypy in CI — don't merge code with type errors
- Use `reveal_type(expr)` during development to inspect inferred types
