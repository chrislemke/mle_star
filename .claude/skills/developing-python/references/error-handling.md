# Error Handling — Python Reference

Exception hierarchy design, error handling patterns, and logging practices.

## Contents

- [Exception Hierarchy](#exception-hierarchy)
- [Handling Patterns](#handling-patterns)
- [Context Managers for Cleanup](#context-managers-for-cleanup)
- [Logging Best Practices](#logging-best-practices)
- [Validation Patterns](#validation-patterns)

## Exception Hierarchy

Design a domain exception hierarchy rooted in a single base class:

```python
class AppError(Exception):
    """Base for all application-specific errors."""

class ValidationError(AppError):
    """Invalid input data."""
    def __init__(self, field: str, message: str) -> None:
        self.field = field
        super().__init__(f"Validation failed for '{field}': {message}")

class NotFoundError(AppError):
    """Requested entity does not exist."""
    def __init__(self, entity: str, id: str) -> None:
        self.entity = entity
        self.id = id
        super().__init__(f"{entity} not found: {id}")

class ConflictError(AppError):
    """Operation conflicts with current state."""

class AuthenticationError(AppError):
    """Identity verification failed."""

class AuthorizationError(AppError):
    """Insufficient permissions."""

class ExternalServiceError(AppError):
    """Failure in an external dependency."""
    def __init__(self, service: str, cause: Exception) -> None:
        self.service = service
        super().__init__(f"{service} failed: {cause}")
```

**Rules**:
- One base exception per application/library
- Exception names describe the problem, not the solution
- Include relevant context as attributes (not just in the message)
- Map external exceptions to domain exceptions at the boundary

## Handling Patterns

### Catch Specific, Handle Appropriately

```python
# Good: catch specific, handle each case
try:
    user = repo.get(user_id)
    user.update(data)
    repo.save(user)
except NotFoundError:
    raise  # Propagate — caller decides how to handle
except ValidationError as exc:
    logger.warning("Invalid update data for %s: %s", user_id, exc)
    raise
except DatabaseError as exc:
    logger.exception("DB failure updating user %s", user_id)
    raise ExternalServiceError("database", exc) from exc

# Bad: catch-all that swallows errors
try:
    do_something()
except Exception:
    pass  # Silent failure — bugs become invisible
```

### Exception Chaining

Always chain when converting exception types:

```python
try:
    response = httpx.get(url)
    response.raise_for_status()
except httpx.HTTPStatusError as exc:
    raise ExternalServiceError("payment-api", exc) from exc
#                                                    ^^^^^^^^
# Preserves the original traceback for debugging
```

### EAFP vs LBYL

```python
# EAFP (Easier to Ask Forgiveness than Permission) — Pythonic default
try:
    value = mapping[key]
except KeyError:
    value = default

# LBYL (Look Before You Leap) — when check is cheap and failure is expensive
if path.exists():
    data = path.read_text()
# But beware: LBYL has race conditions for file/network operations
# Prefer EAFP for I/O — the file could vanish between check and read
```

### Retry with Backoff

```python
import time
from typing import TypeVar

T = TypeVar("T")

def retry[T](
    func: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> T:
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except exceptions as exc:
            if attempt == max_attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1))  # Exponential backoff
            logger.warning(
                "Attempt %d/%d failed: %s. Retrying in %.1fs",
                attempt, max_attempts, exc, delay,
            )
            time.sleep(delay)
    raise RuntimeError("Unreachable")  # Satisfies type checker
```

## Context Managers for Cleanup

```python
from contextlib import contextmanager, suppress
from typing import Generator

# Ensure cleanup regardless of exceptions
@contextmanager
def managed_connection(url: str) -> Generator[Connection]:
    conn = Connection(url)
    conn.open()
    try:
        yield conn
    finally:
        conn.close()  # Always runs, even on exception

# Suppress specific expected exceptions
with suppress(FileNotFoundError):
    Path("optional-config.toml").unlink()

# Stack multiple context managers
from contextlib import ExitStack

def process_files(paths: list[Path]) -> list[str]:
    with ExitStack() as stack:
        files = [stack.enter_context(p.open()) for p in paths]
        return [f.read() for f in files]
```

## Logging Best Practices

```python
import logging

# Module-level logger
logger = logging.getLogger(__name__)

# Use lazy formatting — don't format strings that won't be logged
logger.debug("Processing user %s with %d items", user_id, len(items))

# Use exception() for caught exceptions — includes traceback
try:
    process(data)
except ProcessingError:
    logger.exception("Failed to process data batch")
    raise

# Structured logging with extra fields
logger.info(
    "Order completed",
    extra={"order_id": order.id, "total": order.total, "items": len(order.items)},
)
```

**Rules**:
- One logger per module: `logger = logging.getLogger(__name__)`
- Use `%s` formatting, not f-strings (lazy evaluation)
- `logger.exception()` inside except blocks (auto-includes traceback)
- Never log secrets, tokens, passwords, or PII
- Use structured logging (extra dict) for machine-parseable fields
- Log levels: DEBUG (development), INFO (normal operations), WARNING (unexpected but handled), ERROR (failure requiring attention)

## Validation Patterns

### Pydantic at Boundaries

```python
from pydantic import BaseModel, Field, field_validator

class CreateOrderRequest(BaseModel):
    model_config = {"extra": "forbid"}  # Reject unknown fields

    product_id: str = Field(min_length=1, max_length=50)
    quantity: int = Field(ge=1, le=1000)
    note: str | None = None

    @field_validator("product_id")
    @classmethod
    def validate_product_id_format(cls, v: str) -> str:
        if not v.startswith("prod-"):
            raise ValueError("Product ID must start with 'prod-'")
        return v
```

### Guard Clauses

```python
def process_payment(order: Order, payment: Payment) -> Receipt:
    # Validate preconditions at the top — fail fast
    if order.status != OrderStatus.PENDING:
        raise ConflictError(f"Order {order.id} is not pending: {order.status}")
    if payment.amount != order.total:
        raise ValidationError("amount", f"Expected {order.total}, got {payment.amount}")
    if payment.is_expired():
        raise ValidationError("payment", "Payment method has expired")

    # Happy path — no deep nesting
    charge = gateway.charge(payment, order.total)
    order.mark_paid(charge.id)
    return Receipt(order=order, charge=charge)
```
