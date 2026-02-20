# Security Checklist — Python Reference

Security practices for Python applications, aligned with OWASP Top 10 (2025).

## Contents

- [Input Validation](#input-validation)
- [Injection Prevention](#injection-prevention)
- [Authentication & Secrets](#authentication--secrets)
- [Dependency Security](#dependency-security)
- [Serialization Safety](#serialization-safety)
- [Error Handling](#error-handling-security)
- [File Operations](#file-operations)

## Input Validation

Validate **all** data crossing trust boundaries (user input, API responses, file content, environment variables).

```python
# Use Pydantic for structured validation at boundaries
from pydantic import BaseModel, Field, EmailStr

class CreateUserRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    email: EmailStr
    age: int = Field(ge=0, le=150)

# Validate early, fail fast
def create_user(raw_data: dict) -> User:
    request = CreateUserRequest.model_validate(raw_data)  # Raises on invalid
    return User(name=request.name, email=request.email)
```

**Rules**:
- Allowlist over denylist — define what IS valid, not what isn't
- Validate type, length, range, and format
- Reject unexpected fields (Pydantic `model_config = {"extra": "forbid"}`)
- Normalize before validation (strip whitespace, lowercase emails)

## Injection Prevention

### SQL Injection

```python
# NEVER do this
cursor.execute(f"SELECT * FROM users WHERE id = '{user_id}'")  # SQL injection!

# ALWAYS parameterize
cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))

# With SQLAlchemy
session.execute(select(User).where(User.id == user_id))
```

### Command Injection

```python
# NEVER do this
import os
os.system(f"convert {filename} output.png")  # Shell injection!

# Use subprocess with list arguments (no shell)
import subprocess
subprocess.run(["convert", filename, "output.png"], check=True)

# If shell is truly needed, use shlex.quote
import shlex
subprocess.run(f"convert {shlex.quote(filename)} output.png", shell=True, check=True)
```

### Template Injection

```python
# Jinja2: enable autoescaping
from jinja2 import Environment, select_autoescape
env = Environment(autoescape=select_autoescape(["html", "xml"]))
```

## Authentication & Secrets

### Password Hashing

```python
# Use bcrypt or argon2 — never MD5, SHA-1, or plain SHA-256
import bcrypt

def hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def verify_password(password: str, hashed: bytes) -> bool:
    return bcrypt.checkpw(password.encode(), hashed)
```

### Secrets and Tokens

```python
# Use secrets module — never random
import secrets

token = secrets.token_urlsafe(32)      # URL-safe random token
api_key = secrets.token_hex(32)         # Hex random token
otp = secrets.randbelow(1_000_000)      # Random integer

# Constant-time comparison to prevent timing attacks
secrets.compare_digest(received_token, stored_token)
```

### Environment Variables

```python
import os

# Read secrets from environment, not source code
DATABASE_URL = os.environ["DATABASE_URL"]       # Raises if missing
DEBUG = os.environ.get("DEBUG", "false") == "true"  # Safe default

# Never log or print secrets
logging.info("Connected to database")  # Good
logging.info(f"Connected to {DATABASE_URL}")  # BAD — leaks credentials
```

## Dependency Security

```bash
# Pin all dependencies (use uv, pip-compile, or poetry.lock)
uv lock

# Audit for known vulnerabilities
pip-audit
# or
safety check

# Update dependencies regularly
uv lock --upgrade
```

**Rules**:
- Pin exact versions in lock files
- Audit dependencies in CI/CD pipelines
- Review new dependencies before adding (check maintenance, popularity, license)
- Minimize dependency count — each dependency is an attack surface

## Serialization Safety

```python
# NEVER unpickle untrusted data
import pickle
data = pickle.loads(untrusted_bytes)  # Remote Code Execution risk!

# Use JSON for data interchange
import json
data = json.loads(untrusted_string)  # Safe — only produces basic types

# NEVER use yaml.load() with untrusted input
import yaml
data = yaml.safe_load(untrusted_string)  # safe_load, not load!

# NEVER use eval/exec on untrusted input
eval(user_input)    # Remote Code Execution!
exec(user_input)    # Remote Code Execution!

# For complex deserialization, use Pydantic
data = MyModel.model_validate_json(untrusted_bytes)
```

## Error Handling Security

```python
# Don't expose internals in error responses
# Bad
except DatabaseError as e:
    return {"error": str(e)}  # Leaks DB schema, query details

# Good
except DatabaseError:
    logger.exception("Database error in create_user")
    return {"error": "An internal error occurred"}

# Don't leak stack traces to users
# Configure error handlers to return generic messages in production
# Log full details server-side
```

**Rules**:
- Log full error details server-side
- Return generic error messages to clients
- Never include tracebacks in API responses
- Use different error detail levels for development vs production

## File Operations

```python
from pathlib import Path

# Prevent path traversal
def safe_read(base_dir: Path, filename: str) -> str:
    # Resolve to absolute path and verify it's within base_dir
    target = (base_dir / filename).resolve()
    if not target.is_relative_to(base_dir.resolve()):
        raise ValueError(f"Path traversal detected: {filename}")
    return target.read_text()

# Set restrictive file permissions for sensitive files
import os
import stat

path = Path("config/secrets.env")
path.write_text(content)
path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 600: owner read/write only

# Use tempfile for temporary files (auto-cleanup, secure creation)
import tempfile
with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=True) as f:
    f.write(data)
    f.flush()
    process(f.name)
```
