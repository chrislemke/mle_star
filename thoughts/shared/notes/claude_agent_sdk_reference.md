# Claude Agent SDK -- Python: Complete API Reference

> Comprehensive extraction from official documentation and source code. Last updated: 2026-02-20.
> Package: `claude-agent-sdk` v0.1.39, bundled CLI v2.1.49, Python >=3.10, License: MIT.

---

## Table of Contents

1. [Overview & Architecture](#1-overview--architecture)
2. [Installation & Getting Started](#2-installation--getting-started)
3. [Core API: `query()` vs `ClaudeSDKClient`](#3-core-api-query-vs-claudesdkclient)
4. [Agent Configuration (`ClaudeAgentOptions`)](#4-agent-configuration-claudeagentoptions)
5. [Tool Definitions & Custom Tools](#5-tool-definitions--custom-tools)
6. [Multi-Agent Patterns & Orchestration (Subagents)](#6-multi-agent-patterns--orchestration-subagents)
7. [Message Types & Streaming](#7-message-types--streaming)
8. [Hooks System](#8-hooks-system)
9. [Permissions & User Input](#9-permissions--user-input)
10. [Session Management](#10-session-management)
11. [MCP (Model Context Protocol) Integration](#11-mcp-integration)
12. [Structured Outputs](#12-structured-outputs)
13. [File Checkpointing](#13-file-checkpointing)
14. [Sandbox Configuration](#14-sandbox-configuration)
15. [Plugins](#15-plugins)
16. [Hosting & Deployment](#16-hosting--deployment)
17. [Error Handling](#17-error-handling)
18. [Built-in Tool Schemas](#18-built-in-tool-schemas)
19. [Best Practices & Patterns](#19-best-practices--patterns)

---

## 1. Overview & Architecture

The Claude Agent SDK (formerly "Claude Code SDK") lets you build AI agents that autonomously read files, run commands, search the web, edit code, and more. It exposes the same tools, agent loop, and context management that power Claude Code, programmable in Python and TypeScript.

**Key architectural difference from the Anthropic Client SDK**: With the Client SDK, you implement a tool loop yourself. With the Agent SDK, Claude handles tool execution autonomously:

```python
# Client SDK: You implement the tool loop
response = client.messages.create(...)
while response.stop_reason == "tool_use":
    result = your_tool_executor(response.tool_use)
    response = client.messages.create(tool_result=result, **params)

# Agent SDK: Claude handles tools autonomously
async for message in query(prompt="Fix the bug in auth.py"):
    print(message)
```

**Architecture:** The SDK wraps the Claude Code CLI via subprocess (stdin/stdout JSON lines). Internal streaming mode is always used (even for `query()`). SDK MCP servers run in-process using the `mcp` library's `Server` class. No settings are loaded by default (isolated by design).

**Authentication**: Set `ANTHROPIC_API_KEY` environment variable. Also supports:
- Amazon Bedrock: `CLAUDE_CODE_USE_BEDROCK=1`
- Google Vertex AI: `CLAUDE_CODE_USE_VERTEX=1`
- Microsoft Azure: `CLAUDE_CODE_USE_FOUNDRY=1`

**Built-in Tools**:

| Tool | What it does |
|------|--------------|
| Read | Read any file in the working directory |
| Write | Create new files |
| Edit | Make precise edits to existing files |
| Bash | Run terminal commands, scripts, git operations |
| Glob | Find files by pattern (`**/*.ts`, `src/**/*.py`) |
| Grep | Search file contents with regex |
| WebSearch | Search the web for current information |
| WebFetch | Fetch and parse web page content |
| AskUserQuestion | Ask the user clarifying questions |
| NotebookEdit | Modify Jupyter notebook cells |
| TodoWrite | Track task progress |
| BashOutput | Read background process output |
| KillBash | Kill background processes |
| Task | Invoke subagents |
| ListMcpResources / ReadMcpResource | Access MCP resources |
| ExitPlanMode | Exit plan mode with a plan |

**Claude Code features available via SDK** (require `setting_sources=["project"]`):

| Feature | Description | Location |
|---------|-------------|----------|
| Skills | Specialized capabilities in Markdown | `.claude/skills/SKILL.md` |
| Slash commands | Custom commands | `.claude/commands/*.md` |
| Memory | Project context/instructions | `CLAUDE.md` or `.claude/CLAUDE.md` |
| Plugins | Custom commands, agents, MCP servers | Programmatic via `plugins` option |

### Source File Layout

| File | Purpose | Size |
|------|---------|------|
| `__init__.py` | Public API, `@tool` decorator, `create_sdk_mcp_server()` | 13.8KB |
| `types.py` | All type definitions (ClaudeAgentOptions, messages, hooks, etc.) | 24.6KB |
| `client.py` | `ClaudeSDKClient` class | 17.3KB |
| `query.py` | `query()` function | 4.5KB |
| `_errors.py` | Error hierarchy | 1.6KB |
| `_internal/query.py` | Core `Query` class (control protocol, hooks, MCP routing) | 25.7KB |
| `_internal/transport/subprocess_cli.py` | CLI subprocess transport | 25.1KB |

---

## 2. Installation & Getting Started

```bash
pip install claude-agent-sdk
```

**Requirements**: Python 3.10+ (supports 3.10, 3.11, 3.12, 3.13). Node.js is bundled with the package (no separate CLI install needed).

**Dependencies**: `anyio>=4.0.0`, `mcp>=0.1.0`, `typing_extensions>=4.0.0` (for py<3.11)

**Quickstart Example**:

```python
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, ResultMessage


async def main():
    async for message in query(
        prompt="Review utils.py for bugs. Fix any issues you find.",
        options=ClaudeAgentOptions(
            allowed_tools=["Read", "Edit", "Glob"],
            permission_mode="acceptEdits",
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    print(block.text)
                elif hasattr(block, "name"):
                    print(f"Tool: {block.name}")
        elif isinstance(message, ResultMessage):
            print(f"Done: {message.subtype}")


asyncio.run(main())
```

---

## 3. Core API: `query()` vs `ClaudeSDKClient`

### Quick Comparison

| Feature | `query()` | `ClaudeSDKClient` |
|:--------|:----------|:-------------------|
| Session | Creates new session each time | Reuses same session |
| Conversation | Single exchange | Multiple exchanges in same context |
| Connection | Managed automatically | Manual control |
| Streaming Input | Supported | Supported |
| Interrupts | Not supported | Supported |
| Hooks | Not supported | Supported |
| Custom Tools | Not supported | Supported |
| Continue Chat | New session each time | Maintains conversation |

### `query()` Function

```python
async def query(
    *,
    prompt: str | AsyncIterable[dict[str, Any]],
    options: ClaudeAgentOptions | None = None,
    transport: Transport | None = None,
) -> AsyncIterator[Message]
```

**Use for**: One-off questions, independent tasks, simple automation scripts, fresh-start scenarios.

```python
options = ClaudeAgentOptions(
    system_prompt="You are an expert Python developer",
    permission_mode="acceptEdits",
    cwd="/home/user/project",
)
async for message in query(prompt="Create a Python web server", options=options):
    print(message)
```

### `ClaudeSDKClient` Class

```python
class ClaudeSDKClient:
    def __init__(self, options: ClaudeAgentOptions | None = None, transport: Transport | None = None)
    async def connect(self, prompt: str | AsyncIterable[dict] | None = None) -> None
    async def query(self, prompt: str | AsyncIterable[dict], session_id: str = "default") -> None
    async def receive_messages(self) -> AsyncIterator[Message]
    async def receive_response(self) -> AsyncIterator[Message]
    async def interrupt(self) -> None
    async def set_permission_mode(self, mode: PermissionMode) -> None
    async def set_model(self, model: str) -> None
    async def rewind_files(self, user_message_uuid: str) -> None
    async def get_mcp_status(self) -> dict
    async def get_server_info(self) -> dict
    async def disconnect(self) -> None
```

**Use for**: Continuing conversations, follow-up questions, interactive applications, session control, custom tools & hooks.

**Context Manager Support**:
```python
async with ClaudeSDKClient() as client:
    await client.query("Hello Claude")
    async for message in client.receive_response():
        print(message)
```

**Continuing a conversation**:
```python
async with ClaudeSDKClient() as client:
    await client.query("What's the capital of France?")
    async for message in client.receive_response():
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(f"Claude: {block.text}")

    await client.query("What's the population of that city?")
    async for message in client.receive_response():
        # Claude remembers the previous context
        ...
```

**Using interrupts**:
```python
async with ClaudeSDKClient(options=options) as client:
    await client.query("Count from 1 to 100 slowly")
    await asyncio.sleep(2)
    await client.interrupt()
    await client.query("Just say hello instead")
    async for message in client.receive_response():
        pass
```

**Important**: Avoid using `break` to exit message iteration early (asyncio cleanup issues). Cannot be used across different async runtime contexts.

---

## 4. Agent Configuration (`ClaudeAgentOptions`)

```python
@dataclass
class ClaudeAgentOptions:
    # Tool Configuration
    tools: list[str] | ToolsPreset | None = None
    allowed_tools: list[str] = field(default_factory=list)
    disallowed_tools: list[str] = field(default_factory=list)

    # System Prompt
    system_prompt: str | SystemPromptPreset | None = None

    # MCP Servers
    mcp_servers: dict[str, McpServerConfig] | str | Path = field(default_factory=dict)

    # Permission Control
    permission_mode: PermissionMode | None = None  # "default"|"acceptEdits"|"plan"|"bypassPermissions"
    can_use_tool: CanUseTool | None = None
    permission_prompt_tool_name: str | None = None

    # Conversation Control
    continue_conversation: bool = False
    resume: str | None = None
    max_turns: int | None = None
    max_budget_usd: float | None = None
    fork_session: bool = False

    # Model Configuration
    model: str | None = None
    fallback_model: str | None = None

    # Beta Features
    betas: list[SdkBeta] = field(default_factory=list)  # ["context-1m-2025-08-07"]

    # Working Directory
    cwd: str | Path | None = None

    # CLI Configuration
    cli_path: str | Path | None = None
    settings: str | None = None
    add_dirs: list[str | Path] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    extra_args: dict[str, str | None] = field(default_factory=dict)
    max_buffer_size: int | None = None

    # Output Callbacks
    stderr: Callable[[str], None] | None = None

    # Hook Configurations
    hooks: dict[HookEvent, list[HookMatcher]] | None = None

    # User Identification
    user: str | None = None

    # Streaming
    include_partial_messages: bool = False

    # Agent Definitions
    agents: dict[str, AgentDefinition] | None = None

    # Setting Sources
    setting_sources: list[SettingSource] | None = None  # ["user", "project", "local"]

    # Sandbox Configuration
    sandbox: SandboxSettings | None = None

    # Plugin Configurations
    plugins: list[SdkPluginConfig] = field(default_factory=list)

    # Thinking Configuration
    thinking: ThinkingConfig | None = None
    effort: Literal["low", "medium", "high", "max"] | None = None
    max_thinking_tokens: int | None = None  # @deprecated

    # Structured Outputs
    output_format: dict[str, Any] | None = None

    # File Checkpointing
    enable_file_checkpointing: bool = False
```

**Key properties explained**:

- **`system_prompt`**: Pass a string for custom, or use preset: `{"type": "preset", "preset": "claude_code", "append": "extra instructions"}`
- **`permission_mode`**: `"default"`, `"acceptEdits"`, `"plan"`, or `"bypassPermissions"`
- **`setting_sources`**: Controls which filesystem settings to load. Default is `None` (no filesystem settings). Use `["project"]` to load CLAUDE.md files. Options: `"user"`, `"project"`, `"local"`
- **`output_format`**: `{"type": "json_schema", "schema": {...}}` for structured outputs
- **`betas`**: Currently supports `"context-1m-2025-08-07"`
- **`agents`**: Dict of `AgentDefinition` for programmatic subagents
- **`can_use_tool`**: Async callback for custom permission handling

**Settings Precedence** (highest to lowest):
1. Local settings (`.claude/settings.local.json`)
2. Project settings (`.claude/settings.json`)
3. User settings (`~/.claude/settings.json`)
4. Programmatic options always override filesystem settings

---

## 5. Tool Definitions & Custom Tools

### The `@tool` Decorator

```python
def tool(
    name: str,
    description: str,
    input_schema: type | dict[str, Any],
    annotations: ToolAnnotations | None = None,
) -> Callable[[Callable[[Any], Awaitable[dict[str, Any]]]], SdkMcpTool[Any]]
```

**Simple type mapping** (recommended):
```python
@tool("greet", "Greet a user", {"name": str})
async def greet(args: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": f"Hello, {args['name']}!"}]}
```

**Type mapping rules**:
- `str` -> `{"type": "string"}`
- `int` -> `{"type": "integer"}`
- `float` -> `{"type": "number"}`
- `bool` -> `{"type": "boolean"}`

**JSON Schema format** (for complex validation):
```python
@tool("advanced_process", "Process data with validation", {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
    },
    "required": ["name"],
})
async def advanced_process(args: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": f"Processed {args['name']}"}]}
```

### Creating an SDK MCP Server

```python
def create_sdk_mcp_server(
    name: str,
    version: str = "1.0.0",
    tools: list[SdkMcpTool[Any]] | None = None
) -> McpSdkServerConfig
```

**Complete example with multiple tools**:
```python
@tool("add", "Add two numbers", {"a": float, "b": float})
async def add(args):
    return {"content": [{"type": "text", "text": f"Sum: {args['a'] + args['b']}"}]}

@tool("multiply", "Multiply two numbers", {"a": float, "b": float})
async def multiply(args):
    return {"content": [{"type": "text", "text": f"Product: {args['a'] * args['b']}"}]}

calculator = create_sdk_mcp_server(
    name="calculator", version="2.0.0", tools=[add, multiply]
)

options = ClaudeAgentOptions(
    mcp_servers={"calc": calculator},
    allowed_tools=["mcp__calc__add", "mcp__calc__multiply"],
)
```

**Important**: Custom MCP tools require streaming input mode. Use an async generator for the `prompt` parameter:
```python
async def message_generator():
    yield {
        "type": "user",
        "message": {"role": "user", "content": "What's 5 + 3?"},
    }

async for message in query(prompt=message_generator(), options=options):
    ...
```

**Tool naming convention**: `mcp__<server_name>__<tool_name>` (e.g., `mcp__calculator__add`)

**Tool return format**: Always return `{"content": [{"type": "text", "text": "..."}]}`. For errors, add `"is_error": True`.

**Mixed Server Support** (SDK + External):
```python
options = ClaudeAgentOptions(
    mcp_servers={
        "internal": sdk_server,          # In-process
        "external": {                    # External subprocess
            "type": "stdio",
            "command": "external-server"
        }
    }
)
```

---

## 6. Multi-Agent Patterns & Orchestration (Subagents)

### AgentDefinition

```python
@dataclass
class AgentDefinition:
    description: str          # When to use this agent
    prompt: str               # System prompt
    tools: list[str] | None = None   # Allowed tools (inherits all if omitted)
    model: Literal["sonnet", "opus", "haiku", "inherit"] | None = None
```

### Creating Subagents

```python
async for message in query(
    prompt="Review the authentication module for security issues",
    options=ClaudeAgentOptions(
        allowed_tools=["Read", "Grep", "Glob", "Task"],  # Task tool is REQUIRED
        agents={
            "code-reviewer": AgentDefinition(
                description="Expert code review specialist.",
                prompt="Analyze code quality and suggest improvements.",
                tools=["Read", "Glob", "Grep"],   # Read-only
                model="sonnet",
            ),
            "test-runner": AgentDefinition(
                description="Runs and analyzes test suites.",
                prompt="Run tests and provide clear analysis.",
                tools=["Bash", "Read", "Grep"],
            ),
        },
    ),
):
    if hasattr(message, "result"):
        print(message.result)
```

**Key rules**:
- `Task` must be in `allowed_tools` for subagent invocation
- Subagents cannot spawn their own subagents (don't include `Task` in a subagent's tools)
- Claude automatically decides when to invoke subagents based on `description`
- Explicit invocation: `"Use the code-reviewer agent to check the auth module"`
- Messages from subagent context include `parent_tool_use_id`

**Common tool combinations for subagents**:

| Use case | Tools |
|:---------|:------|
| Read-only analysis | `Read`, `Grep`, `Glob` |
| Test execution | `Bash`, `Read`, `Grep` |
| Code modification | `Read`, `Edit`, `Write`, `Grep`, `Glob` |
| Full access | All tools (omit `tools` field) |

**Built-in general-purpose subagent**: Even without custom agents, Claude can spawn a built-in `general-purpose` subagent when `Task` is in `allowedTools`.

**Filesystem Agents**: Agents can also be loaded from `.claude/agents/*.md` files when `setting_sources=["project"]` is set.

---

## 7. Message Types & Streaming

### Message Union Type

```python
Message = UserMessage | AssistantMessage | SystemMessage | ResultMessage | StreamEvent
```

**UserMessage**:
```python
@dataclass
class UserMessage:
    content: str | list[ContentBlock]
    uuid: str | None = None
    parent_tool_use_id: str | None = None
    tool_use_result: dict[str, Any] | None = None
```

**AssistantMessage**:
```python
@dataclass
class AssistantMessage:
    content: list[ContentBlock]
    model: str
    parent_tool_use_id: str | None = None
    error: AssistantMessageError | None = None
```

**SystemMessage**:
```python
@dataclass
class SystemMessage:
    subtype: str
    data: dict[str, Any]
```

**ResultMessage**:
```python
@dataclass
class ResultMessage:
    subtype: str             # "success", "error", "error_max_structured_output_retries"
    duration_ms: int
    duration_api_ms: int
    is_error: bool
    num_turns: int
    session_id: str
    total_cost_usd: float | None = None
    usage: dict[str, Any] | None = None
    result: str | None = None
    structured_output: Any = None
```

**StreamEvent** (only when `include_partial_messages=True`):
```python
@dataclass
class StreamEvent:
    uuid: str
    session_id: str
    event: dict[str, Any]
    parent_tool_use_id: str | None = None
```

### Content Block Types

```python
ContentBlock = TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock

@dataclass
class TextBlock:
    text: str

@dataclass
class ThinkingBlock:
    thinking: str
    signature: str

@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]

@dataclass
class ToolResultBlock:
    tool_use_id: str
    content: str | list[dict[str, Any]] | None = None
    is_error: bool | None = None
```

### Streaming vs. Single Message Input

**Streaming Input (recommended)**: Persistent interactive session with full capabilities.
```python
async def message_generator():
    yield {"type": "user", "message": {"role": "user", "content": "Analyze this codebase"}}
    await asyncio.sleep(2)
    yield {"type": "user", "message": {"role": "user", "content": [
        {"type": "text", "text": "Review this diagram"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_data}},
    ]}}

async with ClaudeSDKClient(options) as client:
    await client.query(message_generator())
    async for message in client.receive_response():
        ...
```

Supports: image uploads, queued messages, tool integration, hooks, real-time feedback, context persistence.

**Single Message Input**: One-shot queries for stateless environments. Does NOT support: image attachments, dynamic message queueing, real-time interruption, hook integration.

---

## 8. Hooks System

Hooks intercept agent execution at key points for validation, logging, security, and custom logic.

### Available Hook Events

| Hook Event | Description | Available in Python SDK |
|------------|-------------|------------------------|
| `PreToolUse` | Before tool execution (can block/modify) | Yes |
| `PostToolUse` | After tool execution | Yes |
| `PostToolUseFailure` | After a tool fails | Yes |
| `UserPromptSubmit` | When user submits a prompt | Yes |
| `Stop` | When agent execution stops | Yes |
| `SubagentStop` | When a subagent completes | Yes |
| `SubagentStart` | When a subagent starts | Yes |
| `PreCompact` | Before conversation compaction | Yes |
| `Notification` | Notification events | Yes |
| `PermissionRequest` | Permission request events | Yes |

### Hook Configuration

```python
@dataclass
class HookMatcher:
    matcher: str | None = None    # Tool name pattern (e.g., "Bash", "Write|Edit")
    hooks: list[HookCallback] = field(default_factory=list)
    timeout: float | None = None  # Seconds (default: 60)
```

**Hook callback signature**:
```python
HookCallback = Callable[
    [HookInput, str | None, HookContext],  # input, tool_use_id, context
    Awaitable[HookJSONOutput]
]
```

### Hook Output Fields

**Top-level**: `continue_` (bool), `stopReason` (str), `suppressOutput` (bool), `systemMessage` (str), `decision` ("block"), `reason` (str)

**Inside `hookSpecificOutput`**:
- `hookEventName`: Required, use `input_data["hook_event_name"]`
- `permissionDecision`: `"allow"`, `"deny"`, or `"ask"`
- `permissionDecisionReason`: Explanation string
- `updatedInput`: Modified tool input (requires `permissionDecision: "allow"`)
- `additionalContext`: Context added to conversation

**Permission decision flow**: Deny (first) -> Ask (second) -> Allow (third) -> Default to Ask

**Important Python notes**: Use `continue_` (with underscore) instead of `continue`; use `async_` instead of `async`.

### Example: Security Hook

```python
async def validate_bash_command(input_data, tool_use_id, context):
    if input_data["tool_name"] == "Bash":
        command = input_data["tool_input"].get("command", "")
        if "rm -rf /" in command:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "Dangerous command blocked",
                }
            }
    return {}

options = ClaudeAgentOptions(
    hooks={
        "PreToolUse": [
            HookMatcher(matcher="Bash", hooks=[validate_bash_command], timeout=120),
            HookMatcher(hooks=[log_tool_use]),  # All tools
        ],
    }
)
```

### Hook Chaining

```python
hooks={
    "PreToolUse": [
        HookMatcher(hooks=[rate_limiter]),
        HookMatcher(hooks=[authorization_check]),
        HookMatcher(hooks=[input_sanitizer]),
        HookMatcher(hooks=[audit_logger]),
    ]
}
```

---

## 9. Permissions & User Input

### Permission Evaluation Order

1. **Hooks** (can allow, deny, or continue)
2. **Permission rules** (deny first, then allow, then ask)
3. **Permission mode** (global setting)
4. **`can_use_tool` callback** (runtime decision)

### Permission Modes

| Mode | Description |
|:-----|:------------|
| `default` | No auto-approvals; unmatched tools trigger `canUseTool` |
| `acceptEdits` | Auto-approve file edits and filesystem operations |
| `bypassPermissions` | All tools run without prompts (use with caution) |
| `plan` | No tool execution; Claude plans without making changes |

**Warning**: `bypassPermissions` is inherited by all subagents and cannot be overridden.

### `can_use_tool` Callback

```python
CanUseTool = Callable[
    [str, dict[str, Any], ToolPermissionContext],
    Awaitable[PermissionResult]
]

@dataclass
class ToolPermissionContext:
    signal: Any | None = None
    suggestions: list[PermissionUpdate] = field(default_factory=list)

@dataclass
class PermissionResultAllow:
    behavior: Literal["allow"] = "allow"
    updated_input: dict[str, Any] | None = None
    updated_permissions: list[PermissionUpdate] | None = None

@dataclass
class PermissionResultDeny:
    behavior: Literal["deny"] = "deny"
    message: str = ""
    interrupt: bool = False
```

### Handling Clarifying Questions (AskUserQuestion)

```python
async def can_use_tool(tool_name, input_data, context):
    if tool_name == "AskUserQuestion":
        answers = {}
        for q in input_data.get("questions", []):
            # Display q["question"], q["options"] to user
            # Collect answer
            answers[q["question"]] = selected_label
        return PermissionResultAllow(
            updated_input={"questions": input_data["questions"], "answers": answers}
        )
    return PermissionResultAllow(updated_input=input_data)
```

**Python workaround**: `can_use_tool` requires streaming mode and a `PreToolUse` hook that returns `{"continue_": True}` to keep the stream open.

---

## 10. Session Management

### Getting the Session ID

```python
session_id = None
async for message in query(prompt="Help me build a web application", options=options):
    if hasattr(message, "subtype") and message.subtype == "init":
        session_id = message.data.get("session_id")
```

### Resuming Sessions

```python
async for message in query(
    prompt="Continue where we left off",
    options=ClaudeAgentOptions(resume=session_id),
):
    print(message)
```

### Forking Sessions

```python
async for message in query(
    prompt="Try a different approach",
    options=ClaudeAgentOptions(
        resume=session_id,
        fork_session=True,  # Creates a new session ID, preserves original
    ),
):
    ...
```

| Behavior | `fork_session=False` | `fork_session=True` |
|----------|---------------------|---------------------|
| Session ID | Same as original | New ID generated |
| History | Appends to original | New branch from resume point |
| Original Session | Modified | Preserved unchanged |

---

## 11. MCP Integration

### Transport Types

**stdio** (local processes):
```python
options = ClaudeAgentOptions(
    mcp_servers={
        "github": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {"GITHUB_TOKEN": os.environ["GITHUB_TOKEN"]},
        }
    },
    allowed_tools=["mcp__github__list_issues"],
)
```

**HTTP/SSE** (remote servers):
```python
options = ClaudeAgentOptions(
    mcp_servers={
        "remote-api": {
            "type": "sse",
            "url": "https://api.example.com/mcp/sse",
            "headers": {"Authorization": f"Bearer {os.environ['API_TOKEN']}"},
        }
    },
)
```

**SDK** (in-process, via `create_sdk_mcp_server()`): See section 5.

### MCP Server Config Types

```python
# Stdio (external process)
class McpStdioServerConfig(TypedDict):
    type: NotRequired[Literal["stdio"]]
    command: str
    args: NotRequired[list[str]]
    env: NotRequired[dict[str, str]]

# SSE
class McpSSEServerConfig(TypedDict):
    type: Literal["sse"]
    url: str
    headers: NotRequired[dict[str, str]]

# HTTP
class McpHttpServerConfig(TypedDict):
    type: Literal["http"]
    url: str
    headers: NotRequired[dict[str, str]]

# SDK (in-process)
class McpSdkServerConfig(TypedDict):
    type: Literal["sdk"]
    name: str
    instance: McpServer
```

### MCP Tool Search

Dynamically loads tools on-demand when tool definitions exceed context limits. Controlled by `ENABLE_TOOL_SEARCH` env var:
- `auto` (default): activates at >10% of context
- `auto:5`: activates at 5%
- `true`: always enabled
- `false`: disabled

Requires Sonnet 4+ or Opus 4+ models.

### `.mcp.json` Config File

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_TOKEN": "${GITHUB_TOKEN}" }
    }
  }
}
```

---

## 12. Structured Outputs

Define exact JSON shape for agent output using JSON Schema or Pydantic.

### With Pydantic

```python
from pydantic import BaseModel
from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

class Step(BaseModel):
    step_number: int
    description: str
    estimated_complexity: str

class FeaturePlan(BaseModel):
    feature_name: str
    summary: str
    steps: list[Step]
    risks: list[str]

async for message in query(
    prompt="Plan how to add dark mode support.",
    options=ClaudeAgentOptions(
        output_format={
            "type": "json_schema",
            "schema": FeaturePlan.model_json_schema(),
        }
    ),
):
    if isinstance(message, ResultMessage) and message.structured_output:
        plan = FeaturePlan.model_validate(message.structured_output)
        print(f"Feature: {plan.feature_name}")
```

### With Raw JSON Schema

```python
schema = {
    "type": "object",
    "properties": {
        "company_name": {"type": "string"},
        "founded_year": {"type": "number"},
    },
    "required": ["company_name"],
}
options = ClaudeAgentOptions(output_format={"type": "json_schema", "schema": schema})
```

**Error subtypes**: `"success"` or `"error_max_structured_output_retries"`.

---

## 13. File Checkpointing

Track file modifications through Write, Edit, and NotebookEdit tools and restore to any previous state.

**Enable**:
```python
options = ClaudeAgentOptions(
    enable_file_checkpointing=True,
    permission_mode="acceptEdits",
    extra_args={"replay-user-messages": None},
    env={**os.environ, "CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING": "1"},
)
```

**Capture checkpoint UUID**:
```python
async for message in client.receive_response():
    if isinstance(message, UserMessage) and message.uuid:
        checkpoint_id = message.uuid
    if isinstance(message, ResultMessage):
        session_id = message.session_id
```

**Rewind**:
```python
async with ClaudeSDKClient(
    ClaudeAgentOptions(enable_file_checkpointing=True, resume=session_id)
) as client:
    await client.query("")
    async for message in client.receive_response():
        await client.rewind_files(checkpoint_id)
        break
```

**Limitations**: Only tracks Write/Edit/NotebookEdit (not Bash changes); same-session only; file content only (not directories); local files only.

---

## 14. Sandbox Configuration

```python
class SandboxSettings(TypedDict, total=False):
    enabled: bool
    autoAllowBashIfSandboxed: bool
    excludedCommands: list[str]
    allowUnsandboxedCommands: bool
    network: SandboxNetworkConfig
    ignoreViolations: SandboxIgnoreViolations
    enableWeakerNestedSandbox: bool

sandbox_settings = {
    "enabled": True,
    "autoAllowBashIfSandboxed": True,
    "network": {"allowLocalBinding": True},
}
options = ClaudeAgentOptions(sandbox=sandbox_settings)
```

**`SandboxNetworkConfig`**: `allowLocalBinding`, `allowUnixSockets`, `allowAllUnixSockets`, `httpProxyPort`, `socksProxyPort`.

---

## 15. Plugins

Load custom plugins from local paths to add commands, agents, skills, hooks, and MCP servers.

```python
options = ClaudeAgentOptions(
    plugins=[
        {"type": "local", "path": "./my-plugin"},
        {"type": "local", "path": "/absolute/path/to/plugin"},
    ],
)
```

**Plugin directory structure**:
```
my-plugin/
  .claude-plugin/plugin.json    # Required manifest
  commands/custom-cmd.md
  agents/specialist.md
  skills/my-skill/SKILL.md
  hooks/hooks.json
  .mcp.json
```

Commands are namespaced: `plugin-name:command-name`.

---

## 16. Hosting & Deployment

### System Requirements

- Python 3.10+ / Node.js 18+
- Claude Code CLI (bundled with SDK)
- 1 GiB RAM, 5 GiB disk, 1 CPU (minimum recommended)
- Outbound HTTPS to `api.anthropic.com`

### Deployment Patterns

| Pattern | Description | Examples |
|---------|-------------|----------|
| Ephemeral Sessions | New container per task, destroyed on completion | Bug fixes, invoice processing |
| Long-Running Sessions | Persistent containers, multiple agent processes | Email agents, site builders |
| Hybrid Sessions | Ephemeral containers hydrated with history/state | Project managers, deep research |
| Single Containers | Multiple agent processes in one container | Simulations |

### Sandbox Providers

Modal Sandbox, Cloudflare Sandboxes, Daytona, E2B, Fly Machines, Vercel Sandbox.

**Cost**: Dominant cost is tokens; container cost ~5 cents/hour minimum. Agent sessions do not timeout (use `max_turns` to prevent loops).

---

## 17. Error Handling

```python
class ClaudeSDKError(Exception): ...           # Base error
class CLIConnectionError(ClaudeSDKError): ...  # Connection issues
class CLINotFoundError(CLIConnectionError):    # CLI not installed
    def __init__(self, message="Claude Code not found", cli_path=None): ...
class ProcessError(ClaudeSDKError):            # Process failed
    def __init__(self, message, exit_code=None, stderr=None): ...
class CLIJSONDecodeError(ClaudeSDKError):      # JSON parsing issues
    def __init__(self, line, original_error): ...
class MessageParseError(ClaudeSDKError):       # Message parsing issues
    def __init__(self, data): ...
```

**Usage**:
```python
from claude_agent_sdk import query, CLINotFoundError, ProcessError, CLIJSONDecodeError

try:
    async for message in query(prompt="Hello"):
        print(message)
except CLINotFoundError:
    print("Please install Claude Code: npm install -g @anthropic-ai/claude-code")
except ProcessError as e:
    print(f"Process failed with exit code: {e.exit_code}")
except CLIJSONDecodeError as e:
    print(f"Failed to parse response: {e}")
```

---

## 18. Built-in Tool Schemas

**Bash**: `{"command": str, "timeout": int|None, "description": str|None, "run_in_background": bool|None}` -> `{"output": str, "exitCode": int, "killed": bool|None}`

**Read**: `{"file_path": str, "offset": int|None, "limit": int|None}` -> `{"content": str, "total_lines": int}`

**Write**: `{"file_path": str, "content": str}` -> `{"message": str, "bytes_written": int}`

**Edit**: `{"file_path": str, "old_string": str, "new_string": str, "replace_all": bool|None}` -> `{"message": str, "replacements": int}`

**Glob**: `{"pattern": str, "path": str|None}` -> `{"matches": list[str], "count": int}`

**Grep**: `{"pattern": str, "path": str|None, "glob": str|None, ...}` -> `{"matches": [...], "total_matches": int}`

**Task**: `{"description": str, "prompt": str, "subagent_type": str}` -> `{"result": str, "usage": dict|None, "total_cost_usd": float|None}`

**AskUserQuestion**: `{"questions": [...], "answers": dict|None}` (questions have `question`, `header`, `options`, `multiSelect` fields)

---

## 19. Best Practices & Patterns

### Tool Configuration Patterns

| Tools | Agent Capability |
|-------|-----------------|
| `Read`, `Glob`, `Grep` | Read-only analysis |
| `Read`, `Edit`, `Glob` | Analyze and modify code |
| `Read`, `Edit`, `Bash`, `Glob`, `Grep` | Full automation |

### Permission Mode Selection

| Mode | Behavior | Use Case |
|------|----------|----------|
| `acceptEdits` | Auto-approves file edits | Trusted development |
| `bypassPermissions` | No prompts | CI/CD, automation |
| `default` | Requires `canUseTool` callback | Custom approval flows |
| `plan` | No execution | Code review, proposal |

### Continuous Conversation Pattern

```python
class ConversationSession:
    def __init__(self, options):
        self.client = ClaudeSDKClient(options)
        self.turn_count = 0

    async def start(self):
        await self.client.connect()
        while True:
            user_input = input(f"[Turn {self.turn_count + 1}] You: ")
            if user_input.lower() == "exit":
                break
            await self.client.query(user_input)
            self.turn_count += 1
            async for message in self.client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            print(block.text, end="")
            print()
        await self.client.disconnect()
```

---

## Sources

- [Agent SDK Reference - Python](https://platform.claude.com/docs/en/agent-sdk/python)
- [Agent SDK Overview](https://platform.claude.com/docs/en/agent-sdk/overview)
- [Agent SDK Quickstart](https://platform.claude.com/docs/en/agent-sdk/quickstart)
- [Hooks Guide](https://platform.claude.com/docs/en/agent-sdk/hooks)
- [Structured Outputs](https://platform.claude.com/docs/en/agent-sdk/structured-outputs)
- [Handle Approvals and User Input](https://platform.claude.com/docs/en/agent-sdk/user-input)
- [Configure Permissions](https://platform.claude.com/docs/en/agent-sdk/permissions)
- [Session Management](https://platform.claude.com/docs/en/agent-sdk/sessions)
- [Subagents](https://platform.claude.com/docs/en/agent-sdk/subagents)
- [MCP Integration](https://platform.claude.com/docs/en/agent-sdk/mcp)
- [Custom Tools](https://platform.claude.com/docs/en/agent-sdk/custom-tools)
- [Streaming vs Single Mode](https://platform.claude.com/docs/en/agent-sdk/streaming-vs-single-mode)
- [Hosting](https://platform.claude.com/docs/en/agent-sdk/hosting)
- [File Checkpointing](https://platform.claude.com/docs/en/agent-sdk/file-checkpointing)
- [Plugins](https://platform.claude.com/docs/en/agent-sdk/plugins)
- [GitHub - anthropics/claude-agent-sdk-python](https://github.com/anthropics/claude-agent-sdk-python)
- [PyPI - claude-agent-sdk](https://pypi.org/project/claude-agent-sdk/)
