# Claude Agent SDK -- Python: Code Examples

> All example files from the `anthropics/claude-agent-sdk-python` repository. Last updated: 2026-02-20.

---

## Table of Contents

1. [quick_start.py -- Basic Usage](#1-quick_startpy)
2. [streaming_mode.py -- ClaudeSDKClient Patterns](#2-streaming_modepy)
3. [agents.py -- Custom Agent Definitions](#3-agentspy)
4. [hooks.py -- Hook Patterns](#4-hookspy)
5. [mcp_calculator.py -- In-Process MCP Server](#5-mcp_calculatorpy)
6. [tool_permission_callback.py -- Permission Callbacks](#6-tool_permission_callbackpy)
7. [system_prompt.py -- System Prompt Configurations](#7-system_promptpy)
8. [tools_option.py -- Tool Configurations](#8-tools_optionpy)
9. [setting_sources.py -- Setting Source Control](#9-setting_sourcespy)
10. [filesystem_agents.py -- Agents from Markdown Files](#10-filesystem_agentspy)
11. [plugin_example.py -- Plugin Loading](#11-plugin_examplepy)
12. [include_partial_messages.py -- Streaming Partial Messages](#12-include_partial_messagespy)
13. [max_budget_usd.py -- Cost Control](#13-max_budget_usdpy)
14. [streaming_mode_ipython.py -- IPython Interactive Examples](#14-streaming_mode_ipythonpy)
15. [streaming_mode_trio.py -- Trio Async Runtime](#15-streaming_mode_triopy)
16. [Architecture Notes](#16-architecture-notes)
17. [Key Patterns Summary](#17-key-patterns-summary)

---

## 1. quick_start.py

Basic usage patterns demonstrating `query()` with various options.

```python
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, ResultMessage


async def basic_query():
    """Simplest possible query."""
    async for message in query(prompt="What is 2 + 2?"):
        print(message)


async def query_with_options():
    """Query with custom options."""
    options = ClaudeAgentOptions(
        system_prompt="You are an expert Python developer",
        permission_mode="acceptEdits",
        cwd="/home/user/project",
    )
    async for message in query(prompt="Create a Python web server", options=options):
        print(message)


async def query_with_tools():
    """Query with specific tool restrictions."""
    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Edit", "Glob"],
        permission_mode="acceptEdits",
    )
    async for message in query(
        prompt="Review utils.py for bugs that would cause crashes. Fix any issues.",
        options=options,
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    print(block.text)
                elif hasattr(block, "name"):
                    print(f"Tool: {block.name}")
        elif isinstance(message, ResultMessage):
            print(f"Done: {message.subtype}")


asyncio.run(basic_query())
```

---

## 2. streaming_mode.py

Comprehensive `ClaudeSDKClient` patterns (10 examples).

### Pattern 1: Basic Streaming
```python
async with ClaudeSDKClient() as client:
    await client.query("What is 2 + 2?")
    async for message in client.receive_response():
        print(message)
```

### Pattern 2: Multi-Turn Conversation
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

### Pattern 3: Concurrent Sessions
```python
async with ClaudeSDKClient() as client:
    await client.query("Analyze auth.py", session_id="analysis")
    await client.query("Write tests for auth.py", session_id="testing")

    async for message in client.receive_messages():
        if hasattr(message, "session_id"):
            print(f"[{message.session_id}]: {message}")
```

### Pattern 4: Interrupt
```python
async with ClaudeSDKClient() as client:
    await client.query("Count from 1 to 100 slowly")
    await asyncio.sleep(2)
    await client.interrupt()
    await client.query("Just say hello instead")
    async for message in client.receive_response():
        pass
```

### Pattern 5: Manual Connection Management
```python
client = ClaudeSDKClient(options=options)
await client.connect()
try:
    await client.query("Hello")
    async for message in client.receive_response():
        process(message)
finally:
    await client.disconnect()
```

### Pattern 6: With Options
```python
options = ClaudeAgentOptions(
    system_prompt="You are a code review expert",
    allowed_tools=["Read", "Grep", "Glob"],
    permission_mode="acceptEdits",
    max_turns=10,
)
async with ClaudeSDKClient(options=options) as client:
    await client.query("Review the codebase for security issues")
    async for message in client.receive_response():
        print(message)
```

### Pattern 7: Async Iterable Input
```python
async def message_generator():
    yield {
        "type": "user",
        "message": {"role": "user", "content": "Hello, how are you?"},
    }
    await asyncio.sleep(1)
    yield {
        "type": "user",
        "message": {"role": "user", "content": "Tell me a joke"},
    }

async with ClaudeSDKClient() as client:
    await client.query(message_generator())
    async for message in client.receive_response():
        print(message)
```

### Pattern 8: Bash Tool Usage
```python
options = ClaudeAgentOptions(
    allowed_tools=["Bash"],
    permission_mode="bypassPermissions",
)
async with ClaudeSDKClient(options=options) as client:
    await client.query("List all Python files in the current directory")
    async for message in client.receive_response():
        print(message)
```

### Pattern 9: Control Protocol Messages
```python
async with ClaudeSDKClient() as client:
    await client.query("Hello")
    async for message in client.receive_messages():
        if isinstance(message, SystemMessage):
            print(f"System: {message.subtype} -> {message.data}")
        elif isinstance(message, AssistantMessage):
            print(f"Assistant: {message.content}")
        elif isinstance(message, ResultMessage):
            print(f"Result: {message.subtype}, turns={message.num_turns}")
            break
```

### Pattern 10: Error Handling
```python
from claude_agent_sdk import CLINotFoundError, ProcessError

try:
    async with ClaudeSDKClient() as client:
        await client.query("Hello")
        async for message in client.receive_response():
            print(message)
except CLINotFoundError:
    print("Claude Code CLI not found")
except ProcessError as e:
    print(f"Process error: {e.exit_code}")
```

---

## 3. agents.py

Custom agent definitions using `AgentDefinition`.

### Single Agent
```python
options = ClaudeAgentOptions(
    allowed_tools=["Read", "Grep", "Glob", "Task"],
    agents={
        "code-reviewer": AgentDefinition(
            description="Reviews code for best practices and potential issues",
            prompt="You are a code reviewer. Analyze code for bugs, performance, security.",
            tools=["Read", "Grep"],
            model="sonnet",
        ),
    },
)

async for message in query(
    prompt="Use the code-reviewer agent to review the code",
    options=options,
):
    print(message)
```

### Multiple Agents
```python
options = ClaudeAgentOptions(
    allowed_tools=["Read", "Grep", "Glob", "Task", "Write", "Edit", "Bash"],
    agents={
        "code-reviewer": AgentDefinition(
            description="Reviews code for best practices",
            prompt="You are a code reviewer.",
            tools=["Read", "Grep", "Glob"],
            model="sonnet",
        ),
        "doc-writer": AgentDefinition(
            description="Writes comprehensive documentation",
            prompt="You are a technical documentation expert.",
            tools=["Read", "Write", "Edit"],
            model="sonnet",
        ),
        "test-runner": AgentDefinition(
            description="Runs and analyzes test suites",
            prompt="Run tests and provide clear analysis.",
            tools=["Bash", "Read", "Grep"],
        ),
    },
)
```

### Dynamic Agent Configuration
```python
def create_security_agent(security_level: str) -> AgentDefinition:
    is_strict = security_level == "strict"
    return AgentDefinition(
        description="Security code reviewer",
        prompt=f"You are a {'strict' if is_strict else 'balanced'} security reviewer...",
        tools=["Read", "Grep", "Glob"],
        model="opus" if is_strict else "sonnet",
    )
```

---

## 4. hooks.py

5 hook patterns demonstrating the hooks system.

### PreToolUse: Block Dangerous Commands
```python
async def check_bash_command(input_data, tool_use_id, context):
    tool_name = input_data["tool_name"]
    tool_input = input_data["tool_input"]
    if tool_name != "Bash":
        return {}
    command = tool_input.get("command", "")
    block_patterns = ["rm -rf", "foo.sh"]
    for pattern in block_patterns:
        if pattern in command:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": f"Blocked: {pattern}",
                }
            }
    return {}

options = ClaudeAgentOptions(
    hooks={
        "PreToolUse": [
            HookMatcher(matcher="Bash", hooks=[check_bash_command]),
        ],
    }
)
```

### UserPromptSubmit: Validate Input
```python
async def validate_prompt(input_data, tool_use_id, context):
    prompt = input_data.get("prompt", "")
    if len(prompt) > 10000:
        return {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "permissionDecision": "deny",
                "permissionDecisionReason": "Prompt too long",
            }
        }
    return {}
```

### PostToolUse: Log Results
```python
async def log_tool_result(input_data, tool_use_id, context):
    tool_name = input_data.get("tool_name", "unknown")
    result = input_data.get("tool_result", {})
    print(f"[LOG] {tool_name} completed: {str(result)[:100]}")
    return {}
```

### Decision Fields
```python
async def modify_input(input_data, tool_use_id, context):
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "updatedInput": {**input_data["tool_input"], "timeout": 30000},
        }
    }
```

### Continue Control
```python
async def stop_on_error(input_data, tool_use_id, context):
    if input_data.get("tool_result", {}).get("is_error"):
        return {
            "continue_": False,
            "stopReason": "Tool execution failed",
        }
    return {}
```

---

## 5. mcp_calculator.py

Complete in-process MCP server with 6 mathematical tools.

```python
from claude_agent_sdk import (
    ClaudeAgentOptions, ClaudeSDKClient,
    create_sdk_mcp_server, tool,
    AssistantMessage, ResultMessage, TextBlock, ToolUseBlock,
)
import math

@tool("add", "Add two numbers", {"a": float, "b": float})
async def add(args):
    result = args["a"] + args["b"]
    return {"content": [{"type": "text", "text": f"{args['a']} + {args['b']} = {result}"}]}

@tool("subtract", "Subtract b from a", {"a": float, "b": float})
async def subtract(args):
    result = args["a"] - args["b"]
    return {"content": [{"type": "text", "text": f"{args['a']} - {args['b']} = {result}"}]}

@tool("multiply", "Multiply two numbers", {"a": float, "b": float})
async def multiply(args):
    result = args["a"] * args["b"]
    return {"content": [{"type": "text", "text": f"{args['a']} * {args['b']} = {result}"}]}

@tool("divide", "Divide a by b", {"a": float, "b": float})
async def divide(args):
    if args["b"] == 0:
        return {"content": [{"type": "text", "text": "Error: Division by zero"}], "is_error": True}
    result = args["a"] / args["b"]
    return {"content": [{"type": "text", "text": f"{args['a']} / {args['b']} = {result}"}]}

@tool("power", "Raise a to the power of b", {"base": float, "exponent": float})
async def power(args):
    result = math.pow(args["base"], args["exponent"])
    return {"content": [{"type": "text", "text": f"{args['base']}^{args['exponent']} = {result}"}]}

@tool("sqrt", "Square root of a number", {"number": float})
async def sqrt(args):
    if args["number"] < 0:
        return {"content": [{"type": "text", "text": "Error: Cannot take sqrt of negative"}], "is_error": True}
    result = math.sqrt(args["number"])
    return {"content": [{"type": "text", "text": f"sqrt({args['number']}) = {result}"}]}

calculator = create_sdk_mcp_server(
    name="calculator", version="2.0.0",
    tools=[add, subtract, multiply, divide, power, sqrt],
)

options = ClaudeAgentOptions(
    mcp_servers={"calc": calculator},
    allowed_tools=[
        "mcp__calc__add", "mcp__calc__subtract",
        "mcp__calc__multiply", "mcp__calc__divide",
        "mcp__calc__power", "mcp__calc__sqrt",
    ],
    permission_mode="bypassPermissions",
)

async def main():
    async with ClaudeSDKClient(options=options) as client:
        await client.query("What is (15 + 27) * 3?")
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text)
                    elif isinstance(block, ToolUseBlock):
                        print(f"Calling: {block.name}({block.input})")
            elif isinstance(message, ResultMessage):
                print(f"Done in {message.duration_ms}ms")

asyncio.run(main())
```

---

## 6. tool_permission_callback.py

Custom `can_use_tool` callback with allow/deny/redirect logic.

```python
from claude_agent_sdk import (
    ClaudeAgentOptions, ClaudeSDKClient,
    PermissionResultAllow, PermissionResultDeny,
    ToolPermissionContext, HookMatcher,
)

async def my_permission_callback(
    tool_name: str,
    input_data: dict,
    context: ToolPermissionContext,
) -> PermissionResultAllow | PermissionResultDeny:
    # Auto-allow read-only tools
    if tool_name in ["Read", "Glob", "Grep"]:
        return PermissionResultAllow()

    # Block dangerous bash commands
    if tool_name == "Bash":
        command = input_data.get("command", "")
        if "rm -rf" in command:
            return PermissionResultDeny(message="Dangerous command blocked")
        return PermissionResultAllow()

    # Redirect write to specific directory
    if tool_name == "Write":
        file_path = input_data.get("file_path", "")
        if not file_path.startswith("/allowed/path/"):
            return PermissionResultDeny(message=f"Cannot write to {file_path}")
        return PermissionResultAllow()

    # Block unknown tools
    return PermissionResultDeny(message=f"Unknown tool: {tool_name}")

# IMPORTANT: can_use_tool requires a PreToolUse hook that returns continue_=True
async def passthrough_hook(input_data, tool_use_id, context):
    return {"continue_": True}

options = ClaudeAgentOptions(
    can_use_tool=my_permission_callback,
    permission_mode="default",
    hooks={
        "PreToolUse": [HookMatcher(hooks=[passthrough_hook])],
    },
)
```

---

## 7. system_prompt.py

Three system prompt configuration methods.

```python
# Method 1: String prompt
options_string = ClaudeAgentOptions(
    system_prompt="You are a helpful coding assistant specializing in Python."
)

# Method 2: Preset prompt (uses Claude Code's default system prompt)
options_preset = ClaudeAgentOptions(
    system_prompt={"type": "preset", "preset": "claude_code"}
)

# Method 3: Preset with additional instructions
options_preset_append = ClaudeAgentOptions(
    system_prompt={
        "type": "preset",
        "preset": "claude_code",
        "append": "Always explain your reasoning step by step. Focus on security."
    }
)
```

---

## 8. tools_option.py

Tool configuration options.

```python
# Array of specific tools
options_array = ClaudeAgentOptions(
    tools=["Read", "Write", "Edit", "Bash"]
)

# Empty array (no tools)
options_empty = ClaudeAgentOptions(
    tools=[]
)

# Preset: all tools
options_preset = ClaudeAgentOptions(
    tools={"type": "preset", "preset": "all"}
)

# allowed_tools (additive to tools)
options_allowed = ClaudeAgentOptions(
    allowed_tools=["Read", "Grep", "Glob"]
)

# disallowed_tools (subtractive)
options_disallowed = ClaudeAgentOptions(
    disallowed_tools=["Bash", "Write"]
)
```

---

## 9. setting_sources.py

Controlling which filesystem settings are loaded.

```python
# No settings loaded (default, isolated)
options_none = ClaudeAgentOptions()

# User settings only (~/.claude/)
options_user = ClaudeAgentOptions(
    setting_sources=["user"]
)

# Project settings (loads CLAUDE.md, .claude/settings.json)
options_project = ClaudeAgentOptions(
    setting_sources=["project"]
)

# All settings
options_all = ClaudeAgentOptions(
    setting_sources=["user", "project", "local"]
)
```

---

## 10. filesystem_agents.py

Loading agents from `.claude/agents/*.md` files.

```python
# Agents defined in .claude/agents/test-agent.md are automatically
# available when setting_sources includes "project"
options = ClaudeAgentOptions(
    setting_sources=["project"],
    allowed_tools=["Task", "Read", "Grep"],
)

# The agent markdown file format:
# ---
# description: "Description of what this agent does"
# model: sonnet
# tools: ["Read", "Grep"]
# ---
# You are a specialized agent that...
```

---

## 11. plugin_example.py

Loading local plugins.

```python
options = ClaudeAgentOptions(
    plugins=[
        {"type": "local", "path": "./examples/plugins/demo-plugin"},
    ],
    setting_sources=["project"],
)

# Plugin structure:
# examples/plugins/demo-plugin/
#   .claude-plugin/plugin.json  (manifest)
#   commands/greet.md           (custom command)
```

---

## 12. include_partial_messages.py

Streaming partial messages for real-time display.

```python
from claude_agent_sdk import ClaudeAgentOptions, StreamEvent

options = ClaudeAgentOptions(
    include_partial_messages=True,
)

async with ClaudeSDKClient(options=options) as client:
    await client.query("Write a short poem about coding")
    async for message in client.receive_response():
        if isinstance(message, StreamEvent):
            event = message.event
            # StreamEvents contain partial content as it's generated
            print(f"Stream: {event}")
        elif isinstance(message, AssistantMessage):
            # Full message when complete
            print(f"Complete: {message.content}")
```

---

## 13. max_budget_usd.py

Cost control with budget limits.

```python
options = ClaudeAgentOptions(
    max_budget_usd=0.50,  # Stop after $0.50 spent
    max_turns=20,         # Also limit turns
)

async for message in query(prompt="Analyze the entire codebase", options=options):
    if isinstance(message, ResultMessage):
        print(f"Cost: ${message.total_cost_usd:.4f}")
        print(f"Turns: {message.num_turns}")
```

---

## 14. streaming_mode_ipython.py

IPython/Jupyter interactive examples.

```python
# In IPython/Jupyter, use nest_asyncio for nested event loops
import nest_asyncio
nest_asyncio.apply()

import asyncio
from claude_agent_sdk import ClaudeSDKClient

async def chat():
    async with ClaudeSDKClient() as client:
        await client.query("Hello from Jupyter!")
        async for message in client.receive_response():
            print(message)

asyncio.run(chat())
```

---

## 15. streaming_mode_trio.py

Trio async runtime support (alternative to asyncio).

```python
import trio
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

async def main():
    options = ClaudeAgentOptions(
        permission_mode="bypassPermissions",
    )
    async with ClaudeSDKClient(options=options) as client:
        await client.query("Hello from Trio!")
        async for message in client.receive_response():
            print(message)

trio.run(main)
```

---

## 16. Architecture Notes

### Subprocess-Based CLI Wrapping

The SDK does not directly call the Anthropic API. Instead, it:

1. **Spawns a subprocess** running the bundled Claude Code CLI
2. **Communicates via stdin/stdout** using JSON lines protocol
3. **Routes MCP calls** for SDK MCP servers back to the Python process
4. **Handles hooks** as callbacks during the control protocol handshake

```
Your Python Code
    |
    v
claude_agent_sdk (Python)
    |
    v  (subprocess, JSON lines over stdin/stdout)
Claude Code CLI (Node.js)
    |
    v  (HTTPS)
Anthropic API
```

### Internal Class Hierarchy

```
query() function
  -> InternalClient.process_query()
    -> Query class (control protocol)
      -> SubprocessCLITransport
        -> Claude Code CLI subprocess

ClaudeSDKClient
  -> Query class (control protocol)
    -> SubprocessCLITransport
      -> Claude Code CLI subprocess
```

### Message Flow

```
CLI stdout -> JSON parsing -> message_parser.py -> typed Message objects -> user code
```

### Key Design Decisions

1. **No settings loaded by default** (v0.1.0 breaking change) -- SDK runs in isolated mode
2. **Always streaming internally** -- Even `query()` uses streaming mode under the hood
3. **SDK MCP servers run in-process** -- No subprocess overhead for custom tools
4. **`anyio` for async runtime compatibility** -- Supports both `asyncio` and `trio`

---

## 17. Key Patterns Summary

| Pattern | API | Key Options |
|---------|-----|-------------|
| One-shot query | `query()` | `prompt`, basic options |
| Multi-turn chat | `ClaudeSDKClient` | `connect()`, `query()`, `receive_response()` |
| Custom tools | `@tool` + `create_sdk_mcp_server()` | `mcp_servers`, `allowed_tools` |
| Security hooks | `HookMatcher` | `hooks={"PreToolUse": [...]}` |
| Subagents | `AgentDefinition` | `agents={...}`, `allowed_tools=["Task"]` |
| Permission control | `can_use_tool` callback | `permission_mode="default"` |
| Structured output | Pydantic/JSON Schema | `output_format={"type": "json_schema", ...}` |
| Session resume | `resume` option | `ClaudeAgentOptions(resume=session_id)` |
| Cost control | `max_budget_usd` | `max_budget_usd=0.50` |
| File rollback | File checkpointing | `enable_file_checkpointing=True` |
