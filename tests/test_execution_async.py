"""Tests for async script execution in the execution module (Task 13).

Validates ``ExecutionRawResult`` and ``execute_script`` which runs a Python
script as an async subprocess, captures stdout/stderr/exit_code, measures
wall-clock duration, and enforces timeouts with SIGTERM/SIGKILL escalation
and orphan child process cleanup.

Tests are written TDD-first and serve as the executable specification for
REQ-EX-007 through REQ-EX-010, REQ-EX-037, and REQ-EX-043.

Refs:
    SRS 02a (Execution Environment), IMPLEMENTATION_PLAN.md Task 13.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import tempfile
import textwrap

from hypothesis import HealthCheck, given, settings, strategies as st
from mle_star.execution import ExecutionRawResult, execute_script
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_script(tmp_path: Path, content: str, name: str = "script.py") -> str:
    """Write a Python script to tmp_path and return its absolute path."""
    script_path = tmp_path / name
    script_path.write_text(textwrap.dedent(content), encoding="utf-8")
    return str(script_path.resolve())


# ===========================================================================
# REQ-EX-008: ExecutionRawResult -- Model Structure
# ===========================================================================


@pytest.mark.unit
class TestExecutionRawResultFields:
    """ExecutionRawResult has correct fields with correct types (REQ-EX-008)."""

    def test_has_stdout_field(self) -> None:
        """Model has a stdout field."""
        result = ExecutionRawResult(
            stdout="hello",
            stderr="",
            exit_code=0,
            duration_seconds=1.0,
            timed_out=False,
        )
        assert result.stdout == "hello"

    def test_has_stderr_field(self) -> None:
        """Model has a stderr field."""
        result = ExecutionRawResult(
            stdout="",
            stderr="error",
            exit_code=1,
            duration_seconds=1.0,
            timed_out=False,
        )
        assert result.stderr == "error"

    def test_has_exit_code_field(self) -> None:
        """Model has an exit_code field."""
        result = ExecutionRawResult(
            stdout="",
            stderr="",
            exit_code=42,
            duration_seconds=1.0,
            timed_out=False,
        )
        assert result.exit_code == 42

    def test_has_duration_seconds_field(self) -> None:
        """Model has a duration_seconds field."""
        result = ExecutionRawResult(
            stdout="",
            stderr="",
            exit_code=0,
            duration_seconds=3.14,
            timed_out=False,
        )
        assert result.duration_seconds == pytest.approx(3.14)

    def test_has_timed_out_field(self) -> None:
        """Model has a timed_out field."""
        result = ExecutionRawResult(
            stdout="",
            stderr="",
            exit_code=0,
            duration_seconds=1.0,
            timed_out=True,
        )
        assert result.timed_out is True

    def test_stdout_is_str(self) -> None:
        """Stdout field is of type str."""
        result = ExecutionRawResult(
            stdout="text",
            stderr="",
            exit_code=0,
            duration_seconds=0.1,
            timed_out=False,
        )
        assert isinstance(result.stdout, str)

    def test_stderr_is_str(self) -> None:
        """Stderr field is of type str."""
        result = ExecutionRawResult(
            stdout="",
            stderr="err",
            exit_code=0,
            duration_seconds=0.1,
            timed_out=False,
        )
        assert isinstance(result.stderr, str)

    def test_exit_code_is_int(self) -> None:
        """exit_code field is of type int."""
        result = ExecutionRawResult(
            stdout="",
            stderr="",
            exit_code=0,
            duration_seconds=0.1,
            timed_out=False,
        )
        assert isinstance(result.exit_code, int)

    def test_duration_seconds_is_float(self) -> None:
        """duration_seconds field is of type float."""
        result = ExecutionRawResult(
            stdout="",
            stderr="",
            exit_code=0,
            duration_seconds=1.5,
            timed_out=False,
        )
        assert isinstance(result.duration_seconds, float)

    def test_timed_out_is_bool(self) -> None:
        """timed_out field is of type bool."""
        result = ExecutionRawResult(
            stdout="",
            stderr="",
            exit_code=0,
            duration_seconds=0.1,
            timed_out=False,
        )
        assert isinstance(result.timed_out, bool)


# ===========================================================================
# REQ-EX-008: ExecutionRawResult -- Immutability
# ===========================================================================


@pytest.mark.unit
class TestExecutionRawResultFrozen:
    """ExecutionRawResult is frozen (immutable) following project conventions (REQ-EX-008)."""

    def test_cannot_modify_stdout(self) -> None:
        """Attempting to modify stdout raises an error."""
        result = ExecutionRawResult(
            stdout="original",
            stderr="",
            exit_code=0,
            duration_seconds=0.1,
            timed_out=False,
        )
        with pytest.raises(Exception):  # noqa: B017
            result.stdout = "modified"  # type: ignore[misc]

    def test_cannot_modify_stderr(self) -> None:
        """Attempting to modify stderr raises an error."""
        result = ExecutionRawResult(
            stdout="",
            stderr="original",
            exit_code=0,
            duration_seconds=0.1,
            timed_out=False,
        )
        with pytest.raises(Exception):  # noqa: B017
            result.stderr = "modified"  # type: ignore[misc]

    def test_cannot_modify_exit_code(self) -> None:
        """Attempting to modify exit_code raises an error."""
        result = ExecutionRawResult(
            stdout="",
            stderr="",
            exit_code=0,
            duration_seconds=0.1,
            timed_out=False,
        )
        with pytest.raises(Exception):  # noqa: B017
            result.exit_code = 99  # type: ignore[misc]

    def test_cannot_modify_duration_seconds(self) -> None:
        """Attempting to modify duration_seconds raises an error."""
        result = ExecutionRawResult(
            stdout="",
            stderr="",
            exit_code=0,
            duration_seconds=0.1,
            timed_out=False,
        )
        with pytest.raises(Exception):  # noqa: B017
            result.duration_seconds = 999.0  # type: ignore[misc]

    def test_cannot_modify_timed_out(self) -> None:
        """Attempting to modify timed_out raises an error."""
        result = ExecutionRawResult(
            stdout="",
            stderr="",
            exit_code=0,
            duration_seconds=0.1,
            timed_out=False,
        )
        with pytest.raises(Exception):  # noqa: B017
            result.timed_out = True  # type: ignore[misc]


# ===========================================================================
# REQ-EX-008: ExecutionRawResult -- All Fields Required
# ===========================================================================


@pytest.mark.unit
class TestExecutionRawResultRequired:
    """All fields of ExecutionRawResult are required (REQ-EX-008)."""

    def test_missing_stdout_raises(self) -> None:
        """Omitting stdout raises a validation error."""
        with pytest.raises(Exception):  # noqa: B017
            ExecutionRawResult(  # type: ignore[call-arg]
                stderr="",
                exit_code=0,
                duration_seconds=0.1,
                timed_out=False,
            )

    def test_missing_stderr_raises(self) -> None:
        """Omitting stderr raises a validation error."""
        with pytest.raises(Exception):  # noqa: B017
            ExecutionRawResult(  # type: ignore[call-arg]
                stdout="",
                exit_code=0,
                duration_seconds=0.1,
                timed_out=False,
            )

    def test_missing_exit_code_raises(self) -> None:
        """Omitting exit_code raises a validation error."""
        with pytest.raises(Exception):  # noqa: B017
            ExecutionRawResult(  # type: ignore[call-arg]
                stdout="",
                stderr="",
                duration_seconds=0.1,
                timed_out=False,
            )

    def test_missing_duration_seconds_raises(self) -> None:
        """Omitting duration_seconds raises a validation error."""
        with pytest.raises(Exception):  # noqa: B017
            ExecutionRawResult(  # type: ignore[call-arg]
                stdout="",
                stderr="",
                exit_code=0,
                timed_out=False,
            )

    def test_missing_timed_out_raises(self) -> None:
        """Omitting timed_out raises a validation error."""
        with pytest.raises(Exception):  # noqa: B017
            ExecutionRawResult(  # type: ignore[call-arg]
                stdout="",
                stderr="",
                exit_code=0,
                duration_seconds=0.1,
            )


# ===========================================================================
# REQ-EX-007: execute_script -- Happy Path
# ===========================================================================


@pytest.mark.unit
class TestExecuteScriptHappyPath:
    """execute_script runs a script and captures output correctly (REQ-EX-007)."""

    async def test_returns_execution_raw_result(self, tmp_path: Path) -> None:
        """Return value is an ExecutionRawResult instance."""
        script = _write_script(tmp_path, "print('hello')")
        result = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=10,
        )
        assert isinstance(result, ExecutionRawResult)

    async def test_captures_stdout(self, tmp_path: Path) -> None:
        """Captured stdout from the script appears in result.stdout."""
        script = _write_script(tmp_path, "print('hello world')")
        result = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=10,
        )
        assert "hello world" in result.stdout

    async def test_captures_stderr(self, tmp_path: Path) -> None:
        """Captured stderr from the script appears in result.stderr."""
        script = _write_script(
            tmp_path,
            """\
            import sys
            print('error output', file=sys.stderr)
            """,
        )
        result = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=10,
        )
        assert "error output" in result.stderr

    async def test_exit_code_zero_for_success(self, tmp_path: Path) -> None:
        """Successful script returns exit_code=0."""
        script = _write_script(tmp_path, "x = 1 + 1")
        result = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=10,
        )
        assert result.exit_code == 0

    async def test_duration_seconds_is_positive(self, tmp_path: Path) -> None:
        """duration_seconds is greater than zero for any script."""
        script = _write_script(tmp_path, "x = 1")
        result = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=10,
        )
        assert result.duration_seconds > 0.0

    async def test_timed_out_false_for_fast_script(self, tmp_path: Path) -> None:
        """timed_out is False for a script that completes within timeout."""
        script = _write_script(tmp_path, "print('fast')")
        result = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=10,
        )
        assert result.timed_out is False

    async def test_stdout_and_stderr_captured_separately(self, tmp_path: Path) -> None:
        """Streams stdout and stderr contain their respective output only."""
        script = _write_script(
            tmp_path,
            """\
            import sys
            print('out_marker')
            print('err_marker', file=sys.stderr)
            """,
        )
        result = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=10,
        )
        assert "out_marker" in result.stdout
        assert "err_marker" in result.stderr
        assert "err_marker" not in result.stdout
        assert "out_marker" not in result.stderr

    async def test_multiline_stdout_captured(self, tmp_path: Path) -> None:
        """Multi-line stdout is captured fully."""
        script = _write_script(
            tmp_path,
            """\
            for i in range(5):
                print(f'line_{i}')
            """,
        )
        result = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=10,
        )
        for i in range(5):
            assert f"line_{i}" in result.stdout


# ===========================================================================
# REQ-EX-007: execute_script -- Error Handling (Non-zero Exit Codes)
# ===========================================================================


@pytest.mark.unit
class TestExecuteScriptErrorHandling:
    """execute_script captures non-zero exit codes without raising (REQ-EX-007)."""

    async def test_nonzero_exit_code_not_raised(self, tmp_path: Path) -> None:
        """Non-zero exit code does NOT cause an exception."""
        script = _write_script(
            tmp_path,
            """\
            import sys
            sys.exit(1)
            """,
        )
        # Should not raise
        result = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=10,
        )
        assert result.exit_code != 0

    async def test_exit_code_captured_for_sys_exit(self, tmp_path: Path) -> None:
        """sys.exit(42) is captured as exit_code=42."""
        script = _write_script(
            tmp_path,
            """\
            import sys
            sys.exit(42)
            """,
        )
        result = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=10,
        )
        assert result.exit_code == 42

    async def test_syntax_error_returns_nonzero(self, tmp_path: Path) -> None:
        """Script with syntax error returns non-zero exit code."""
        script = _write_script(tmp_path, "def foo(:\n    pass")
        result = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=10,
        )
        assert result.exit_code != 0

    async def test_syntax_error_stderr_has_output(self, tmp_path: Path) -> None:
        """Script with syntax error produces stderr output."""
        script = _write_script(tmp_path, "def foo(:\n    pass")
        result = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=10,
        )
        assert len(result.stderr) > 0

    async def test_runtime_exception_returns_nonzero(self, tmp_path: Path) -> None:
        """Script that raises an exception returns non-zero exit code."""
        script = _write_script(
            tmp_path,
            """\
            raise ValueError('test error')
            """,
        )
        result = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=10,
        )
        assert result.exit_code != 0

    async def test_runtime_exception_stderr_has_traceback(self, tmp_path: Path) -> None:
        """Script that raises an exception has traceback in stderr."""
        script = _write_script(
            tmp_path,
            """\
            raise ValueError('specific_error_marker')
            """,
        )
        result = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=10,
        )
        assert "specific_error_marker" in result.stderr

    async def test_import_error_returns_nonzero(self, tmp_path: Path) -> None:
        """Script with missing import returns non-zero exit code."""
        script = _write_script(
            tmp_path,
            """\
            import nonexistent_module_xyz_abc_123
            """,
        )
        result = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=10,
        )
        assert result.exit_code != 0

    async def test_stderr_captured_alongside_stdout(self, tmp_path: Path) -> None:
        """Script that writes to both stdout and stderr before crashing captures both."""
        script = _write_script(
            tmp_path,
            """\
            import sys
            print('stdout_before_crash')
            print('stderr_before_crash', file=sys.stderr)
            raise RuntimeError('crash')
            """,
        )
        result = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=10,
        )
        assert "stdout_before_crash" in result.stdout
        assert "stderr_before_crash" in result.stderr


# ===========================================================================
# REQ-EX-009: execute_script -- Timeout Enforcement
# ===========================================================================


@pytest.mark.unit
class TestExecuteScriptTimeout:
    """execute_script enforces timeout with SIGTERM/SIGKILL escalation (REQ-EX-009)."""

    async def test_timed_out_is_true(self, tmp_path: Path) -> None:
        """Script exceeding timeout has timed_out=True."""
        script = _write_script(
            tmp_path,
            """\
            import time
            time.sleep(60)
            """,
        )
        result = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=1,
        )
        assert result.timed_out is True

    async def test_exit_code_is_negative_one_on_timeout(self, tmp_path: Path) -> None:
        """Timed-out result has exit_code=-1."""
        script = _write_script(
            tmp_path,
            """\
            import time
            time.sleep(60)
            """,
        )
        result = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=1,
        )
        assert result.exit_code == -1

    async def test_partial_stdout_preserved_on_timeout(self, tmp_path: Path) -> None:
        """Partial stdout written before timeout is preserved in result."""
        script = _write_script(
            tmp_path,
            """\
            import sys
            import time
            print('before_timeout_marker', flush=True)
            sys.stdout.flush()
            time.sleep(60)
            print('after_timeout_marker')
            """,
        )
        result = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=2,
        )
        assert result.timed_out is True
        assert "before_timeout_marker" in result.stdout

    async def test_partial_stderr_preserved_on_timeout(self, tmp_path: Path) -> None:
        """Partial stderr written before timeout is preserved in result."""
        script = _write_script(
            tmp_path,
            """\
            import sys
            import time
            print('stderr_before_timeout', file=sys.stderr, flush=True)
            sys.stderr.flush()
            time.sleep(60)
            """,
        )
        result = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=2,
        )
        assert result.timed_out is True
        assert "stderr_before_timeout" in result.stderr

    async def test_duration_approximately_equals_timeout(self, tmp_path: Path) -> None:
        """Duration is approximately the timeout value, not the script sleep time."""
        timeout = 2
        script = _write_script(
            tmp_path,
            """\
            import time
            time.sleep(60)
            """,
        )
        result = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=timeout,
        )
        assert result.timed_out is True
        # Duration should be close to timeout, with some slack for SIGTERM grace
        assert result.duration_seconds >= timeout - 0.5
        # Allow generous upper bound for SIGTERM+SIGKILL grace period
        assert result.duration_seconds < timeout + 10

    async def test_fast_script_does_not_timeout(self, tmp_path: Path) -> None:
        """A script that finishes within timeout is NOT timed out."""
        script = _write_script(tmp_path, "print('fast')")
        result = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=10,
        )
        assert result.timed_out is False
        assert result.exit_code == 0


# ===========================================================================
# REQ-EX-007: execute_script -- Environment Variables
# ===========================================================================


@pytest.mark.unit
class TestExecuteScriptEnvironment:
    """execute_script passes custom env to subprocess (REQ-EX-007)."""

    async def test_custom_env_passed_to_subprocess(self, tmp_path: Path) -> None:
        """Script can access environment variables passed via env parameter."""
        script = _write_script(
            tmp_path,
            """\
            import os
            print(os.environ.get('MLE_TEST_MARKER', 'NOT_FOUND'))
            """,
        )
        custom_env = dict(os.environ)
        custom_env["MLE_TEST_MARKER"] = "custom_value_42"
        result = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=10,
            env=custom_env,
        )
        assert "custom_value_42" in result.stdout

    async def test_env_none_inherits_current_environment(self, tmp_path: Path) -> None:
        """When env is None, subprocess inherits the current environment."""
        script = _write_script(
            tmp_path,
            """\
            import os
            # PATH should always be set in inherited environment
            print(os.environ.get('PATH', 'NO_PATH'))
            """,
        )
        result = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=10,
            env=None,
        )
        assert result.exit_code == 0
        assert "NO_PATH" not in result.stdout

    async def test_custom_env_isolates_variables(self, tmp_path: Path) -> None:
        """Custom env dict restricts visible variables to only those provided."""
        script = _write_script(
            tmp_path,
            """\
            import os
            # Check for a variable that only exists in the custom env
            val = os.environ.get('ISOLATED_VAR', 'MISSING')
            print(f'ISOLATED_VAR={val}')
            """,
        )
        # Build a minimal env with just what Python needs plus our marker
        minimal_env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "ISOLATED_VAR": "present_42",
        }
        # Add PYTHONPATH if present
        if "PYTHONPATH" in os.environ:
            minimal_env["PYTHONPATH"] = os.environ["PYTHONPATH"]
        result = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=10,
            env=minimal_env,
        )
        assert "ISOLATED_VAR=present_42" in result.stdout

    async def test_multiple_custom_env_vars(self, tmp_path: Path) -> None:
        """Multiple custom environment variables are all accessible."""
        script = _write_script(
            tmp_path,
            """\
            import os
            print(os.environ.get('VAR_A', ''))
            print(os.environ.get('VAR_B', ''))
            print(os.environ.get('VAR_C', ''))
            """,
        )
        custom_env = dict(os.environ)
        custom_env["VAR_A"] = "alpha"
        custom_env["VAR_B"] = "beta"
        custom_env["VAR_C"] = "gamma"
        result = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=10,
            env=custom_env,
        )
        assert "alpha" in result.stdout
        assert "beta" in result.stdout
        assert "gamma" in result.stdout


# ===========================================================================
# REQ-EX-007: execute_script -- Working Directory
# ===========================================================================


@pytest.mark.unit
class TestExecuteScriptWorkingDirectory:
    """execute_script sets cwd to working_dir (REQ-EX-007)."""

    async def test_cwd_is_working_dir(self, tmp_path: Path) -> None:
        """Script's working directory matches the provided working_dir."""
        script = _write_script(
            tmp_path,
            """\
            import os
            print(os.getcwd())
            """,
        )
        result = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=10,
        )
        assert str(tmp_path.resolve()) in result.stdout.strip()

    async def test_script_can_read_files_in_working_dir(self, tmp_path: Path) -> None:
        """Script can read files placed in the working directory."""
        data_file = tmp_path / "data.txt"
        data_file.write_text("secret_content_xyz", encoding="utf-8")
        script = _write_script(
            tmp_path,
            """\
            with open('data.txt', 'r') as f:
                print(f.read())
            """,
        )
        result = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=10,
        )
        assert "secret_content_xyz" in result.stdout

    async def test_script_can_write_files_in_working_dir(self, tmp_path: Path) -> None:
        """Script can write files to the working directory."""
        script = _write_script(
            tmp_path,
            """\
            with open('output.txt', 'w') as f:
                f.write('written_by_script')
            print('done')
            """,
        )
        result = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=10,
        )
        assert result.exit_code == 0
        output_file = tmp_path / "output.txt"
        assert output_file.exists()
        assert output_file.read_text(encoding="utf-8") == "written_by_script"

    async def test_different_working_dir_from_script_location(
        self, tmp_path: Path
    ) -> None:
        """Script can run with a working_dir different from where the script lives."""
        script_dir = tmp_path / "scripts"
        script_dir.mkdir()
        work_dir = tmp_path / "workspace"
        work_dir.mkdir()
        (work_dir / "input.txt").write_text("workspace_data", encoding="utf-8")

        script = _write_script(
            script_dir,
            """\
            import os
            print(os.getcwd())
            with open('input.txt', 'r') as f:
                print(f.read())
            """,
        )
        result = await execute_script(
            script_path=script,
            working_dir=str(work_dir),
            timeout_seconds=10,
        )
        assert result.exit_code == 0
        assert "workspace_data" in result.stdout


# ===========================================================================
# REQ-EX-037: execute_script -- Orphan Child Process Cleanup
# ===========================================================================


@pytest.mark.unit
class TestExecuteScriptOrphanCleanup:
    """execute_script cleans up orphan child processes via os.killpg (REQ-EX-037)."""

    async def test_child_processes_killed_on_timeout(self, tmp_path: Path) -> None:
        """After timeout, child processes spawned by the script are also terminated."""
        marker_file = tmp_path / "child_alive_marker.txt"
        # Write the child script separately to avoid dedent issues
        child_script = tmp_path / "child.py"
        child_script.write_text(
            "import time\n"
            "while True:\n"
            f"    with open({str(marker_file)!r}, 'w') as f:\n"
            "        f.write(str(time.time()))\n"
            "    time.sleep(0.1)\n",
            encoding="utf-8",
        )
        script = _write_script(
            tmp_path,
            """\
            import subprocess
            import sys
            import time

            child = subprocess.Popen([sys.executable, 'child.py'])
            print('parent_started', flush=True)
            time.sleep(60)
            """,
        )
        result = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=2,
        )
        assert result.timed_out is True

        # Wait briefly for cleanup to complete
        await asyncio.sleep(1.0)

        # Record the marker file content, then wait and check it stopped updating
        if marker_file.exists():
            content_before = marker_file.read_text(encoding="utf-8")
            await asyncio.sleep(1.0)
            content_after = marker_file.read_text(encoding="utf-8")
            assert content_before == content_after, (
                "Child process is still updating marker file after parent was killed"
            )


# ===========================================================================
# REQ-EX-010: execute_script -- Resource Isolation
# ===========================================================================


@pytest.mark.unit
class TestExecuteScriptResourceIsolation:
    """Each execute_script invocation runs in a new subprocess (REQ-EX-010)."""

    async def test_separate_invocations_have_separate_pids(
        self, tmp_path: Path
    ) -> None:
        """Two invocations produce different PIDs (different subprocesses)."""
        script = _write_script(
            tmp_path,
            """\
            import os
            print(os.getpid())
            """,
        )
        result1 = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=10,
        )
        result2 = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=10,
        )
        pid1 = int(result1.stdout.strip())
        pid2 = int(result2.stdout.strip())
        assert pid1 != pid2

    async def test_no_shared_state_between_invocations(self, tmp_path: Path) -> None:
        """Global variables set in one invocation are not visible in another."""
        script1 = _write_script(
            tmp_path,
            """\
            import builtins
            builtins._mle_test_shared = 'set_by_first'
            print('first_done')
            """,
            name="script1.py",
        )
        script2 = _write_script(
            tmp_path,
            """\
            import builtins
            val = getattr(builtins, '_mle_test_shared', 'NOT_FOUND')
            print(val)
            """,
            name="script2.py",
        )
        await execute_script(
            script_path=script1,
            working_dir=str(tmp_path),
            timeout_seconds=10,
        )
        result2 = await execute_script(
            script_path=script2,
            working_dir=str(tmp_path),
            timeout_seconds=10,
        )
        assert "NOT_FOUND" in result2.stdout


# ===========================================================================
# REQ-EX-043: execute_script -- UTF-8 Decoding with errors="replace"
# ===========================================================================


@pytest.mark.unit
class TestExecuteScriptUtf8Decoding:
    """stdout/stderr decoded as UTF-8 with errors='replace' (REQ-EX-043)."""

    async def test_utf8_stdout_decoded_correctly(self, tmp_path: Path) -> None:
        """Valid UTF-8 in stdout is decoded correctly."""
        script = _write_script(
            tmp_path,
            """\
            print('Caf\\u00e9 \\u00fc\\u00f6\\u00e4')
            """,
        )
        result = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=10,
        )
        assert "Caf\u00e9" in result.stdout

    async def test_invalid_bytes_replaced_not_error(self, tmp_path: Path) -> None:
        """Invalid UTF-8 bytes are replaced (not raise), producing replacement chars."""
        script = _write_script(
            tmp_path,
            """\
            import sys
            # Write raw invalid UTF-8 bytes to stdout
            sys.stdout.buffer.write(b'valid_prefix_')
            sys.stdout.buffer.write(b'\\xff\\xfe')
            sys.stdout.buffer.write(b'_valid_suffix\\n')
            sys.stdout.buffer.flush()
            """,
        )
        result = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=10,
        )
        # Should not raise and should contain the valid parts
        assert "valid_prefix_" in result.stdout
        assert "_valid_suffix" in result.stdout
        # Invalid bytes should be replaced with replacement character
        assert "\ufffd" in result.stdout

    async def test_utf8_stderr_decoded_correctly(self, tmp_path: Path) -> None:
        """Valid UTF-8 in stderr is decoded correctly."""
        script = _write_script(
            tmp_path,
            """\
            import sys
            print('Error: \\u00e4\\u00f6\\u00fc\\u00df', file=sys.stderr)
            """,
        )
        result = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=10,
        )
        assert "\u00e4\u00f6\u00fc\u00df" in result.stderr


# ===========================================================================
# REQ-EX-007: execute_script -- Is Async
# ===========================================================================


@pytest.mark.unit
class TestExecuteScriptIsAsync:
    """execute_script is a coroutine function (REQ-EX-007)."""

    def test_execute_script_is_coroutine_function(self) -> None:
        """execute_script is an async function (returns a coroutine)."""
        assert asyncio.iscoroutinefunction(execute_script)


# ===========================================================================
# REQ-EX-008: ExecutionRawResult -- Property-Based Tests
# ===========================================================================


@pytest.mark.unit
class TestExecutionRawResultPropertyBased:
    """Property-based tests for ExecutionRawResult using Hypothesis."""

    @given(
        stdout=st.text(max_size=200),
        stderr=st.text(max_size=200),
        exit_code=st.integers(min_value=-128, max_value=255),
        duration_seconds=st.floats(min_value=0.0, max_value=1e6, allow_nan=False),
        timed_out=st.booleans(),
    )
    @settings(max_examples=50)
    def test_all_fields_roundtrip(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
        duration_seconds: float,
        timed_out: bool,
    ) -> None:
        """Property: all field values stored and retrieved correctly."""
        result = ExecutionRawResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_seconds=duration_seconds,
            timed_out=timed_out,
        )
        assert result.stdout == stdout
        assert result.stderr == stderr
        assert result.exit_code == exit_code
        assert result.duration_seconds == duration_seconds
        assert result.timed_out == timed_out

    @given(
        stdout=st.text(max_size=100),
        stderr=st.text(max_size=100),
        exit_code=st.integers(min_value=-128, max_value=255),
        duration_seconds=st.floats(min_value=0.0, max_value=1e6, allow_nan=False),
        timed_out=st.booleans(),
    )
    @settings(max_examples=30)
    def test_is_always_frozen(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
        duration_seconds: float,
        timed_out: bool,
    ) -> None:
        """Property: ExecutionRawResult is always frozen for any field values."""
        result = ExecutionRawResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_seconds=duration_seconds,
            timed_out=timed_out,
        )
        with pytest.raises(Exception):  # noqa: B017
            result.stdout = "mutated"  # type: ignore[misc]


# ===========================================================================
# REQ-EX-007: execute_script -- Property-Based Tests
# ===========================================================================


@pytest.mark.unit
class TestExecuteScriptPropertyBased:
    """Property-based tests for execute_script using Hypothesis."""

    @given(
        message=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N"),
                whitelist_characters=" _-",
            ),
            min_size=1,
            max_size=50,
        ),
    )
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    async def test_stdout_always_contains_printed_message(
        self, tmp_path: Path, message: str
    ) -> None:
        """Property: stdout always contains the printed message."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            script = _write_script(tmp_dir, f"print({message!r})")
            result = await execute_script(
                script_path=script,
                working_dir=tmp,
                timeout_seconds=10,
            )
            assert message in result.stdout

    @given(
        exit_code=st.integers(min_value=0, max_value=125),
    )
    @settings(
        max_examples=20,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    async def test_exit_code_always_captured(
        self, tmp_path: Path, exit_code: int
    ) -> None:
        """Property: any exit code used in sys.exit is captured without raising."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            script = _write_script(
                tmp_dir,
                f"""\
                import sys
                sys.exit({exit_code})
                """,
            )
            result = await execute_script(
                script_path=script,
                working_dir=tmp,
                timeout_seconds=10,
            )
            assert result.exit_code == exit_code

    @given(
        message=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N"),
                whitelist_characters=" _-",
            ),
            min_size=1,
            max_size=50,
        ),
    )
    @settings(
        max_examples=15,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    async def test_duration_always_positive(self, tmp_path: Path, message: str) -> None:
        """Property: duration_seconds is always positive regardless of script content."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            script = _write_script(tmp_dir, f"print({message!r})")
            result = await execute_script(
                script_path=script,
                working_dir=tmp,
                timeout_seconds=10,
            )
            assert result.duration_seconds > 0.0

    @given(
        message=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N"),
                whitelist_characters=" _-",
            ),
            min_size=1,
            max_size=50,
        ),
    )
    @settings(
        max_examples=15,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    async def test_successful_script_always_not_timed_out(
        self, tmp_path: Path, message: str
    ) -> None:
        """Property: a fast successful script never reports timed_out=True."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            script = _write_script(tmp_dir, f"print({message!r})")
            result = await execute_script(
                script_path=script,
                working_dir=tmp,
                timeout_seconds=30,
            )
            assert result.timed_out is False


# ===========================================================================
# REQ-EX-007: execute_script -- Parametrized Exit Codes
# ===========================================================================


@pytest.mark.unit
class TestExecuteScriptParametrizedExitCodes:
    """execute_script captures various exit codes correctly (REQ-EX-007)."""

    @pytest.mark.parametrize(
        "code",
        [0, 1, 2, 42, 127],
        ids=["zero", "one", "two", "forty_two", "one_twenty_seven"],
    )
    async def test_various_exit_codes(self, tmp_path: Path, code: int) -> None:
        """Exit codes from 0 to 127 are faithfully captured."""
        script = _write_script(
            tmp_path,
            f"""\
            import sys
            sys.exit({code})
            """,
        )
        result = await execute_script(
            script_path=script,
            working_dir=str(tmp_path),
            timeout_seconds=10,
        )
        assert result.exit_code == code
