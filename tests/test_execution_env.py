"""Tests for MLE-STAR working directory and environment setup (Task 11).

Validates ``setup_working_directory``, ``clean_output_directory``,
``detect_gpu_info``, and ``build_execution_env`` functions defined in
``src/mle_star/execution.py``.  These tests are written TDD-first -- the
implementation does not yet exist.  They serve as the executable
specification for REQ-EX-001 through REQ-EX-004.

Refs:
    SRS 01c (Execution Environment), IMPLEMENTATION_PLAN.md Task 11.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import MagicMock, patch

from hypothesis import given, settings, strategies as st
from mle_star.execution import (
    build_execution_env,
    clean_output_directory,
    detect_gpu_info,
    setup_working_directory,
)
import pytest

# ===========================================================================
# REQ-EX-001: setup_working_directory -- Happy Path
# ===========================================================================


@pytest.mark.unit
class TestSetupWorkingDirectoryHappyPath:
    """setup_working_directory creates input/ and final/ dirs (REQ-EX-001)."""

    def test_creates_input_directory(self, tmp_path: Path) -> None:
        """The input/ subdirectory is created under base_path."""
        base = str(tmp_path / "workspace")
        setup_working_directory(base)
        assert (Path(base) / "input").is_dir()

    def test_creates_final_directory(self, tmp_path: Path) -> None:
        """The final/ subdirectory is created under base_path."""
        base = str(tmp_path / "workspace")
        setup_working_directory(base)
        assert (Path(base) / "final").is_dir()

    def test_creates_both_directories(self, tmp_path: Path) -> None:
        """Both input/ and final/ subdirectories are created."""
        base = str(tmp_path / "workspace")
        setup_working_directory(base)
        assert (Path(base) / "input").is_dir()
        assert (Path(base) / "final").is_dir()

    def test_returns_absolute_path(self, tmp_path: Path) -> None:
        """Returns the absolute path to base_path."""
        base = str(tmp_path / "workspace")
        result = setup_working_directory(base)
        assert os.path.isabs(result)

    def test_returns_string_type(self, tmp_path: Path) -> None:
        """Return value is a string."""
        base = str(tmp_path / "workspace")
        result = setup_working_directory(base)
        assert isinstance(result, str)

    def test_returned_path_matches_base_path(self, tmp_path: Path) -> None:
        """Returned path resolves to the same location as base_path."""
        base = str(tmp_path / "workspace")
        result = setup_working_directory(base)
        assert Path(result).resolve() == Path(base).resolve()

    def test_creates_base_directory_if_not_exists(self, tmp_path: Path) -> None:
        """The base_path directory itself is created if it does not exist."""
        base = str(tmp_path / "nested" / "deep" / "workspace")
        setup_working_directory(base)
        assert Path(base).is_dir()

    def test_base_path_exists_after_call(self, tmp_path: Path) -> None:
        """After calling, the base_path directory exists."""
        base = str(tmp_path / "workspace")
        setup_working_directory(base)
        assert Path(base).exists()


# ===========================================================================
# REQ-EX-001: setup_working_directory -- Idempotency
# ===========================================================================


@pytest.mark.unit
class TestSetupWorkingDirectoryIdempotency:
    """setup_working_directory is idempotent; calling twice does not error (REQ-EX-001)."""

    def test_calling_twice_does_not_raise(self, tmp_path: Path) -> None:
        """Calling setup_working_directory twice on the same path does not raise."""
        base = str(tmp_path / "workspace")
        setup_working_directory(base)
        # Second call should not raise
        setup_working_directory(base)

    def test_directories_still_exist_after_second_call(self, tmp_path: Path) -> None:
        """Both subdirectories still exist after a second call."""
        base = str(tmp_path / "workspace")
        setup_working_directory(base)
        setup_working_directory(base)
        assert (Path(base) / "input").is_dir()
        assert (Path(base) / "final").is_dir()

    def test_existing_files_in_input_preserved(self, tmp_path: Path) -> None:
        """Files already in input/ are preserved when called again."""
        base = str(tmp_path / "workspace")
        setup_working_directory(base)
        # Create a file in input/
        sentinel = Path(base) / "input" / "train.csv"
        sentinel.write_text("data")
        # Call again
        setup_working_directory(base)
        assert sentinel.exists()
        assert sentinel.read_text() == "data"

    def test_existing_files_in_final_preserved(self, tmp_path: Path) -> None:
        """Files already in final/ are preserved when called again."""
        base = str(tmp_path / "workspace")
        setup_working_directory(base)
        sentinel = Path(base) / "final" / "submission.csv"
        sentinel.write_text("output")
        setup_working_directory(base)
        assert sentinel.exists()
        assert sentinel.read_text() == "output"

    def test_return_value_consistent_across_calls(self, tmp_path: Path) -> None:
        """Return value is the same across multiple calls."""
        base = str(tmp_path / "workspace")
        first = setup_working_directory(base)
        second = setup_working_directory(base)
        assert first == second


# ===========================================================================
# REQ-EX-001: setup_working_directory -- Edge Cases
# ===========================================================================


@pytest.mark.unit
class TestSetupWorkingDirectoryEdgeCases:
    """setup_working_directory edge cases for boundary inputs (REQ-EX-001)."""

    def test_with_trailing_slash(self, tmp_path: Path) -> None:
        """Handles base_path with a trailing slash."""
        base = str(tmp_path / "workspace") + "/"
        result = setup_working_directory(base)
        assert isinstance(result, str)
        assert (Path(result) / "input").is_dir() or (
            Path(base).resolve() / "input"
        ).is_dir()

    def test_with_relative_path_returns_absolute(self, tmp_path: Path) -> None:
        """Even if given a relative-ish path, returns an absolute path."""
        base = str(tmp_path / "workspace")
        result = setup_working_directory(base)
        assert os.path.isabs(result)


# ===========================================================================
# REQ-EX-001: setup_working_directory -- Property-based
# ===========================================================================


@pytest.mark.unit
class TestSetupWorkingDirectoryPropertyBased:
    """Property-based tests for setup_working_directory using Hypothesis."""

    @given(
        dirname=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N"),
                whitelist_characters="_-",
            ),
            min_size=1,
            max_size=20,
        ),
    )
    @settings(max_examples=30)
    def test_always_creates_both_subdirs(self, dirname: str) -> None:
        """Property: for any valid directory name, both subdirs are created."""
        with tempfile.TemporaryDirectory() as tmp:
            base = str(Path(tmp) / dirname)
            setup_working_directory(base)
            assert (Path(base) / "input").is_dir()
            assert (Path(base) / "final").is_dir()

    @given(
        dirname=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N"),
                whitelist_characters="_-",
            ),
            min_size=1,
            max_size=20,
        ),
    )
    @settings(max_examples=30)
    def test_return_value_is_always_absolute(self, dirname: str) -> None:
        """Property: return value is always an absolute path."""
        with tempfile.TemporaryDirectory() as tmp:
            base = str(Path(tmp) / dirname)
            result = setup_working_directory(base)
            assert os.path.isabs(result)


# ===========================================================================
# REQ-EX-002: clean_output_directory -- Happy Path
# ===========================================================================


@pytest.mark.unit
class TestCleanOutputDirectoryHappyPath:
    """clean_output_directory removes files in final/ (REQ-EX-002)."""

    def test_removes_single_file(self, tmp_path: Path) -> None:
        """A single file in final/ is removed."""
        base = tmp_path / "workspace"
        final_dir = base / "final"
        final_dir.mkdir(parents=True)
        (final_dir / "submission.csv").write_text("data")

        clean_output_directory(str(base))

        assert final_dir.is_dir()
        assert list(final_dir.iterdir()) == []

    def test_removes_multiple_files(self, tmp_path: Path) -> None:
        """Multiple files in final/ are all removed."""
        base = tmp_path / "workspace"
        final_dir = base / "final"
        final_dir.mkdir(parents=True)
        (final_dir / "file1.csv").write_text("a")
        (final_dir / "file2.csv").write_text("b")
        (final_dir / "file3.txt").write_text("c")

        clean_output_directory(str(base))

        assert final_dir.is_dir()
        remaining = list(final_dir.iterdir())
        assert remaining == []

    def test_final_directory_still_exists_after_cleaning(self, tmp_path: Path) -> None:
        """The final/ directory itself is not deleted."""
        base = tmp_path / "workspace"
        final_dir = base / "final"
        final_dir.mkdir(parents=True)
        (final_dir / "output.txt").write_text("data")

        clean_output_directory(str(base))

        assert final_dir.is_dir()

    def test_does_not_affect_input_directory(self, tmp_path: Path) -> None:
        """Files in input/ are not affected by cleaning."""
        base = tmp_path / "workspace"
        input_dir = base / "input"
        final_dir = base / "final"
        input_dir.mkdir(parents=True)
        final_dir.mkdir(parents=True)
        (input_dir / "train.csv").write_text("training data")
        (final_dir / "output.csv").write_text("output")

        clean_output_directory(str(base))

        assert (input_dir / "train.csv").exists()
        assert (input_dir / "train.csv").read_text() == "training data"

    def test_returns_none(self, tmp_path: Path) -> None:
        """clean_output_directory returns None."""
        base = tmp_path / "workspace"
        final_dir = base / "final"
        final_dir.mkdir(parents=True)

        result = clean_output_directory(str(base))  # type: ignore[func-returns-value]

        assert result is None


# ===========================================================================
# REQ-EX-002: clean_output_directory -- Empty Directory
# ===========================================================================


@pytest.mark.unit
class TestCleanOutputDirectoryEmpty:
    """clean_output_directory handles already-empty directories gracefully (REQ-EX-002)."""

    def test_empty_final_does_not_raise(self, tmp_path: Path) -> None:
        """Cleaning an already-empty final/ directory does not raise."""
        base = tmp_path / "workspace"
        final_dir = base / "final"
        final_dir.mkdir(parents=True)

        # Should not raise
        clean_output_directory(str(base))

    def test_empty_final_still_exists(self, tmp_path: Path) -> None:
        """After cleaning an empty final/, the directory still exists."""
        base = tmp_path / "workspace"
        final_dir = base / "final"
        final_dir.mkdir(parents=True)

        clean_output_directory(str(base))

        assert final_dir.is_dir()

    def test_calling_twice_on_empty_does_not_raise(self, tmp_path: Path) -> None:
        """Calling clean twice on an already-cleaned directory is safe."""
        base = tmp_path / "workspace"
        final_dir = base / "final"
        final_dir.mkdir(parents=True)
        (final_dir / "file.txt").write_text("data")

        clean_output_directory(str(base))
        # Second call on now-empty directory should not raise
        clean_output_directory(str(base))

        assert final_dir.is_dir()


# ===========================================================================
# REQ-EX-002: clean_output_directory -- Edge Cases
# ===========================================================================


@pytest.mark.unit
class TestCleanOutputDirectoryEdgeCases:
    """clean_output_directory edge cases (REQ-EX-002)."""

    def test_removes_files_with_various_extensions(self, tmp_path: Path) -> None:
        """Files with different extensions are all removed."""
        base = tmp_path / "workspace"
        final_dir = base / "final"
        final_dir.mkdir(parents=True)
        for ext in [".csv", ".txt", ".pkl", ".json", ".py"]:
            (final_dir / f"output{ext}").write_text("content")

        clean_output_directory(str(base))

        assert list(final_dir.iterdir()) == []

    def test_removes_hidden_files(self, tmp_path: Path) -> None:
        """Hidden files (dot-prefixed) in final/ are also removed."""
        base = tmp_path / "workspace"
        final_dir = base / "final"
        final_dir.mkdir(parents=True)
        (final_dir / ".hidden").write_text("secret")

        clean_output_directory(str(base))

        assert list(final_dir.iterdir()) == []


# ===========================================================================
# REQ-EX-002: clean_output_directory -- Property-based
# ===========================================================================


@pytest.mark.unit
class TestCleanOutputDirectoryPropertyBased:
    """Property-based tests for clean_output_directory using Hypothesis."""

    @given(
        num_files=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=20)
    def test_final_dir_always_empty_after_clean(self, num_files: int) -> None:
        """Property: regardless of file count, final/ is empty after cleaning."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "workspace"
            final_dir = base / "final"
            final_dir.mkdir(parents=True)

            for i in range(num_files):
                (final_dir / f"file_{i}.txt").write_text(f"content_{i}")

            clean_output_directory(str(base))

            # Directory must exist and be empty of files
            assert final_dir.is_dir()
            remaining_files = [p for p in final_dir.iterdir() if p.is_file()]
            assert remaining_files == []

    @given(
        num_files=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=20)
    def test_final_dir_always_exists_after_clean(self, num_files: int) -> None:
        """Property: final/ directory always exists after cleaning."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "workspace"
            final_dir = base / "final"
            final_dir.mkdir(parents=True)

            for i in range(num_files):
                (final_dir / f"file_{i}.txt").write_text(f"content_{i}")

            clean_output_directory(str(base))

            assert final_dir.is_dir()


# ===========================================================================
# REQ-EX-003: detect_gpu_info -- Return Structure
# ===========================================================================


@pytest.mark.unit
class TestDetectGpuInfoReturnStructure:
    """detect_gpu_info returns dict with correct keys (REQ-EX-003)."""

    def test_returns_dict(self) -> None:
        """detect_gpu_info returns a dict."""
        with patch("mle_star.execution.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("nvidia-smi not found")
            result = detect_gpu_info()
        assert isinstance(result, dict)

    def test_has_cuda_available_key(self) -> None:
        """Returned dict contains 'cuda_available' key."""
        with patch("mle_star.execution.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("nvidia-smi not found")
            result = detect_gpu_info()
        assert "cuda_available" in result

    def test_has_gpu_count_key(self) -> None:
        """Returned dict contains 'gpu_count' key."""
        with patch("mle_star.execution.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("nvidia-smi not found")
            result = detect_gpu_info()
        assert "gpu_count" in result

    def test_has_gpu_names_key(self) -> None:
        """Returned dict contains 'gpu_names' key."""
        with patch("mle_star.execution.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("nvidia-smi not found")
            result = detect_gpu_info()
        assert "gpu_names" in result

    def test_has_exactly_three_keys(self) -> None:
        """Returned dict has exactly three keys."""
        with patch("mle_star.execution.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("nvidia-smi not found")
            result = detect_gpu_info()
        assert set(result.keys()) == {"cuda_available", "gpu_count", "gpu_names"}

    def test_cuda_available_is_bool(self) -> None:
        """'cuda_available' value is a bool."""
        with patch("mle_star.execution.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("nvidia-smi not found")
            result = detect_gpu_info()
        assert isinstance(result["cuda_available"], bool)

    def test_gpu_count_is_int(self) -> None:
        """'gpu_count' value is an int."""
        with patch("mle_star.execution.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("nvidia-smi not found")
            result = detect_gpu_info()
        assert isinstance(result["gpu_count"], int)

    def test_gpu_names_is_list(self) -> None:
        """'gpu_names' value is a list."""
        with patch("mle_star.execution.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("nvidia-smi not found")
            result = detect_gpu_info()
        assert isinstance(result["gpu_names"], list)


# ===========================================================================
# REQ-EX-003: detect_gpu_info -- No GPU (Detection Fails)
# ===========================================================================


@pytest.mark.unit
class TestDetectGpuInfoNoGpu:
    """detect_gpu_info returns safe defaults when detection fails (REQ-EX-003)."""

    def test_file_not_found_returns_defaults(self) -> None:
        """FileNotFoundError (nvidia-smi missing) returns safe defaults."""
        with patch("mle_star.execution.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("nvidia-smi not found")
            result = detect_gpu_info()

        assert result["cuda_available"] is False
        assert result["gpu_count"] == 0
        assert result["gpu_names"] == []

    def test_subprocess_timeout_returns_defaults(self) -> None:
        """subprocess.TimeoutExpired returns safe defaults."""
        with patch("mle_star.execution.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(
                cmd="nvidia-smi", timeout=5
            )
            result = detect_gpu_info()

        assert result["cuda_available"] is False
        assert result["gpu_count"] == 0
        assert result["gpu_names"] == []

    def test_nonzero_exit_code_returns_defaults(self) -> None:
        """Non-zero exit code from nvidia-smi returns safe defaults."""
        with patch("mle_star.execution.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            result = detect_gpu_info()

        assert result["cuda_available"] is False
        assert result["gpu_count"] == 0
        assert result["gpu_names"] == []

    def test_os_error_returns_defaults(self) -> None:
        """OSError during detection returns safe defaults."""
        with patch("mle_star.execution.subprocess.run") as mock_run:
            mock_run.side_effect = OSError("Permission denied")
            result = detect_gpu_info()

        assert result["cuda_available"] is False
        assert result["gpu_count"] == 0
        assert result["gpu_names"] == []

    def test_never_raises_exception(self) -> None:
        """detect_gpu_info must never raise an exception."""
        with patch("mle_star.execution.subprocess.run") as mock_run:
            mock_run.side_effect = RuntimeError("Unexpected error")
            # Must not raise
            result = detect_gpu_info()
        assert isinstance(result, dict)

    def test_empty_stdout_returns_defaults(self) -> None:
        """Empty stdout from nvidia-smi returns safe defaults."""
        with patch("mle_star.execution.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="")
            result = detect_gpu_info()

        assert result["cuda_available"] is False
        assert result["gpu_count"] == 0
        assert result["gpu_names"] == []


# ===========================================================================
# REQ-EX-003: detect_gpu_info -- GPU Detected
# ===========================================================================


@pytest.mark.unit
class TestDetectGpuInfoWithGpu:
    """detect_gpu_info correctly parses nvidia-smi output (REQ-EX-003)."""

    def test_single_gpu_detected(self) -> None:
        """Single GPU reported by nvidia-smi is correctly parsed."""
        nvidia_output = "NVIDIA GeForce RTX 3090\n"
        with patch("mle_star.execution.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=nvidia_output)
            result = detect_gpu_info()

        assert result["cuda_available"] is True
        assert result["gpu_count"] == 1
        assert result["gpu_names"] == ["NVIDIA GeForce RTX 3090"]

    def test_multiple_gpus_detected(self) -> None:
        """Multiple GPUs reported by nvidia-smi are correctly parsed."""
        nvidia_output = "NVIDIA A100\nNVIDIA A100\n"
        with patch("mle_star.execution.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=nvidia_output)
            result = detect_gpu_info()

        assert result["cuda_available"] is True
        assert result["gpu_count"] == 2
        assert result["gpu_names"] == ["NVIDIA A100", "NVIDIA A100"]

    def test_four_gpus_detected(self) -> None:
        """Four GPUs reported by nvidia-smi are correctly parsed."""
        nvidia_output = "NVIDIA A100\nNVIDIA A100\nNVIDIA A100\nNVIDIA A100\n"
        with patch("mle_star.execution.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=nvidia_output)
            result = detect_gpu_info()

        assert result["cuda_available"] is True
        assert result["gpu_count"] == 4
        assert result["gpu_names"] == ["NVIDIA A100"] * 4

    def test_mixed_gpu_models(self) -> None:
        """Different GPU models are correctly reported."""
        nvidia_output = "NVIDIA GeForce RTX 3090\nNVIDIA Tesla V100\n"
        with patch("mle_star.execution.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=nvidia_output)
            result = detect_gpu_info()

        assert result["cuda_available"] is True
        assert result["gpu_count"] == 2
        assert result["gpu_names"] == [
            "NVIDIA GeForce RTX 3090",
            "NVIDIA Tesla V100",
        ]

    def test_gpu_names_are_stripped(self) -> None:
        """GPU names have leading/trailing whitespace stripped."""
        nvidia_output = "  NVIDIA A100  \n  NVIDIA V100  \n"
        with patch("mle_star.execution.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=nvidia_output)
            result = detect_gpu_info()

        assert result["gpu_names"] == ["NVIDIA A100", "NVIDIA V100"]

    def test_blank_lines_in_output_ignored(self) -> None:
        """Blank lines in nvidia-smi output are ignored."""
        nvidia_output = "NVIDIA A100\n\n\nNVIDIA V100\n\n"
        with patch("mle_star.execution.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=nvidia_output)
            result = detect_gpu_info()

        assert result["gpu_count"] == 2
        assert result["gpu_names"] == ["NVIDIA A100", "NVIDIA V100"]


# ===========================================================================
# REQ-EX-003: detect_gpu_info -- Property-based
# ===========================================================================


@pytest.mark.unit
class TestDetectGpuInfoPropertyBased:
    """Property-based tests for detect_gpu_info using Hypothesis."""

    @given(
        gpu_names=st.lists(
            st.text(
                alphabet=st.characters(
                    whitelist_categories=("L", "N"),
                    whitelist_characters=" -",
                ),
                min_size=3,
                max_size=40,
            ),
            min_size=1,
            max_size=8,
        ),
    )
    @settings(max_examples=30)
    def test_gpu_count_equals_gpu_names_length(self, gpu_names: list[str]) -> None:
        """Property: gpu_count always equals len(gpu_names) when GPUs detected."""
        nvidia_output = "\n".join(gpu_names) + "\n"
        with patch("mle_star.execution.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=nvidia_output)
            result = detect_gpu_info()

        gpu_names_list = result["gpu_names"]
        assert isinstance(gpu_names_list, list)
        assert result["gpu_count"] == len(gpu_names_list)

    @given(
        gpu_names=st.lists(
            st.text(
                alphabet=st.characters(
                    whitelist_categories=("L", "N"),
                    whitelist_characters=" -",
                ),
                min_size=3,
                max_size=40,
            ),
            min_size=1,
            max_size=8,
        ),
    )
    @settings(max_examples=30)
    def test_cuda_available_true_when_gpus_present(self, gpu_names: list[str]) -> None:
        """Property: cuda_available is True when at least one GPU is detected."""
        nvidia_output = "\n".join(gpu_names) + "\n"
        with patch("mle_star.execution.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=nvidia_output)
            result = detect_gpu_info()

        assert result["cuda_available"] is True

    @given(
        exception_type=st.sampled_from(
            [FileNotFoundError, OSError, RuntimeError, PermissionError]
        ),
    )
    @settings(max_examples=10)
    def test_any_exception_returns_safe_defaults(
        self, exception_type: type[Exception]
    ) -> None:
        """Property: any exception type results in safe default values."""
        with patch("mle_star.execution.subprocess.run") as mock_run:
            mock_run.side_effect = exception_type("simulated error")
            result = detect_gpu_info()

        assert result == {
            "cuda_available": False,
            "gpu_count": 0,
            "gpu_names": [],
        }


# ===========================================================================
# REQ-EX-004: build_execution_env -- Basic Behavior
# ===========================================================================


@pytest.mark.unit
class TestBuildExecutionEnvBasic:
    """build_execution_env sets PYTHONUNBUFFERED and PYTHONHASHSEED (REQ-EX-004)."""

    def test_returns_dict(self) -> None:
        """build_execution_env returns a dict."""
        result = build_execution_env()
        assert isinstance(result, dict)

    def test_sets_pythonunbuffered(self) -> None:
        """PYTHONUNBUFFERED is set to '1'."""
        result = build_execution_env()
        assert result["PYTHONUNBUFFERED"] == "1"

    def test_sets_pythonhashseed(self) -> None:
        """PYTHONHASHSEED is set to '0'."""
        result = build_execution_env()
        assert result["PYTHONHASHSEED"] == "0"

    def test_pythonunbuffered_value_is_string(self) -> None:
        """PYTHONUNBUFFERED value is a string, not an int."""
        result = build_execution_env()
        assert isinstance(result["PYTHONUNBUFFERED"], str)

    def test_pythonhashseed_value_is_string(self) -> None:
        """PYTHONHASHSEED value is a string, not an int."""
        result = build_execution_env()
        assert isinstance(result["PYTHONHASHSEED"], str)


# ===========================================================================
# REQ-EX-004: build_execution_env -- Returns Copy of os.environ
# ===========================================================================


@pytest.mark.unit
class TestBuildExecutionEnvCopiesEnviron:
    """build_execution_env returns a copy of os.environ with additions (REQ-EX-004)."""

    def test_contains_existing_env_vars(self) -> None:
        """Returned dict contains existing environment variables."""
        with patch.dict(os.environ, {"MY_TEST_VAR": "hello"}, clear=False):
            result = build_execution_env()
        assert result["MY_TEST_VAR"] == "hello"

    def test_does_not_modify_original_environ(self) -> None:
        """Calling build_execution_env does not modify os.environ."""
        original_keys = set(os.environ.keys())
        _result = build_execution_env()
        # _result may have extra keys, but os.environ should be unchanged
        current_keys = set(os.environ.keys())
        assert original_keys == current_keys

    def test_returns_new_dict_not_os_environ(self) -> None:
        """Returned dict is not the same object as os.environ."""
        result = build_execution_env()
        assert result is not os.environ

    def test_mutations_to_result_do_not_affect_environ(self) -> None:
        """Mutating the returned dict does not affect os.environ."""
        result = build_execution_env()
        result["TOTALLY_NEW_VAR"] = "test"
        assert "TOTALLY_NEW_VAR" not in os.environ

    def test_overrides_existing_pythonunbuffered(self) -> None:
        """Overrides existing PYTHONUNBUFFERED even if already set."""
        with patch.dict(os.environ, {"PYTHONUNBUFFERED": "0"}, clear=False):
            result = build_execution_env()
        assert result["PYTHONUNBUFFERED"] == "1"

    def test_overrides_existing_pythonhashseed(self) -> None:
        """Overrides existing PYTHONHASHSEED even if already set."""
        with patch.dict(os.environ, {"PYTHONHASHSEED": "42"}, clear=False):
            result = build_execution_env()
        assert result["PYTHONHASHSEED"] == "0"


# ===========================================================================
# REQ-EX-004: build_execution_env -- gpu_indices=None (Default)
# ===========================================================================


@pytest.mark.unit
class TestBuildExecutionEnvNoGpuIndices:
    """build_execution_env with gpu_indices=None does not set CUDA_VISIBLE_DEVICES (REQ-EX-004)."""

    def test_default_does_not_set_cuda_visible_devices(self) -> None:
        """When gpu_indices is None (default), CUDA_VISIBLE_DEVICES is not set."""
        with patch.dict(os.environ, {}, clear=False):
            # Ensure CUDA_VISIBLE_DEVICES is not in the base env
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            result = build_execution_env()
        assert "CUDA_VISIBLE_DEVICES" not in result

    def test_explicit_none_does_not_set_cuda_visible_devices(self) -> None:
        """Explicitly passing gpu_indices=None does not set CUDA_VISIBLE_DEVICES."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            result = build_execution_env(gpu_indices=None)
        assert "CUDA_VISIBLE_DEVICES" not in result

    def test_inherits_parent_cuda_visible_devices(self) -> None:
        """When gpu_indices is None, inherits parent's CUDA_VISIBLE_DEVICES if set."""
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3"}, clear=False):
            result = build_execution_env(gpu_indices=None)
        assert result["CUDA_VISIBLE_DEVICES"] == "2,3"


# ===========================================================================
# REQ-EX-004: build_execution_env -- gpu_indices with Values
# ===========================================================================


@pytest.mark.unit
class TestBuildExecutionEnvWithGpuIndices:
    """build_execution_env with gpu_indices sets CUDA_VISIBLE_DEVICES (REQ-EX-004)."""

    def test_single_gpu_index(self) -> None:
        """gpu_indices=[0] sets CUDA_VISIBLE_DEVICES='0'."""
        result = build_execution_env(gpu_indices=[0])
        assert result["CUDA_VISIBLE_DEVICES"] == "0"

    def test_two_gpu_indices(self) -> None:
        """gpu_indices=[0, 1] sets CUDA_VISIBLE_DEVICES='0,1'."""
        result = build_execution_env(gpu_indices=[0, 1])
        assert result["CUDA_VISIBLE_DEVICES"] == "0,1"

    def test_multiple_gpu_indices(self) -> None:
        """gpu_indices=[0, 2, 4] sets CUDA_VISIBLE_DEVICES='0,2,4'."""
        result = build_execution_env(gpu_indices=[0, 2, 4])
        assert result["CUDA_VISIBLE_DEVICES"] == "0,2,4"

    def test_empty_gpu_indices(self) -> None:
        """gpu_indices=[] sets CUDA_VISIBLE_DEVICES=''."""
        result = build_execution_env(gpu_indices=[])
        assert result["CUDA_VISIBLE_DEVICES"] == ""

    def test_single_high_index(self) -> None:
        """gpu_indices=[7] sets CUDA_VISIBLE_DEVICES='7'."""
        result = build_execution_env(gpu_indices=[7])
        assert result["CUDA_VISIBLE_DEVICES"] == "7"

    def test_overrides_parent_cuda_visible_devices(self) -> None:
        """gpu_indices overrides any existing CUDA_VISIBLE_DEVICES in parent env."""
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}, clear=False):
            result = build_execution_env(gpu_indices=[0])
        assert result["CUDA_VISIBLE_DEVICES"] == "0"

    def test_acceptance_criterion_two_gpus(self) -> None:
        """Acceptance criterion: build_execution_env(gpu_indices=[0, 1]) -> '0,1'."""
        result = build_execution_env(gpu_indices=[0, 1])
        assert result["CUDA_VISIBLE_DEVICES"] == "0,1"

    def test_acceptance_criterion_empty_list(self) -> None:
        """Acceptance criterion: build_execution_env(gpu_indices=[]) -> ''."""
        result = build_execution_env(gpu_indices=[])
        assert result["CUDA_VISIBLE_DEVICES"] == ""


# ===========================================================================
# REQ-EX-004: build_execution_env -- All Required Vars Present
# ===========================================================================


@pytest.mark.unit
class TestBuildExecutionEnvAllVars:
    """build_execution_env with gpu_indices sets all three env vars (REQ-EX-004)."""

    def test_all_three_vars_when_gpu_indices_provided(self) -> None:
        """PYTHONUNBUFFERED, PYTHONHASHSEED, and CUDA_VISIBLE_DEVICES all present."""
        result = build_execution_env(gpu_indices=[0])
        assert result["PYTHONUNBUFFERED"] == "1"
        assert result["PYTHONHASHSEED"] == "0"
        assert result["CUDA_VISIBLE_DEVICES"] == "0"

    def test_only_two_vars_when_no_gpu_indices(self) -> None:
        """Only PYTHONUNBUFFERED and PYTHONHASHSEED set; no CUDA_VISIBLE_DEVICES added."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            result = build_execution_env()
        assert "PYTHONUNBUFFERED" in result
        assert "PYTHONHASHSEED" in result
        assert "CUDA_VISIBLE_DEVICES" not in result


# ===========================================================================
# REQ-EX-004: build_execution_env -- Return Type Consistency
# ===========================================================================


@pytest.mark.unit
class TestBuildExecutionEnvReturnTypes:
    """build_execution_env return value types are correct (REQ-EX-004)."""

    def test_all_keys_are_strings(self) -> None:
        """All keys in the returned dict are strings."""
        result = build_execution_env(gpu_indices=[0])
        for key in result:
            assert isinstance(key, str), f"Key {key!r} is not a string"

    def test_all_values_are_strings(self) -> None:
        """All values in the returned dict are strings."""
        result = build_execution_env(gpu_indices=[0])
        for key, value in result.items():
            assert isinstance(value, str), (
                f"Value for key {key!r} is {type(value).__name__}, not str"
            )

    def test_cuda_visible_devices_value_is_string(self) -> None:
        """CUDA_VISIBLE_DEVICES value is a string, not a list."""
        result = build_execution_env(gpu_indices=[0, 1])
        assert isinstance(result["CUDA_VISIBLE_DEVICES"], str)


# ===========================================================================
# REQ-EX-004: build_execution_env -- Property-based
# ===========================================================================


@pytest.mark.unit
class TestBuildExecutionEnvPropertyBased:
    """Property-based tests for build_execution_env using Hypothesis."""

    @given(
        gpu_indices=st.lists(
            st.integers(min_value=0, max_value=15),
            min_size=0,
            max_size=8,
        ),
    )
    @settings(max_examples=50)
    def test_cuda_visible_devices_is_comma_separated_indices(
        self, gpu_indices: list[int]
    ) -> None:
        """Property: CUDA_VISIBLE_DEVICES is comma-separated string of the indices."""
        result = build_execution_env(gpu_indices=gpu_indices)
        expected = ",".join(str(i) for i in gpu_indices)
        assert result["CUDA_VISIBLE_DEVICES"] == expected

    @given(
        gpu_indices=st.one_of(
            st.none(),
            st.lists(
                st.integers(min_value=0, max_value=15),
                min_size=0,
                max_size=8,
            ),
        ),
    )
    @settings(max_examples=50)
    def test_pythonunbuffered_always_set(self, gpu_indices: list[int] | None) -> None:
        """Property: PYTHONUNBUFFERED is always '1' regardless of gpu_indices."""
        result = build_execution_env(gpu_indices=gpu_indices)
        assert result["PYTHONUNBUFFERED"] == "1"

    @given(
        gpu_indices=st.one_of(
            st.none(),
            st.lists(
                st.integers(min_value=0, max_value=15),
                min_size=0,
                max_size=8,
            ),
        ),
    )
    @settings(max_examples=50)
    def test_pythonhashseed_always_set(self, gpu_indices: list[int] | None) -> None:
        """Property: PYTHONHASHSEED is always '0' regardless of gpu_indices."""
        result = build_execution_env(gpu_indices=gpu_indices)
        assert result["PYTHONHASHSEED"] == "0"

    @given(
        gpu_indices=st.one_of(
            st.none(),
            st.lists(
                st.integers(min_value=0, max_value=15),
                min_size=0,
                max_size=8,
            ),
        ),
    )
    @settings(max_examples=50)
    def test_result_is_never_os_environ(self, gpu_indices: list[int] | None) -> None:
        """Property: returned dict is never the same object as os.environ."""
        result = build_execution_env(gpu_indices=gpu_indices)
        assert result is not os.environ

    @given(
        gpu_indices=st.lists(
            st.integers(min_value=0, max_value=15),
            min_size=1,
            max_size=8,
        ),
    )
    @settings(max_examples=30)
    def test_gpu_indices_preserved_in_order(self, gpu_indices: list[int]) -> None:
        """Property: GPU indices appear in the same order as provided."""
        result = build_execution_env(gpu_indices=gpu_indices)
        parts = result["CUDA_VISIBLE_DEVICES"].split(",")
        parsed = [int(p) for p in parts]
        assert parsed == gpu_indices


# ===========================================================================
# Integration: setup_working_directory + clean_output_directory
# ===========================================================================


@pytest.mark.unit
class TestSetupAndCleanIntegration:
    """Integration tests combining setup and clean operations."""

    def test_setup_then_clean(self, tmp_path: Path) -> None:
        """Setting up then cleaning produces an empty final/ directory."""
        base = str(tmp_path / "workspace")
        setup_working_directory(base)
        # Add files to final/
        (Path(base) / "final" / "output.csv").write_text("data")

        clean_output_directory(base)

        assert (Path(base) / "final").is_dir()
        assert list((Path(base) / "final").iterdir()) == []
        # input/ should still exist
        assert (Path(base) / "input").is_dir()

    def test_setup_clean_setup_cycle(self, tmp_path: Path) -> None:
        """Full cycle: setup -> populate -> clean -> setup again works."""
        base = str(tmp_path / "workspace")
        setup_working_directory(base)
        (Path(base) / "final" / "output.csv").write_text("data")
        clean_output_directory(base)
        setup_working_directory(base)

        assert (Path(base) / "input").is_dir()
        assert (Path(base) / "final").is_dir()
        assert list((Path(base) / "final").iterdir()) == []
