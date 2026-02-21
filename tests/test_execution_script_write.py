"""Tests for write_script function in execution module (Task 12).

Validates ``write_script`` which writes a ``SolutionScript.content`` to disk
with pre-validation for empty content and forbidden exit calls.  Tests are
written TDD-first and serve as the executable specification for REQ-EX-005,
REQ-EX-006, and REQ-EX-044.

Refs:
    SRS 02a (Execution Environment), IMPLEMENTATION_PLAN.md Task 12.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

from hypothesis import given, settings, strategies as st
from mle_star.execution import write_script
from mle_star.models import SolutionPhase, SolutionScript
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_solution(content: str) -> SolutionScript:
    """Create a SolutionScript with the given content for testing."""
    return SolutionScript(content=content, phase=SolutionPhase.INIT)


# ===========================================================================
# REQ-EX-005: write_script -- Happy Path
# ===========================================================================


@pytest.mark.unit
class TestWriteScriptHappyPath:
    """write_script writes content to disk and returns absolute path (REQ-EX-005)."""

    def test_returns_string(self, tmp_path: Path) -> None:
        """Return value is a string."""
        solution = _make_solution("print('hello')")
        result = write_script(solution, str(tmp_path))
        assert isinstance(result, str)

    def test_returns_absolute_path(self, tmp_path: Path) -> None:
        """Returned path is absolute."""
        solution = _make_solution("print('hello')")
        result = write_script(solution, str(tmp_path))
        assert os.path.isabs(result)

    def test_file_exists_after_write(self, tmp_path: Path) -> None:
        """The file exists at the returned path after writing."""
        solution = _make_solution("print('hello')")
        result = write_script(solution, str(tmp_path))
        assert Path(result).exists()

    def test_file_is_regular_file(self, tmp_path: Path) -> None:
        """The written path is a regular file, not a directory."""
        solution = _make_solution("print('hello')")
        result = write_script(solution, str(tmp_path))
        assert Path(result).is_file()

    def test_file_content_matches_solution(self, tmp_path: Path) -> None:
        """File content is exactly solution.content when read back."""
        content = "import pandas as pd\ndf = pd.read_csv('train.csv')\nprint(df.shape)"
        solution = _make_solution(content)
        result = write_script(solution, str(tmp_path))
        assert Path(result).read_text(encoding="utf-8") == content

    def test_returned_path_ends_with_default_filename(self, tmp_path: Path) -> None:
        """Returned path ends with the default filename 'solution.py'."""
        solution = _make_solution("x = 1")
        result = write_script(solution, str(tmp_path))
        assert Path(result).name == "solution.py"

    def test_returned_path_parent_is_working_dir(self, tmp_path: Path) -> None:
        """Returned path's parent directory matches the working_dir."""
        solution = _make_solution("x = 1")
        result = write_script(solution, str(tmp_path))
        assert Path(result).parent.resolve() == tmp_path.resolve()


# ===========================================================================
# REQ-EX-005: write_script -- Custom Filename
# ===========================================================================


@pytest.mark.unit
class TestWriteScriptCustomFilename:
    """write_script accepts a custom filename parameter (REQ-EX-005)."""

    def test_custom_filename_used(self, tmp_path: Path) -> None:
        """File is written with the custom filename."""
        solution = _make_solution("x = 1")
        result = write_script(solution, str(tmp_path), filename="train.py")
        assert Path(result).name == "train.py"

    def test_custom_filename_content_correct(self, tmp_path: Path) -> None:
        """File with custom filename contains correct content."""
        content = "print('custom')"
        solution = _make_solution(content)
        result = write_script(solution, str(tmp_path), filename="my_script.py")
        assert Path(result).read_text(encoding="utf-8") == content

    def test_custom_filename_returns_absolute_path(self, tmp_path: Path) -> None:
        """Return value with custom filename is still an absolute path."""
        solution = _make_solution("x = 1")
        result = write_script(solution, str(tmp_path), filename="run.py")
        assert os.path.isabs(result)


# ===========================================================================
# REQ-EX-005: write_script -- UTF-8 Encoding
# ===========================================================================


@pytest.mark.unit
class TestWriteScriptUtf8Encoding:
    """write_script writes files with UTF-8 encoding (REQ-EX-005)."""

    def test_ascii_content_roundtrips(self, tmp_path: Path) -> None:
        """Pure ASCII content is written and read back correctly."""
        content = "x = 42\nprint(x)"
        solution = _make_solution(content)
        result = write_script(solution, str(tmp_path))
        assert Path(result).read_text(encoding="utf-8") == content

    def test_unicode_content_roundtrips(self, tmp_path: Path) -> None:
        """Content with Unicode characters roundtrips through UTF-8."""
        content = "# Umlaut: \u00e4\u00f6\u00fc\u00df\nprint('\u00c9l\u00e8ve')"
        solution = _make_solution(content)
        result = write_script(solution, str(tmp_path))
        assert Path(result).read_text(encoding="utf-8") == content

    def test_emoji_content_roundtrips(self, tmp_path: Path) -> None:
        """Content with emoji characters roundtrips through UTF-8."""
        content = "# Status: \u2705\nresult = 'done \U0001f389'"
        solution = _make_solution(content)
        result = write_script(solution, str(tmp_path))
        assert Path(result).read_text(encoding="utf-8") == content

    def test_cjk_content_roundtrips(self, tmp_path: Path) -> None:
        """Content with CJK characters roundtrips through UTF-8."""
        content = "# \u6a21\u578b\u8bad\u7ec3\nprint('\u4f60\u597d\u4e16\u754c')"
        solution = _make_solution(content)
        result = write_script(solution, str(tmp_path))
        assert Path(result).read_text(encoding="utf-8") == content

    def test_file_is_utf8_encoded_on_disk(self, tmp_path: Path) -> None:
        """File bytes on disk are valid UTF-8 encoding of the content."""
        content = "name = '\u00e4\u00f6\u00fc'"
        solution = _make_solution(content)
        result = write_script(solution, str(tmp_path))
        raw_bytes = Path(result).read_bytes()
        assert raw_bytes == content.encode("utf-8")


# ===========================================================================
# REQ-EX-005: write_script -- Overwrite Existing File
# ===========================================================================


@pytest.mark.unit
class TestWriteScriptOverwrite:
    """write_script overwrites any existing file at the target path (REQ-EX-005)."""

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        """An existing file at the target path is overwritten."""
        target = tmp_path / "solution.py"
        target.write_text("old content", encoding="utf-8")

        new_content = "new content = True"
        solution = _make_solution(new_content)
        result = write_script(solution, str(tmp_path))

        assert Path(result).read_text(encoding="utf-8") == new_content

    def test_overwrite_changes_content_completely(self, tmp_path: Path) -> None:
        """After overwrite, no trace of old content remains."""
        target = tmp_path / "solution.py"
        old_content = "x = 'this is the old content that should be gone'"
        target.write_text(old_content, encoding="utf-8")

        new_content = "y = 1"
        solution = _make_solution(new_content)
        write_script(solution, str(tmp_path))

        actual = target.read_text(encoding="utf-8")
        assert actual == new_content
        assert old_content not in actual

    def test_overwrite_returns_same_path(self, tmp_path: Path) -> None:
        """Overwriting returns the same absolute path as writing fresh."""
        target = tmp_path / "solution.py"
        target.write_text("old", encoding="utf-8")

        solution = _make_solution("new = True")
        result = write_script(solution, str(tmp_path))

        assert Path(result).resolve() == target.resolve()


# ===========================================================================
# REQ-EX-006: write_script -- Empty Content Validation
# ===========================================================================


@pytest.mark.unit
class TestWriteScriptEmptyContentValidation:
    """write_script raises ValueError for empty or whitespace-only content (REQ-EX-006)."""

    def test_empty_string_raises_value_error(self, tmp_path: Path) -> None:
        """Empty string content raises ValueError."""
        solution = _make_solution("")
        with pytest.raises(ValueError, match="empty"):
            write_script(solution, str(tmp_path))

    def test_whitespace_only_raises_value_error(self, tmp_path: Path) -> None:
        """Whitespace-only content raises ValueError."""
        solution = _make_solution("   \t  ")
        with pytest.raises(ValueError, match="empty"):
            write_script(solution, str(tmp_path))

    def test_newlines_only_raises_value_error(self, tmp_path: Path) -> None:
        """Newlines-only content raises ValueError."""
        solution = _make_solution("\n\n\n")
        with pytest.raises(ValueError, match="empty"):
            write_script(solution, str(tmp_path))

    def test_tabs_and_spaces_raises_value_error(self, tmp_path: Path) -> None:
        """Tabs and spaces only raises ValueError."""
        solution = _make_solution("\t \t \n \t")
        with pytest.raises(ValueError, match="empty"):
            write_script(solution, str(tmp_path))

    def test_no_file_created_on_empty_content(self, tmp_path: Path) -> None:
        """No file is written when validation fails for empty content."""
        solution = _make_solution("")
        with pytest.raises(ValueError):
            write_script(solution, str(tmp_path))
        assert not (tmp_path / "solution.py").exists()


# ===========================================================================
# REQ-EX-006 / REQ-EX-044: write_script -- Exit Call Validation (Rejected)
# ===========================================================================


@pytest.mark.unit
class TestWriteScriptExitCallsRejected:
    """write_script raises ValueError for content containing exit calls (REQ-EX-006, REQ-EX-044)."""

    def test_exit_call_rejected(self, tmp_path: Path) -> None:
        """Content with exit() is rejected."""
        solution = _make_solution("exit()")
        with pytest.raises(ValueError, match="exit"):
            write_script(solution, str(tmp_path))

    def test_sys_exit_call_rejected(self, tmp_path: Path) -> None:
        """Content with sys.exit() is rejected."""
        solution = _make_solution("import sys\nsys.exit()")
        with pytest.raises(ValueError, match="exit"):
            write_script(solution, str(tmp_path))

    def test_os_exit_call_rejected(self, tmp_path: Path) -> None:
        """Content with os._exit() is rejected."""
        solution = _make_solution("import os\nos._exit(0)")
        with pytest.raises(ValueError, match="exit"):
            write_script(solution, str(tmp_path))

    def test_quit_call_rejected(self, tmp_path: Path) -> None:
        """Content with quit() is rejected."""
        solution = _make_solution("quit()")
        with pytest.raises(ValueError, match="quit"):
            write_script(solution, str(tmp_path))

    def test_sys_exit_with_code_rejected(self, tmp_path: Path) -> None:
        """Content with sys.exit(0) is rejected."""
        solution = _make_solution("import sys\nsys.exit(0)")
        with pytest.raises(ValueError, match="exit"):
            write_script(solution, str(tmp_path))

    def test_sys_exit_with_nonzero_code_rejected(self, tmp_path: Path) -> None:
        """Content with sys.exit(1) is rejected."""
        solution = _make_solution("import sys\nsys.exit(1)")
        with pytest.raises(ValueError, match="exit"):
            write_script(solution, str(tmp_path))

    def test_os_exit_with_code_rejected(self, tmp_path: Path) -> None:
        """Content with os._exit(1) is rejected."""
        solution = _make_solution("import os\nos._exit(1)")
        with pytest.raises(ValueError, match="exit"):
            write_script(solution, str(tmp_path))

    def test_exit_with_message_rejected(self, tmp_path: Path) -> None:
        """Content with exit('error') is rejected."""
        solution = _make_solution("exit('error message')")
        with pytest.raises(ValueError, match="exit"):
            write_script(solution, str(tmp_path))

    def test_quit_with_code_rejected(self, tmp_path: Path) -> None:
        """Content with quit(1) is rejected."""
        solution = _make_solution("quit(1)")
        with pytest.raises(ValueError, match="quit"):
            write_script(solution, str(tmp_path))

    def test_exit_in_comment_still_rejected(self, tmp_path: Path) -> None:
        """Content with exit() in a comment is still rejected (regex does not distinguish)."""
        solution = _make_solution("# exit()\nx = 1")
        with pytest.raises(ValueError, match="exit"):
            write_script(solution, str(tmp_path))

    def test_exit_in_multiline_code_rejected(self, tmp_path: Path) -> None:
        """Content with exit() buried in valid code is rejected."""
        content = "import pandas as pd\ndf = pd.read_csv('data.csv')\nexit()\nprint(df)"
        solution = _make_solution(content)
        with pytest.raises(ValueError, match="exit"):
            write_script(solution, str(tmp_path))

    def test_exit_with_spaces_before_paren_rejected(self, tmp_path: Path) -> None:
        """Content with exit  () (spaces before paren) is rejected."""
        solution = _make_solution("exit  ()")
        with pytest.raises(ValueError, match="exit"):
            write_script(solution, str(tmp_path))

    def test_sys_exit_with_spaces_before_paren_rejected(self, tmp_path: Path) -> None:
        """Content with sys.exit  () is rejected."""
        solution = _make_solution("sys.exit  ()")
        with pytest.raises(ValueError, match="exit"):
            write_script(solution, str(tmp_path))

    def test_no_file_created_on_exit_rejection(self, tmp_path: Path) -> None:
        """No file is written when content contains forbidden exit calls."""
        solution = _make_solution("exit()")
        with pytest.raises(ValueError):
            write_script(solution, str(tmp_path))
        assert not (tmp_path / "solution.py").exists()

    @pytest.mark.parametrize(
        "forbidden_call",
        [
            "exit()",
            "sys.exit()",
            "os._exit(0)",
            "quit()",
            "exit(1)",
            "sys.exit(1)",
            "os._exit(1)",
            "quit(0)",
        ],
        ids=[
            "exit",
            "sys_exit",
            "os_exit_0",
            "quit",
            "exit_1",
            "sys_exit_1",
            "os_exit_1",
            "quit_0",
        ],
    )
    def test_all_forbidden_calls_rejected(
        self, tmp_path: Path, forbidden_call: str
    ) -> None:
        """All four exit call variants and their parameterized forms are rejected."""
        content = f"x = 1\n{forbidden_call}\ny = 2"
        solution = _make_solution(content)
        with pytest.raises(ValueError):
            write_script(solution, str(tmp_path))


# ===========================================================================
# REQ-EX-006 / REQ-EX-044: write_script -- Exit-Like Names NOT Rejected
# ===========================================================================


@pytest.mark.unit
class TestWriteScriptExitLikeNamesAccepted:
    """write_script does NOT reject variable names containing 'exit' (word boundary, REQ-EX-044)."""

    def test_exit_code_variable_accepted(self, tmp_path: Path) -> None:
        """Variable name 'exit_code' is NOT rejected (word boundary check)."""
        content = "exit_code = 0\nprint(exit_code)"
        solution = _make_solution(content)
        result = write_script(solution, str(tmp_path))
        assert Path(result).read_text(encoding="utf-8") == content

    def test_exit_status_variable_accepted(self, tmp_path: Path) -> None:
        """Variable name 'exit_status' is NOT rejected."""
        content = "exit_status = 'success'"
        solution = _make_solution(content)
        result = write_script(solution, str(tmp_path))
        assert Path(result).read_text(encoding="utf-8") == content

    def test_on_exit_function_accepted(self, tmp_path: Path) -> None:
        """Function name 'on_exit' is NOT rejected (no word boundary at 'exit')."""
        content = "def on_exit():\n    pass"
        solution = _make_solution(content)
        result = write_script(solution, str(tmp_path))
        assert Path(result).read_text(encoding="utf-8") == content

    def test_quitting_variable_accepted(self, tmp_path: Path) -> None:
        """Variable name 'quitting' is NOT rejected."""
        content = "quitting = False"
        solution = _make_solution(content)
        result = write_script(solution, str(tmp_path))
        assert Path(result).read_text(encoding="utf-8") == content

    def test_exit_without_parens_accepted(self, tmp_path: Path) -> None:
        """The word 'exit' without parentheses is NOT rejected."""
        content = "# We do not exit here\nx = 1"
        solution = _make_solution(content)
        result = write_script(solution, str(tmp_path))
        assert Path(result).read_text(encoding="utf-8") == content

    def test_quit_without_parens_accepted(self, tmp_path: Path) -> None:
        """The word 'quit' without parentheses is NOT rejected."""
        content = "# Do not quit\nkeep_running = True"
        solution = _make_solution(content)
        result = write_script(solution, str(tmp_path))
        assert Path(result).read_text(encoding="utf-8") == content

    def test_sys_exit_as_attribute_without_call_accepted(self, tmp_path: Path) -> None:
        """Reference to sys.exit without calling it (no parens) is NOT rejected."""
        content = "import sys\nhandler = sys.exit"
        solution = _make_solution(content)
        result = write_script(solution, str(tmp_path))
        assert Path(result).read_text(encoding="utf-8") == content


# ===========================================================================
# REQ-EX-005: write_script -- Property-Based Tests
# ===========================================================================


@pytest.mark.unit
class TestWriteScriptPropertyBased:
    """Property-based tests for write_script using Hypothesis."""

    @given(
        content=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "Z"),
                whitelist_characters="_=+\n ",
            ),
            min_size=1,
            max_size=500,
        ).filter(lambda s: s.strip()),
    )
    @settings(max_examples=50)
    def test_content_roundtrips_through_write_and_read(self, content: str) -> None:
        """Property: any non-empty content (without exit calls) roundtrips correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            solution = _make_solution(content)
            result = write_script(solution, tmp)
            actual = Path(result).read_text(encoding="utf-8")
            assert actual == content

    @given(
        content=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N"),
                whitelist_characters="_=\n ",
            ),
            min_size=1,
            max_size=200,
        ).filter(lambda s: s.strip()),
    )
    @settings(max_examples=50)
    def test_return_value_is_always_absolute(self, content: str) -> None:
        """Property: return value is always an absolute path."""
        with tempfile.TemporaryDirectory() as tmp:
            solution = _make_solution(content)
            result = write_script(solution, tmp)
            assert os.path.isabs(result)

    @given(
        content=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N"),
                whitelist_characters="_=\n ",
            ),
            min_size=1,
            max_size=200,
        ).filter(lambda s: s.strip()),
    )
    @settings(max_examples=50)
    def test_written_file_always_exists(self, content: str) -> None:
        """Property: the file at the returned path always exists after writing."""
        with tempfile.TemporaryDirectory() as tmp:
            solution = _make_solution(content)
            result = write_script(solution, tmp)
            assert Path(result).exists()

    @given(
        content=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N"),
                whitelist_characters="_=\n ",
            ),
            min_size=1,
            max_size=200,
        ).filter(lambda s: s.strip()),
        filename=st.from_regex(r"[a-z_]{1,10}\.py", fullmatch=True),
    )
    @settings(max_examples=30)
    def test_custom_filename_always_used(self, content: str, filename: str) -> None:
        """Property: custom filename is always reflected in the returned path."""
        with tempfile.TemporaryDirectory() as tmp:
            solution = _make_solution(content)
            result = write_script(solution, tmp, filename=filename)
            assert Path(result).name == filename

    @given(
        content=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N"),
                whitelist_characters="_=\n ",
            ),
            min_size=1,
            max_size=200,
        ).filter(lambda s: s.strip()),
    )
    @settings(max_examples=30)
    def test_file_bytes_are_utf8_encoded_content(self, content: str) -> None:
        """Property: file bytes on disk are always the UTF-8 encoding of content."""
        with tempfile.TemporaryDirectory() as tmp:
            solution = _make_solution(content)
            result = write_script(solution, tmp)
            raw_bytes = Path(result).read_bytes()
            assert raw_bytes == content.encode("utf-8")

    @given(
        ws=st.text(
            alphabet=st.sampled_from([" ", "\t", "\n", "\r"]),
            min_size=1,
            max_size=20,
        ),
    )
    @settings(max_examples=30)
    def test_whitespace_only_always_raises(self, ws: str) -> None:
        """Property: any whitespace-only content always raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp:
            solution = _make_solution(ws)
            with pytest.raises(ValueError, match="empty"):
                write_script(solution, tmp)


# ===========================================================================
# REQ-EX-005 / REQ-EX-006: write_script -- Solution Phase Variations
# ===========================================================================


@pytest.mark.unit
class TestWriteScriptWithDifferentPhases:
    """write_script works with SolutionScript from any phase (REQ-EX-005)."""

    @pytest.mark.parametrize(
        "phase",
        list(SolutionPhase),
        ids=[p.value for p in SolutionPhase],
    )
    def test_all_phases_write_successfully(
        self, tmp_path: Path, phase: SolutionPhase
    ) -> None:
        """SolutionScript from any phase can be written."""
        content = f"# phase: {phase.value}\nprint('ok')"
        solution = SolutionScript(content=content, phase=phase)
        result = write_script(solution, str(tmp_path))
        assert Path(result).read_text(encoding="utf-8") == content
