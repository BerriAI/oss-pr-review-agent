"""Unit tests for `_extract_failure_window` — the gather script's
log-truncation helper.

Why this module exists: the previous truncation strategy was
`text[-MAX_LOG_CHARS:]`, which silently discarded the actual failure on
any log where the runner appends post-job cleanup steps after the test
output. Concrete repro: BerriAI/litellm PR #26460's `documentation`
check (test `tests/documentation_tests/test_env_keys.py`) emitted
`Exception: Keys not documented in 'environment settings - Reference':
{'LITELLM_EXPIRED_UI_SESSION_KEY_CLEANUP_*'}` at byte ~49,700 of a
~248,000-byte log; the last 3,000 chars were `git config --local
--unset-all` lines from the runner's cleanup phase. The triage agent
classified the check as infra/unrelated because nothing in its
inputs referenced the diff. That regression motivated the
failure-window helper. This test pins the helper's behavior so we
can't silently re-regress.

Pure unit tests — no LLM, no network, no subprocess.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(REPO_ROOT / "skills/pr-review-agent-skills/litellm-pr-reviewer/scripts"),
)

from gather_pr_triage_data import _extract_failure_window  # noqa: E402


def test_short_log_returned_unchanged():
    """Logs already under MAX_LOG_CHARS are passed through verbatim."""
    short = "everything fine, exit 0\n"
    assert _extract_failure_window(short, max_chars=3000) == short


def test_no_marker_falls_back_to_tail():
    """Marker-less long logs degrade to the previous tail-truncation
    behavior so callers see the same shape they used to."""
    body = "filler line\n" * 10_000  # ~110_000 chars, no markers
    out = _extract_failure_window(body, max_chars=3000)
    assert out.startswith("...[truncated]\n")
    # Length should be roughly max_chars + the truncation prefix.
    assert 2900 <= len(out) <= 3050
    # Tail is still the end of the body (no failure to centre on).
    assert out.endswith(body[-100:])


def test_python_exception_in_actions_log_shape():
    """Reproduces PR #26460's bug: long log with a `Traceback (most
    recent call last):` and `Exception:` ~50KB before the end. Helper
    must surface those lines, not the post-job cleanup tail."""
    pre = "fluff line\n" * 1000  # ~11_000 chars of pretest noise
    diagnostic = (
        "2026-04-25T00:02:42.3064537Z Traceback (most recent call last):\n"
        '2026-04-25T00:02:42.3070589Z   File "tests/foo.py", line 125, in <module>\n'
        "2026-04-25T00:02:42.3071182Z     raise Exception(\n"
        "2026-04-25T00:02:42.3071401Z Exception: \n"
        "2026-04-25T00:02:42.3072200Z Keys not documented: "
        "{'LITELLM_FOO', 'LITELLM_BAR'}\n"
    )
    post = "git config --local --unset-all foo\n" * 5000  # post-job runner cleanup, ~150_000 chars
    log = pre + diagnostic + post

    out = _extract_failure_window(log, max_chars=3000)

    assert "Traceback (most recent call last):" in out, (
        "extracted window must contain the Traceback line — that's the "
        "highest-priority marker and the centre of the window"
    )
    assert "Exception:" in out
    assert "Keys not documented" in out
    assert "LITELLM_FOO" in out and "LITELLM_BAR" in out
    # Window is truncated front and back since it's interior to the log.
    assert out.startswith("...[truncated]\n")
    assert out.endswith("\n...[truncated]")


def test_traceback_outranks_actions_error_marker():
    """When both a Python `Traceback` and a GitHub Actions `##[error]` are
    present, the Traceback wins — it's the actual diagnostic, the
    `##[error]` is just the runner echoing the non-zero exit a few KB
    later. Without this priority order the helper would centre on
    `##[error]Process completed with exit code 1` and once again miss
    the failure body."""
    pre = "x" * 1000
    traceback = (
        "Traceback (most recent call last):\n"
        '  File "foo.py", line 1\n'
        "Exception: real cause\n"
    )
    middle = "y" * 50_000  # forces the two markers far apart
    actions_error = "##[error]Process completed with exit code 1.\n"
    post = "z" * 50_000
    log = pre + traceback + middle + actions_error + post

    out = _extract_failure_window(log, max_chars=3000)

    assert "Traceback (most recent call last):" in out
    assert "real cause" in out


def test_pytest_failed_summary_centred():
    """`FAILED tests/foo.py::test_bar` lines are the pytest summary
    shape — the helper centres on them when no Traceback is present
    (e.g. on assertion-style failures that don't raise)."""
    pre = "y" * 50_000
    summary = (
        "FAILED tests/foo.py::test_alpha - assert 1 == 2\n"
        "FAILED tests/foo.py::test_beta - assert 'x' in 'y'\n"
    )
    post = "git cleanup line\n" * 5000
    log = pre + summary + post

    out = _extract_failure_window(log, max_chars=3000)

    assert "FAILED tests/foo.py::test_beta" in out, (
        "helper must centre on the LAST FAILED line so the user sees the "
        "summary block, not just the first individual test failure"
    )
    assert "test_alpha" in out  # both should fit in a 3000-char window


def test_first_priority_marker_wins_over_lower():
    """If a `Traceback` exists, lower-priority `Error:` matches that
    happen later in the log are ignored — we don't want to centre on a
    runner-emitted `Error: process exited` line when the actual cause
    is a Python exception thousands of bytes earlier."""
    pre = "a" * 1000
    traceback_block = (
        "Traceback (most recent call last):\n"
        "  File 'main.py', line 1\n"
        "Exception: payload-shape mismatch\n"
    )
    middle = "b" * 60_000
    later_error = "Error: subprocess returned non-zero\n"
    post = "c" * 1000
    log = pre + traceback_block + middle + later_error + post

    out = _extract_failure_window(log, max_chars=3000)

    assert "payload-shape mismatch" in out
    assert "subprocess returned non-zero" not in out, (
        "lower-priority Error: marker must NOT win when a higher-priority "
        "Traceback is present elsewhere in the log"
    )


def test_window_size_respects_max_chars():
    """Extracted window must fit within `max_chars + truncation prefix
    + suffix` — guards against a regex bug expanding the slice. Uses
    `\\n` before the marker because the markers require a whitespace
    boundary (real CI logs always have one — either a timestamp prefix
    or a preceding newline)."""
    pre = "a" * 5000 + "\n"
    diagnostic = "Exception: boom\n"
    post = "b" * 5000
    log = pre + diagnostic + post

    out = _extract_failure_window(log, max_chars=2000)

    assert "Exception: boom" in out
    # max_chars + prefix("...[truncated]\n"=15) + suffix("\n...[truncated]"=15) = 2030
    assert len(out) <= 2050, (
        f"window unexpectedly large ({len(out)} chars) — the slice math may "
        "be off, the output should fit max_chars plus prefix/suffix markers"
    )


def test_marker_must_be_at_word_boundary():
    """Sanity check on `(?<!\\S)` lookbehind: a `Foo Exception:` line in
    user output should NOT match (Exception is mid-token from the
    regex's perspective). Real CI logs have whitespace/newlines before
    the diagnostic; this guards against false-positive centering on
    log noise that happens to contain the substring `Exception:`."""
    pre = "a" * 1000
    bogus = "myException: not a real one\n"  # no boundary before "Exception:"
    middle = "b" * 60_000
    real_traceback = (
        "\nTraceback (most recent call last):\n"
        "  File 'x.py', line 1\n"
        "Exception: real\n"
    )
    post = "c" * 1000
    log = pre + bogus + middle + real_traceback + post

    out = _extract_failure_window(log, max_chars=3000)
    assert "Exception: real" in out, (
        "helper should centre on the real Traceback, not the bogus "
        "mid-token 'myException:' which the lookbehind should reject"
    )
