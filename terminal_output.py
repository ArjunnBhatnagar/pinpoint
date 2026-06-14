"""terminal_output.py — Clean per-stage terminal output for Pinpoint.

Set the environment variable PINPOINT_DEBUG=1 (or add DEBUG_LOGGING=True in
config) to see full verbose logs in addition to the clean summary lines.

Usage in pipeline stages:
    from terminal_output import stage_done, stage_skip, stage_warn, debug_print
    stage_done(3, 13, "First-Pass Reasoning", "Candidates: Germany, Austria")
    debug_print("full dump:", json.dumps(result))
"""

from __future__ import annotations

import os
import sys

# ── Debug mode ────────────────────────────────────────────────────────────────
# Enabled when the environment variable PINPOINT_DEBUG is set to a truthy value
# ("1", "true", "yes") or when DEBUG_LOGGING is True in this file.
DEBUG_LOGGING: bool = False  # flip to True for always-on debug

_ENV_DEBUG = os.environ.get("PINPOINT_DEBUG", "").strip().lower()
_DEBUG_MODE: bool = DEBUG_LOGGING or (_ENV_DEBUG in {"1", "true", "yes", "on"})


def is_debug() -> bool:
    return _DEBUG_MODE


# ── Symbols ───────────────────────────────────────────────────────────────────
_DONE = "✓"
_SKIP = "—"
_WARN = "⚠"
_FAIL = "✗"

_STAGE_WIDTH = 22  # fixed width for stage name column


def _stage_label(step: int, total: int, name: str) -> str:
    step_tag = f"[{step}/{total}]"
    return f"{step_tag:<8} {name:<{_STAGE_WIDTH}}"


# ── Public API ────────────────────────────────────────────────────────────────

def section_header(title: str) -> None:
    """Print a top-level section header. Always visible."""
    width = 60
    print()
    print("=" * width)
    print(title)
    print("=" * width)


def run_header(run_id: str) -> None:
    """Print the run banner. Always visible."""
    width = 60
    print()
    print("=" * width)
    print(f"PINPOINT — Run {run_id}")
    print("=" * width)


def stage_done(step: int, total: int, name: str, summary: str) -> None:
    """Print a clean ✓ line for a completed stage. Always visible."""
    label = _stage_label(step, total, name)
    print(f"{label} {_DONE}  {summary}")


def stage_skip(step: int, total: int, name: str, reason: str) -> None:
    """Print a — line for a skipped stage. Always visible."""
    label = _stage_label(step, total, name)
    print(f"{label} {_SKIP}  {reason}")


def stage_warn(step: int, total: int, name: str, warning: str) -> None:
    """Print a ⚠ warning line. Always visible."""
    label = _stage_label(step, total, name)
    print(f"{label} {_WARN}  {warning}", file=sys.stderr)


def stage_error(step: int, total: int, name: str, error: str) -> None:
    """Print a ✗ error line. Always visible."""
    label = _stage_label(step, total, name)
    print(f"{label} {_FAIL}  {error}", file=sys.stderr)


def always_print(*args, **kwargs) -> None:
    """Print that is always shown regardless of debug mode."""
    print(*args, **kwargs)


def debug_print(*args, **kwargs) -> None:
    """Print only when debug mode is active."""
    if _DEBUG_MODE:
        print(*args, **kwargs)


def warn_always(message: str) -> None:
    """Standalone warning line (not tied to a stage). Always visible."""
    print(f"{_WARN}  {message}", file=sys.stderr)
