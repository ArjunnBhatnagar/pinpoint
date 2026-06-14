import time

import keyboard
import pyautogui


# Active perception means the system takes bounded actions to gather more
# environmental evidence before reasoning. Continuous movement improves
# information gain because it samples farther-apart locations instead of several
# nearly identical nearby viewpoints.
#
# Redundant screenshots reduce multimodal effectiveness: if every image shows
# the same road segment, Gemini receives repeated evidence instead of diverse
# terrain, signage, infrastructure, and settlement context.
FORWARD_MOVE_DURATION = 3.0
BACKWARD_MOVE_DURATION = 2.0
MOVEMENT_INTERVAL = 0.45
MOVEMENT_COOLDOWN_SECONDS = 0.35
TURN_DURATION = 0.45

# Post-action stabilization waits — tune these if Street View loads slowly.
# MOVE_STABILIZATION_WAIT: seconds to wait after move_forward completes before
#   any further action (allows Street View to finish panning and re-render).
# TURN_STABILIZATION_WAIT: seconds to wait after a turn drag completes.
# POST_SCREENSHOT_WAIT: seconds to wait before capturing a screenshot; this is
#   also the value used by wait_for_stabilization().
MOVE_STABILIZATION_WAIT = 1.8
TURN_STABILIZATION_WAIT = 1.2
POST_SCREENSHOT_WAIT = 2.0
STABILIZATION_DELAY = POST_SCREENSHOT_WAIT

MAX_TOTAL_MOVEMENT_DURATION = 14.0

# The forward/back controls are usually near the lower center of Street View.
# Tune these percentages if clicks miss the movement arrows in your layout.
FORWARD_CLICK_X_PERCENT = 0.50
FORWARD_CLICK_Y_PERCENT = 0.58
BACKWARD_CLICK_X_PERCENT = 0.50
BACKWARD_CLICK_Y_PERCENT = 0.68

TURN_90_DRAG_PIXELS = 900
TURN_45_DRAG_PIXELS = TURN_90_DRAG_PIXELS // 2  # 450 pixels ≈ 45 degrees

# Backward-compatible aliases used by older code paths.
SCREENSHOT_STABILIZATION_DELAY = STABILIZATION_DELAY
MAX_MOVEMENTS_PER_RUN = 999

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1  # Minimum inter-action buffer for pyautogui calls

_total_movement_duration = 0.0
_last_movement_time = 0.0
_movement_duration_limit = MAX_TOTAL_MOVEMENT_DURATION


def reset_movement_counter() -> None:
    """Reset movement safeguards for a new exploration run."""
    global _last_movement_time, _total_movement_duration

    _total_movement_duration = 0.0
    _last_movement_time = 0.0


def set_movement_duration_limit(seconds: float) -> None:
    """Set a bounded per-run movement budget for the active exploration mode."""
    global _movement_duration_limit
    _movement_duration_limit = max(0.0, float(seconds))


def get_total_movement_duration() -> float:
    """Return registered forward/backward movement time for reporting."""
    return _total_movement_duration


def check_emergency_stop() -> None:
    """Stop immediately if ESC is pressed."""
    if keyboard.is_pressed("esc"):
        raise KeyboardInterrupt("Emergency stop: ESC was pressed.")


def safe_sleep(seconds: float) -> None:
    """Sleep in short steps so ESC can interrupt waits."""
    end_time = time.monotonic() + seconds

    while time.monotonic() < end_time:
        check_emergency_stop()
        time.sleep(0.05)


def _wait_for_cooldown() -> None:
    """Space out movement commands so Street View can respond cleanly."""
    elapsed = time.monotonic() - _last_movement_time
    remaining = MOVEMENT_COOLDOWN_SECONDS - elapsed

    if remaining > 0:
        safe_sleep(remaining)


def _register_movement_duration(seconds: float) -> None:
    """Enforce the per-run maximum movement duration."""
    global _total_movement_duration

    if _total_movement_duration + seconds > _movement_duration_limit:
        raise RuntimeError("Maximum movement duration reached for this run.")

    _total_movement_duration += seconds


def _get_screen_point(x_percent: float, y_percent: float) -> tuple[int, int]:
    """Convert a screen percentage into an absolute click point."""
    screen_width, screen_height = pyautogui.size()
    return int(screen_width * x_percent), int(screen_height * y_percent)


def _click_street_view_control(x_percent: float, y_percent: float) -> None:
    """Click a fixed Street View movement control location."""
    global _last_movement_time

    check_emergency_stop()
    _wait_for_cooldown()

    x, y = _get_screen_point(x_percent, y_percent)
    pyautogui.click(x, y)

    _last_movement_time = time.monotonic()


def _move_by_repeated_clicks(
    seconds: float,
    x_percent: float,
    y_percent: float,
) -> None:
    """Move continuously by repeatedly clicking a Street View movement control."""
    _register_movement_duration(seconds)

    end_time = time.monotonic() + seconds
    while time.monotonic() < end_time:
        _click_street_view_control(x_percent, y_percent)
        safe_sleep(MOVEMENT_INTERVAL)


def _drag_camera(distance_pixels: int) -> None:
    """Turn the camera with a fixed mouse drag."""
    global _last_movement_time

    check_emergency_stop()
    _wait_for_cooldown()

    screen_width, screen_height = pyautogui.size()
    start_x = screen_width // 2
    start_y = screen_height // 2

    pyautogui.moveTo(start_x, start_y, duration=0.1)
    pyautogui.dragRel(
        distance_pixels,
        0,
        duration=TURN_DURATION,
        button="left",
    )

    _last_movement_time = time.monotonic()
    safe_sleep(TURN_STABILIZATION_WAIT)


def wait_for_stabilization() -> None:
    """Wait for camera movement, scene transitions, and loading to settle."""
    safe_sleep(STABILIZATION_DELAY)


def move_forward_for_duration(seconds: float = FORWARD_MOVE_DURATION) -> None:
    """Move forward by repeatedly clicking the Street View forward control."""
    _move_by_repeated_clicks(
        seconds,
        FORWARD_CLICK_X_PERCENT,
        FORWARD_CLICK_Y_PERCENT,
    )
    safe_sleep(MOVE_STABILIZATION_WAIT)


def move_backward_for_duration(seconds: float = BACKWARD_MOVE_DURATION) -> None:
    """Move backward by repeatedly clicking the Street View backward control."""
    _move_by_repeated_clicks(
        seconds,
        BACKWARD_CLICK_X_PERCENT,
        BACKWARD_CLICK_Y_PERCENT,
    )


def turn_left_90() -> None:
    """Turn approximately 90 degrees left using a fixed mouse drag."""
    _drag_camera(-TURN_90_DRAG_PIXELS)


def turn_right_90() -> None:
    """Turn approximately 90 degrees right using a fixed mouse drag."""
    _drag_camera(TURN_90_DRAG_PIXELS)


def turn_right_45() -> None:
    """Turn approximately 45 degrees right using a fixed mouse drag."""
    _drag_camera(TURN_45_DRAG_PIXELS)


def turn_around() -> None:
    """Turn approximately 180 degrees left using fixed mouse drags."""
    turn_left_90()
    turn_left_90()
