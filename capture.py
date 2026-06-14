from datetime import datetime
from pathlib import Path

import mss
from PIL import Image


SCREENSHOTS_DIR = Path("screenshots")


def capture_screen(output_dir: Path = SCREENSHOTS_DIR, label: str = "screenshot") -> Path:
    """Capture the primary monitor and save it as a timestamped PNG file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    screenshot_path = output_dir / f"{label}_{timestamp}.png"

    with mss.mss() as screen_capture:
        monitor = screen_capture.monitors[1]
        screenshot = screen_capture.grab(monitor)

    image = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
    image.save(screenshot_path)

    return screenshot_path


def get_latest_screenshot() -> Path:
    """Return the newest screenshot saved by this project."""
    screenshots = list(SCREENSHOTS_DIR.glob("screenshot_*.png"))

    if not screenshots:
        raise FileNotFoundError(
            "No screenshots found. Press G in main.py to capture one first."
        )

    return max(screenshots, key=lambda path: path.stat().st_mtime)
