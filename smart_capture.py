import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from capture import capture_screen
from movement import turn_right_45, wait_for_stabilization, TURN_DURATION


# Phase 9B: local quality-gated exploration. These values stay visible and
# editable so capture behavior can be tuned without touching reasoning code.
SMART_CAPTURE_ENABLED = True
SMART_CAPTURE_TARGET_GOOD = 5
SMART_CAPTURE_MIN_GOOD = 4
SMART_CAPTURE_MAX_ATTEMPTS = 20
SMART_CAPTURE_MAX_TOTAL_MOVEMENT_DURATION = 45.0
SMART_CAPTURE_MIN_EVIDENCE_SCORE = 6.0    # was 7.5 — road_markings+poles scenes no longer rejected
SMART_CAPTURE_HIGH_EVIDENCE_SCORE = 8.5   # unchanged — high-quality scenes still prioritised
SMART_CAPTURE_EMERGENCY_EVIDENCE_THRESHOLD = 5.0  # applied from attempt 9 when still 0 good shots
SMART_CAPTURE_ALLOW_MEDIUM_IF_NEEDED = True

# Backward-compatible names used by the Phase 9A integration.
MAX_PREVIEW_ATTEMPTS = SMART_CAPTURE_MAX_ATTEMPTS
TARGET_SCREENSHOTS = SMART_CAPTURE_TARGET_GOOD

MIN_BLUR_SCORE = 80.0
MIN_EDGE_DENSITY = 0.025
MIN_ENTROPY = 4.0
MAX_SKY_RATIO = 0.55
MAX_VEGETATION_RATIO = 0.72
MIN_LOWER_STRUCTURE_DENSITY = 0.022
DUPLICATE_THRESHOLD = 0.92          # same-direction duplicate threshold
DUPLICATE_THRESHOLD_NEW_DIRECTION = 0.97  # much harder to be a duplicate from a new direction
STUCK_DUPLICATE_LIMIT = 3

# High-value meta categories: if ANY of these appear in a screenshot that
# weren't seen in previously KEPT screenshots, it is NEVER a duplicate.
HIGH_VALUE_METAS = {
    "road_signs_candidate",
    "signposts_candidate",
    "street_names_or_business_text_candidate",
    "architecture_or_settlement",
}

# Maps movement action names to canonical direction labels.
_MOVEMENT_TO_DIRECTION: dict[str, str] = {
    "turn_left":          "left",
    "rotate_scan_left":   "left",
    "turn_right":         "right",
    "rotate_scan_right":  "right",
    "turn_around":        "backward",
    "turn_around_180":    "backward",
    "move_forward_short": "forward",
    "move_forward_long":  "forward",
    "move_forward_longer":"forward",
    # scan_turn_45 intentionally omitted — direction updated manually in run_360_scan
}
MIN_MEAN_BRIGHTNESS = 45.0
MAX_CLOSE_OBSTRUCTION_RATIO = 0.55
EARLY_COMPLETION_MIN_MOVEMENT_SECONDS = 10.0

# ── 360-degree scan phase ─────────────────────────────────────────────────────
# Runs once at the start of every smart-capture session before reactive
# exploration.  Rotates through SCAN_360_DIRECTIONS evenly-spaced headings and
# keeps the highest-evidence shots.
SCAN_360_ENABLED = True
SCAN_360_DIRECTIONS = 8          # 8 × 45° = full circle
SCAN_360_EVIDENCE_THRESHOLD = 5.0  # lower than main 6.0 — coverage over perfection
SCAN_360_MAX_KEEP = 3            # top N scan shots forwarded to reactive phase
SCAN_360_SKIP_REACTIVE_MIN_GOOD = 4  # skip reactive loop if scan already found this many

ANALYSIS_TOP_CROP_PERCENT = 0.08
ANALYSIS_BOTTOM_CROP_PERCENT = 0.12
ANALYSIS_LEFT_CROP_PERCENT = 0.04
ANALYSIS_RIGHT_CROP_PERCENT = 0.10

logger = logging.getLogger("smart_capture")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[SmartCapture] %(message)s"))
    logger.addHandler(handler)


@dataclass
class PreviewDecision:
    preview_path: str
    saved_path: str | None
    label: str
    attempt: int
    blur_score: float
    edge_density: float
    entropy: float
    sky_ratio: float
    vegetation_ratio: float
    lower_structure_density: float
    brightness: float
    close_obstruction_ratio: float
    duplicate_or_near_duplicate: bool
    duplicate_similarity: float
    visual_quality_score: float
    geo_evidence_score: float
    usefulness_score: float
    evidence_score: float
    evidence_categories: list[str]
    decision: str
    rejection_reason: str | None
    keep_reason: str | None
    recommended_next_action: str


def read_image(path: Path):
    """Read an image for local OpenCV analysis."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read preview screenshot: {path}")
    return image


def crop_analysis_roi(image):
    """Crop browser bars and HUD edges before preview scoring.

    The full screenshot remains saved for downstream OCR and Gemini. Local
    capture scoring uses the gameplay ROI so interface rectangles do not look
    like environmental signs or architecture.
    """
    height, width = image.shape[:2]
    top = int(height * ANALYSIS_TOP_CROP_PERCENT)
    bottom = int(height * (1.0 - ANALYSIS_BOTTOM_CROP_PERCENT))
    left = int(width * ANALYSIS_LEFT_CROP_PERCENT)
    right = int(width * (1.0 - ANALYSIS_RIGHT_CROP_PERCENT))
    return image[top:bottom, left:right]


def detect_blur(image) -> float:
    """Measure sharpness using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def detect_edge_density(image) -> float:
    """Measure structural image detail using Canny edges."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    return float(np.count_nonzero(edges) / edges.size)


def compute_entropy(image) -> float:
    """Measure visual information density."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    probabilities = histogram / max(histogram.sum(), 1)
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def estimate_sky_ratio(image) -> float:
    """Estimate sky dominance in the upper portion of the screenshot."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    upper_mask = np.zeros_like(hue, dtype=bool)
    upper_mask[: image.shape[0] // 2, :] = True
    blue_sky = (hue >= 85) & (hue <= 130) & (saturation > 35) & (value > 100)
    pale_sky = (saturation < 35) & (value > 180)
    sky = (blue_sky | pale_sky) & upper_mask
    return float(np.count_nonzero(sky) / sky.size)


def estimate_vegetation_ratio(image) -> float:
    """Estimate green vegetation dominance."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    vegetation = (hue >= 35) & (hue <= 90) & (saturation > 35) & (value > 45)
    return float(np.count_nonzero(vegetation) / vegetation.size)


def estimate_lower_structure_density(image) -> float:
    """Approximate road and infrastructure visibility below the horizon."""
    return detect_edge_density(image[image.shape[0] // 2 :, :])


def estimate_brightness(image) -> float:
    """Estimate whether the scene is too dark for reliable visual evidence."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def estimate_close_obstruction_ratio(image) -> float:
    """Approximate close walls, vehicles, or buildings that block context."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    height, width = edges.shape
    center = edges[int(height * 0.18) : int(height * 0.82), int(width * 0.18) : int(width * 0.82)]
    return float(np.count_nonzero(center) / max(center.size, 1))


def detect_line_features(image) -> tuple[int, int, int]:
    """Return horizontal-ish, vertical-ish, and angled Hough line counts."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=55, minLineLength=35, maxLineGap=12)
    horizontal = vertical = angled = 0

    for line in lines if lines is not None else []:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 18 or angle > 162:
            horizontal += 1
        elif 72 < angle < 108:
            vertical += 1
        else:
            angled += 1

    return horizontal, vertical, angled


def detect_sign_like_rectangles(image) -> int:
    """Count sign-like rectangles as a conservative text/sign candidate proxy."""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    count = 0

    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        contour_area = cv2.contourArea(contour)
        area_ratio = (box_width * box_height) / max(width * height, 1)
        aspect = box_width / max(box_height, 1)
        extent = contour_area / max(box_width * box_height, 1)
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        if (
            0.0002 <= area_ratio <= 0.04
            and 1.1 <= aspect <= 6.5
            and y < int(height * 0.82)
            and extent >= 0.45
            and 4 <= len(polygon) <= 6
        ):
            count += 1

    return min(count, 12)


def detect_evidence_categories(image) -> list[str]:
    """Detect local visual evidence categories without OCR or Gemini.

    Labels ending in ``_candidate`` are intentionally conservative: OpenCV can
    identify shapes and structure but cannot confirm semantic content.
    """
    categories = []
    height = image.shape[0]
    lower_half = image[height // 2 :, :]
    horizontal, vertical, angled = detect_line_features(image)
    lower_horizontal, _, lower_angled = detect_line_features(lower_half)
    sign_rectangles = detect_sign_like_rectangles(image)
    vegetation_ratio = estimate_vegetation_ratio(image)
    entropy = compute_entropy(image)

    if sign_rectangles >= 2:
        categories.extend(["road_signs_candidate", "signposts_candidate"])
    if sign_rectangles >= 5:
        categories.append("street_names_or_business_text_candidate")
    if lower_horizontal + lower_angled >= 4:
        categories.extend(["road_markings", "road_surface_markings"])
    if lower_angled >= 4 and horizontal >= 4:
        categories.append("intersection_candidate")
    if vertical >= 5:
        categories.append("utility_poles_candidate")
    if horizontal >= 6 and vertical >= 6:
        categories.append("architecture_or_settlement")
    if vegetation_ratio > 0.18 and entropy >= 5.5:
        categories.append("distinctive_vegetation")
    if entropy >= 7.0 and angled >= 5:
        categories.append("distinctive_terrain_or_context")

    return categories


def compute_visual_fingerprint(image) -> np.ndarray:
    """Build a small grayscale fingerprint for near-duplicate checks."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)


def calculate_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Calculate normalized visual similarity between fingerprints."""
    difference = cv2.absdiff(left, right)
    return max(0.0, min(1.0, 1.0 - float(np.mean(difference)) / 255.0))


def find_duplicate_similarity(fingerprint: np.ndarray, fingerprints: list[np.ndarray]) -> float:
    """Return strongest similarity against prior attempted previews."""
    if not fingerprints:
        return 0.0
    return max(calculate_similarity(fingerprint, item) for item in fingerprints)


def calculate_scores(
    *,
    blur_score: float,
    edge_density: float,
    entropy: float,
    sky_ratio: float,
    vegetation_ratio: float,
    lower_structure_density: float,
    brightness: float,
    duplicate_similarity: float,
    evidence_categories: list[str],
    effective_duplicate_threshold: float = DUPLICATE_THRESHOLD,
) -> tuple[float, float]:
    """Calculate separate visual usefulness and geographic evidence scores."""
    usefulness = 0.0
    usefulness += min(2.0, blur_score / 160.0 * 2.0)
    usefulness += min(2.0, edge_density / 0.08 * 2.0)
    usefulness += min(2.0, entropy / 8.0 * 2.0)
    usefulness += min(2.0, lower_structure_density / 0.08 * 2.0)
    usefulness += max(0.0, 1.0 - sky_ratio / max(MAX_SKY_RATIO, 0.01))
    usefulness += max(0.0, 1.0 - vegetation_ratio / max(MAX_VEGETATION_RATIO, 0.01))

    category_weights = {
        # Weights raised so scenes with road markings + poles reliably
        # clear the 6.0 minimum evidence threshold without needing signs.
        "road_signs_candidate": 2.5,
        "signposts_candidate": 3.0,
        "street_names_or_business_text_candidate": 3.0,
        "road_markings": 2.5,
        "road_surface_markings": 2.0,
        "intersection_candidate": 1.5,
        "utility_poles_candidate": 1.5,
        "architecture_or_settlement": 1.0,
        "distinctive_vegetation": 1.5,
        "distinctive_terrain_or_context": 2.0,
    }
    # Visual quality is not geographic evidence. A clear road/tree scene gets a
    # small baseline, while specific metas provide most of the geo score.
    evidence = min(1.5, usefulness * 0.15)
    evidence += sum(category_weights.get(category, 0.0) for category in evidence_categories)
    if (
        "road_signs_candidate" in evidence_categories
        and len(evidence_categories) >= 6
    ):
        evidence += 1.0

    if duplicate_similarity >= effective_duplicate_threshold:
        usefulness -= 3.0
        evidence -= 3.5
    if brightness < MIN_MEAN_BRIGHTNESS:
        evidence -= 1.5
    if vegetation_ratio > MAX_VEGETATION_RATIO and lower_structure_density < MIN_LOWER_STRUCTURE_DENSITY:
        evidence -= 2.0

    return (
        round(max(0.0, min(10.0, usefulness)), 3),
        round(max(0.0, min(10.0, evidence)), 3),
    )


def recommend_next_action(
    evidence_categories: list[str],
    evidence_score: float,
    duplicate: bool,
    attempt: int,
    good_count: int = 0,
) -> str:
    """Choose a movement that escalates when no good screenshots have been collected.

    Escalation ladder (only when good_count == 0):
      Attempts 1-3 : default strategy
      Attempts 4-5 : move_forward_longer (2.5 s forward move)
      Attempt  6   : turn_around_180 (look the opposite direction)
      Attempts 7-8 : default strategy from new direction
      Attempt  9+  : emergency threshold kicks in (handled in capture_preview);
                     continue with default strategy — threshold lowered by caller
    """
    categories = set(evidence_categories)

    # ── Escalation when stuck with zero good screenshots ─────────────────────
    if good_count == 0 and attempt >= 4:
        if attempt in {4, 5}:
            logger.info("[SmartCapture] Escalating to longer forward movement (attempt %d)", attempt)
            return "move_forward_longer"
        if attempt == 6:
            logger.info("[SmartCapture] Turning 180 to try opposite direction (attempt %d)", attempt)
            return "turn_around_180"
        # Attempts 7-8: resume default from new direction (fall through below)

    # ── Default strategy ──────────────────────────────────────────────────────
    if duplicate:
        return "turn_right" if attempt % 2 else "turn_left"
    if categories & {"road_signs_candidate", "street_names_or_business_text_candidate", "intersection_candidate"}:
        return "move_forward_short"
    if evidence_score < SMART_CAPTURE_MIN_EVIDENCE_SCORE:
        return "rotate_scan_left" if attempt % 2 else "rotate_scan_right"
    if "architecture_or_settlement" in categories:
        return "move_forward_short"
    return "move_forward_long"


def decide_preview(
    preview_path: Path,
    label: str,
    prior_fingerprints: list[np.ndarray],
    attempt: int = 1,
    good_count: int = 0,
    effective_min_score: float | None = None,
    effective_duplicate_threshold: float = DUPLICATE_THRESHOLD,
) -> tuple[PreviewDecision, np.ndarray]:
    """Analyze one preview and decide whether it counts as good evidence."""
    image = crop_analysis_roi(read_image(preview_path))
    fingerprint = compute_visual_fingerprint(image)
    blur_score = detect_blur(image)
    edge_density = detect_edge_density(image)
    entropy = compute_entropy(image)
    sky_ratio = estimate_sky_ratio(image)
    vegetation_ratio = estimate_vegetation_ratio(image)
    lower_structure_density = estimate_lower_structure_density(image)
    brightness = estimate_brightness(image)
    close_obstruction_ratio = estimate_close_obstruction_ratio(image)
    evidence_categories = detect_evidence_categories(image)
    duplicate_similarity = find_duplicate_similarity(fingerprint, prior_fingerprints)
    duplicate = duplicate_similarity >= effective_duplicate_threshold
    usefulness_score, evidence_score = calculate_scores(
        blur_score=blur_score,
        edge_density=edge_density,
        entropy=entropy,
        sky_ratio=sky_ratio,
        vegetation_ratio=vegetation_ratio,
        lower_structure_density=lower_structure_density,
        brightness=brightness,
        duplicate_similarity=duplicate_similarity,
        evidence_categories=evidence_categories,
        effective_duplicate_threshold=effective_duplicate_threshold,
    )

    reasons = []
    if blur_score < MIN_BLUR_SCORE:
        reasons.append("blurry motion frame")
    if duplicate:
        reasons.append("duplicate or near-duplicate scene")
    if sky_ratio > MAX_SKY_RATIO:
        reasons.append("mostly sky")
    if brightness < MIN_MEAN_BRIGHTNESS:
        reasons.append("too dark")
    if close_obstruction_ratio > MAX_CLOSE_OBSTRUCTION_RATIO:
        reasons.append("too close to an obstruction with limited context")
    if edge_density < MIN_EDGE_DENSITY or entropy < MIN_ENTROPY:
        reasons.append("low visual information")
    if vegetation_ratio > MAX_VEGETATION_RATIO and lower_structure_density < MIN_LOWER_STRUCTURE_DENSITY:
        reasons.append("vegetation-heavy with little infrastructure")
    min_score = effective_min_score if effective_min_score is not None else SMART_CAPTURE_MIN_EVIDENCE_SCORE

    if not evidence_categories:
        reasons.append("clear scene but no strong geographic evidence")
    if evidence_score < min_score:
        reasons.append(f"evidence score below {min_score:.1f}")

    keep = not reasons
    keep_reason = None
    rejection_reason = None
    if keep:
        evidence_level = (
            "high evidence"
            if evidence_score >= SMART_CAPTURE_HIGH_EVIDENCE_SCORE
            else "good evidence"
        )
        keep_reason = evidence_level + " scene with " + ", ".join(evidence_categories)
    else:
        rejection_reason = " / ".join(dict.fromkeys(reasons))

    next_action = recommend_next_action(evidence_categories, evidence_score, duplicate, attempt, good_count)
    return PreviewDecision(
        preview_path=str(preview_path),
        saved_path=None,
        label=label,
        attempt=attempt,
        blur_score=round(blur_score, 3),
        edge_density=round(edge_density, 4),
        entropy=round(entropy, 3),
        sky_ratio=round(sky_ratio, 4),
        vegetation_ratio=round(vegetation_ratio, 4),
        lower_structure_density=round(lower_structure_density, 4),
        brightness=round(brightness, 3),
        close_obstruction_ratio=round(close_obstruction_ratio, 4),
        duplicate_or_near_duplicate=duplicate,
        duplicate_similarity=round(duplicate_similarity, 4),
        visual_quality_score=usefulness_score,
        geo_evidence_score=evidence_score,
        usefulness_score=usefulness_score,
        evidence_score=evidence_score,
        evidence_categories=evidence_categories,
        decision="keep" if keep else "reject",
        rejection_reason=rejection_reason,
        keep_reason=keep_reason,
        recommended_next_action=next_action,
    ), fingerprint


class SmartCaptureSession:
    """Manage quality-gated previews, selected captures, movement, and reports."""

    def __init__(self, run_folder: Path, analysis_dir: Path):
        self.run_folder = run_folder
        self.analysis_dir = analysis_dir
        self.preview_dir = run_folder / "temporary_previews"
        self.discarded_dir = run_folder / "discarded_previews"
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        self.discarded_dir.mkdir(parents=True, exist_ok=True)
        self.prior_fingerprints: list[np.ndarray] = []
        self.saved_paths: list[Path] = []
        self.decisions: list[PreviewDecision] = []
        self.movement_actions: list[dict] = []
        self.stop_reason: str | None = None
        self.fallback_used = False
        self.consecutive_duplicates = 0
        self.emergency_threshold_active = False
        # Direction tracking for duplicate detection
        self.current_direction: str = "forward"
        self.directions_captured: set[str] = set()
        # Union of evidence categories from all KEPT screenshots
        self.kept_evidence_categories: set[str] = set()
        # 360-scan result stored here after run_360_scan() completes
        self._scan_360_result: dict = {}

    def capture_preview(
        self,
        label: str,
        attempt: int | None = None,
        min_score_override: float | None = None,
    ) -> PreviewDecision:
        """Capture, score, and retain only genuinely high-evidence previews.

        min_score_override: if set, use this evidence threshold instead of the
        session default.  Used by run_360_scan() to apply a lower scan threshold.
        """
        attempt = attempt or len(self.decisions) + 1
        good_count = len(self.saved_paths)

        # Activate emergency threshold from attempt 9 when still 0 good shots.
        if attempt >= 9 and good_count == 0 and not self.emergency_threshold_active:
            self.emergency_threshold_active = True
            logger.info(
                "[SmartCapture] Emergency threshold applied: "
                "lowering geo evidence minimum to %.1f",
                SMART_CAPTURE_EMERGENCY_EVIDENCE_THRESHOLD,
            )
        if min_score_override is not None:
            effective_min_score = min_score_override
        elif self.emergency_threshold_active:
            effective_min_score = SMART_CAPTURE_EMERGENCY_EVIDENCE_THRESHOLD
        else:
            effective_min_score = None

        # Direction-aware duplicate threshold.
        # A view from a direction we haven't captured yet is much less likely
        # to be a true duplicate even if pixel similarity is high.
        is_new_direction = self.current_direction not in self.directions_captured
        if is_new_direction and self.saved_paths:
            effective_dup_threshold = DUPLICATE_THRESHOLD_NEW_DIRECTION
            logger.info(
                "[SmartCapture] Direction '%s' not yet captured — "
                "relaxing duplicate threshold to %.2f",
                self.current_direction,
                effective_dup_threshold,
            )
        else:
            effective_dup_threshold = DUPLICATE_THRESHOLD

        logger.info("Attempt %s/%s", attempt, SMART_CAPTURE_MAX_ATTEMPTS)
        preview_path = capture_screen(self.preview_dir, f"preview_{label}")
        logger.info("Preview captured: %s", preview_path)
        decision, fingerprint = decide_preview(
            preview_path,
            label,
            self.prior_fingerprints,
            attempt,
            good_count=good_count,
            effective_min_score=effective_min_score,
            effective_duplicate_threshold=effective_dup_threshold,
        )

        # High-value meta override: if this screenshot contains a high-value
        # meta category that hasn't appeared in any previously KEPT screenshot,
        # never reject it as a duplicate — it contains new information.
        if decision.decision == "reject" and "duplicate" in (decision.rejection_reason or ""):
            current_metas = set(decision.evidence_categories)
            new_high_value = (HIGH_VALUE_METAS & current_metas) - self.kept_evidence_categories
            if new_high_value:
                logger.info(
                    "[SmartCapture] New high-value metas %s not seen before — "
                    "overriding duplicate rejection",
                    sorted(new_high_value),
                )
                # Remove the duplicate reason and re-evaluate keep status
                decision.rejection_reason = None
                decision.decision = "keep"
                decision.keep_reason = (
                    f"new high-value metas {sorted(new_high_value)} "
                    "override duplicate check"
                )
        self.prior_fingerprints.append(fingerprint)
        logger.info("Visual quality score: %.2f", decision.visual_quality_score)
        logger.info("Geo evidence score: %.2f", decision.geo_evidence_score)
        logger.info("Specific metas: %s", ", ".join(decision.evidence_categories) or "none")

        if decision.decision == "keep":
            saved_path = self.run_folder / preview_path.name.replace("preview_", "", 1)
            shutil.move(str(preview_path), saved_path)
            decision.saved_path = str(saved_path)
            self.saved_paths.append(saved_path)
            self.consecutive_duplicates = 0
            # Track direction and evidence categories for future duplicate checks
            self.directions_captured.add(self.current_direction)
            self.kept_evidence_categories.update(decision.evidence_categories)
            logger.info("Keeping screenshot %s/%s: %s", len(self.saved_paths), SMART_CAPTURE_TARGET_GOOD, decision.keep_reason)
        else:
            discarded_path = self.discarded_dir / preview_path.name
            shutil.move(str(preview_path), discarded_path)
            decision.preview_path = str(discarded_path)
            self.consecutive_duplicates = self.consecutive_duplicates + 1 if decision.duplicate_or_near_duplicate else 0
            logger.info("Rejecting screenshot: %s", decision.rejection_reason)

        logger.info(
            "Good screenshots collected: %s/%s",
            len(self.saved_paths),
            SMART_CAPTURE_TARGET_GOOD,
        )
        logger.info("Next action: %s", decision.recommended_next_action)
        self.decisions.append(decision)
        self.save_report()
        return decision

    def run_360_scan(self) -> dict:
        """Perform a full 360° scan before reactive exploration.

        Rotates through SCAN_360_DIRECTIONS headings spaced 360/SCAN_360_DIRECTIONS
        degrees apart, captures one screenshot per heading, then returns to the
        heading with the highest evidence score.

        Returns a summary dict that main.py uses to update movement_seconds and
        decide whether to skip the reactive exploration loop.
        """
        if not SCAN_360_ENABLED:
            result: dict = {
                "scan_360_completed": False,
                "scan_360_screenshots_kept": 0,
                "scan_360_best_direction_degrees": 0,
                "scan_360_best_evidence_score": 0.0,
                "reactive_exploration_needed": True,
                "skip_reactive": False,
                "movement_seconds_used": 0.0,
            }
            self._scan_360_result = result
            return result

        step_degrees = 360 // SCAN_360_DIRECTIONS
        scan_candidates: list[tuple[float, int, int]] = []  # (score, degrees, index)
        total_movement_seconds = 0.0

        logger.info(
            "Starting 360 degree scan (%d directions, %d degrees per step)",
            SCAN_360_DIRECTIONS,
            step_degrees,
        )

        for i in range(SCAN_360_DIRECTIONS):
            direction_degrees = i * step_degrees
            logger.info(
                "360 scan: direction %d/%d (%d degrees)",
                i + 1,
                SCAN_360_DIRECTIONS,
                direction_degrees,
            )
            # Set direction label before capture so duplicate detection uses it
            self.current_direction = f"scan_{direction_degrees}deg"

            # Always stabilize before capturing — turn_right_45 already sleeps
            # TURN_STABILIZATION_WAIT internally, but we add the full
            # POST_SCREENSHOT_WAIT here so Street View finishes loading the frame.
            wait_for_stabilization()

            label = f"scan360_{i + 1:02d}"
            attempt_num = len(self.decisions) + 1
            decision = self.capture_preview(
                label,
                attempt_num,
                min_score_override=SCAN_360_EVIDENCE_THRESHOLD,
            )
            scan_candidates.append((decision.evidence_score, direction_degrees, i))

            # Turn 45° right after each direction except the last
            if i < SCAN_360_DIRECTIONS - 1:
                turn_right_45()
                self.record_movement("scan_turn_45", TURN_DURATION)
                total_movement_seconds += TURN_DURATION

        # Determine best direction
        best_score, best_degrees, best_idx = max(scan_candidates, key=lambda x: x[0])
        logger.info(
            "Best direction found: %d degrees (evidence score: %.2f)",
            best_degrees,
            best_score,
        )

        # Return to best direction.
        # After SCAN_360_DIRECTIONS-1 right turns we are facing
        # (SCAN_360_DIRECTIONS-1)*step_degrees from the starting heading.
        current_heading = (SCAN_360_DIRECTIONS - 1) * step_degrees
        right_turns_needed = (
            (best_degrees - current_heading) % 360
        ) // step_degrees

        if right_turns_needed > 0:
            logger.info(
                "Returning to best direction: %d right turn(s) of %d degrees each",
                right_turns_needed,
                step_degrees,
            )
            for _ in range(right_turns_needed):
                turn_right_45()
                self.record_movement("scan_turn_45", TURN_DURATION)
                total_movement_seconds += TURN_DURATION
            wait_for_stabilization()

        # Count how many scan screenshots were kept
        scan_kept = sum(
            1
            for d in self.decisions
            if d.label.startswith("scan360_")
            and d.decision in {"keep", "fallback_keep"}
        )

        # Trim kept scan screenshots to top SCAN_360_MAX_KEEP by evidence score.
        # Screenshots already kept in self.saved_paths — remove lower-scoring excess.
        scan_kept_decisions = sorted(
            [
                d for d in self.decisions
                if d.label.startswith("scan360_") and d.decision in {"keep", "fallback_keep"}
            ],
            key=lambda d: d.evidence_score,
            reverse=True,
        )
        if len(scan_kept_decisions) > SCAN_360_MAX_KEEP:
            to_remove = scan_kept_decisions[SCAN_360_MAX_KEEP:]
            for drop in to_remove:
                drop_path = Path(drop.saved_path) if drop.saved_path else None
                if drop_path and drop_path.exists():
                    discarded = self.discarded_dir / drop_path.name
                    shutil.move(str(drop_path), discarded)
                    drop.saved_path = str(discarded)
                drop.decision = "reject"
                drop.rejection_reason = "scan_360_trimmed_to_top_3"
                drop.keep_reason = None
                if drop_path in self.saved_paths:
                    self.saved_paths.remove(drop_path)
            scan_kept = min(scan_kept, SCAN_360_MAX_KEEP)
            logger.info(
                "360 scan trimmed to top %d screenshots (removed %d lower-scoring shots)",
                SCAN_360_MAX_KEEP,
                len(to_remove),
            )

        skip_reactive = scan_kept >= SCAN_360_SKIP_REACTIVE_MIN_GOOD
        if skip_reactive:
            logger.info(
                "360 scan sufficient (%d good screenshots). Skipping reactive exploration.",
                scan_kept,
            )
        else:
            logger.info(
                "360 scan found %d screenshot(s). Continuing with reactive exploration.",
                scan_kept,
            )

        result = {
            "scan_360_completed": True,
            "scan_360_screenshots_kept": scan_kept,
            "scan_360_best_direction_degrees": best_degrees,
            "scan_360_best_evidence_score": round(best_score, 3),
            "reactive_exploration_needed": not skip_reactive,
            "skip_reactive": skip_reactive,
            "movement_seconds_used": round(total_movement_seconds, 3),
        }
        self._scan_360_result = result
        self.save_report()
        return result

    def record_movement(self, action: str, duration: float) -> None:
        """Record movement chosen by the exploration policy and update direction."""
        self.movement_actions.append({"action": action, "duration": round(duration, 3)})
        new_dir = _MOVEMENT_TO_DIRECTION.get(action)
        if new_dir:
            self.current_direction = new_dir
        self.save_report()

    def has_early_completion_evidence(self, movement_seconds: float) -> bool:
        """Allow early stop only after enough good and diverse evidence."""
        if len(self.saved_paths) < SMART_CAPTURE_MIN_GOOD:
            return False
        if movement_seconds < EARLY_COMPLETION_MIN_MOVEMENT_SECONDS:
            return False

        categories = {
            category
            for decision in self.decisions
            if decision.decision == "keep"
            for category in decision.evidence_categories
        }
        high_evidence_count = sum(
            1
            for decision in self.decisions
            if decision.decision == "keep"
            and decision.evidence_score >= SMART_CAPTURE_HIGH_EVIDENCE_SCORE
        )
        sign_evidence = bool(categories & {"road_signs_candidate", "street_names_or_business_text_candidate"})
        infrastructure = categories & {
            "road_markings",
            "intersection_candidate",
            "utility_poles_candidate",
            "architecture_or_settlement",
        }
        return (
            self.has_collection_diversity()
            and high_evidence_count >= 2
            and (sign_evidence or len(infrastructure) >= 3)
        )

    def has_collection_diversity(self) -> bool:
        """Require varied evidence or multiple camera turns before stopping."""
        kept = [decision for decision in self.decisions if decision.decision == "keep"]
        category_signatures = {
            tuple(sorted(decision.evidence_categories))
            for decision in kept
        }
        turn_count = sum(
            1
            for item in self.movement_actions
            if item["action"] in {"turn_left", "turn_right", "turn_around", "rotate_scan_left", "rotate_scan_right"}
        )
        return len(category_signatures) >= 2 or turn_count >= 2

    def should_stop(self, movement_seconds: float) -> tuple[bool, str | None]:
        """Apply target, early completion, stuck-scene, and movement safeguards."""
        if len(self.saved_paths) >= SMART_CAPTURE_TARGET_GOOD and self.has_collection_diversity():
            return True, "target_good_screenshots_reached"
        if self.has_early_completion_evidence(movement_seconds):
            return True, "early_completion_with_strong_evidence"
        if self.consecutive_duplicates >= STUCK_DUPLICATE_LIMIT:
            # Do not give up before attempt 10 if we have zero good screenshots —
            # let the escalation ladder (longer moves, 180 turn) run first.
            attempts_so_far = len(self.decisions)
            if attempts_so_far >= 10 or len(self.saved_paths) > 0:
                return True, "repeated_stuck_or_duplicate_scenes"
        if movement_seconds >= SMART_CAPTURE_MAX_TOTAL_MOVEMENT_DURATION:
            return True, "max_total_movement_duration_reached"
        if len(self.decisions) >= SMART_CAPTURE_MAX_ATTEMPTS:
            return True, "max_preview_attempts_reached"
        return False, None

    def select_best_available(self, minimum: int = SMART_CAPTURE_MIN_GOOD, maximum: int = SMART_CAPTURE_TARGET_GOOD) -> list[Path]:
        """Promote best diverse non-duplicate previews when target is not reached."""
        needed = max(0, minimum - len(self.saved_paths))
        if needed == 0:
            return list(self.saved_paths[:maximum])

        self.fallback_used = True
        selected_category_sets = [
            set(decision.evidence_categories)
            for decision in self.decisions
            if decision.saved_path and decision.decision == "keep"
        ]
        candidates = sorted(
            [
                decision
                for decision in self.decisions
                if decision.decision == "reject"
                and Path(decision.preview_path).exists()
                and not decision.duplicate_or_near_duplicate
                and (
                    SMART_CAPTURE_ALLOW_MEDIUM_IF_NEEDED
                    or decision.evidence_score >= SMART_CAPTURE_MIN_EVIDENCE_SCORE
                )
            ],
            key=lambda item: item.evidence_score,
            reverse=True,
        )

        for decision in candidates:
            if needed <= 0 or len(self.saved_paths) >= maximum:
                break
            categories = set(decision.evidence_categories)
            if selected_category_sets and all(categories == known for known in selected_category_sets):
                continue

            source = Path(decision.preview_path)
            destination = self.run_folder / source.name.replace("preview_", "fallback_", 1)
            shutil.copy2(source, destination)
            decision.saved_path = str(destination)
            decision.decision = "fallback_keep"
            decision.keep_reason = "best available diverse preview after safety stop"
            self.saved_paths.append(destination)
            selected_category_sets.append(categories)
            needed -= 1

        logger.info(
            "Safety limit reached. Saved best %s screenshots out of %s attempts.",
            len(self.saved_paths),
            len(self.decisions),
        )
        self.save_report()
        return list(self.saved_paths[:maximum])

    def finalize(self, stop_reason: str) -> list[Path]:
        """Finalize capture selection and save the Phase 9B report."""
        self.stop_reason = stop_reason
        logger.info("Stop reason: %s", stop_reason)
        if len(self.saved_paths) < SMART_CAPTURE_MIN_GOOD:
            self.select_best_available()
        if len(self.saved_paths) < 2:
            logger.warning(
                "[SmartCapture] WARNING: Only %d image(s) collected. "
                "Pipeline results will be unreliable. "
                "Recommend pressing G again from a better starting position "
                "or moving to a location with more visible features.",
                len(self.saved_paths),
            )
        self.save_report()
        return list(self.saved_paths[:SMART_CAPTURE_TARGET_GOOD])

    def save_report(self) -> None:
        """Save Phase 9B diagnostics in both analysis/ and the run folder."""
        payload = {
            "target_good_screenshots": SMART_CAPTURE_TARGET_GOOD,
            "min_good_screenshots": SMART_CAPTURE_MIN_GOOD,
            "max_attempts": SMART_CAPTURE_MAX_ATTEMPTS,
            "max_total_movement_duration": SMART_CAPTURE_MAX_TOTAL_MOVEMENT_DURATION,
            "min_evidence_score": SMART_CAPTURE_MIN_EVIDENCE_SCORE,
            "high_evidence_score": SMART_CAPTURE_HIGH_EVIDENCE_SCORE,
            "allow_medium_if_needed": SMART_CAPTURE_ALLOW_MEDIUM_IF_NEEDED,
            "scan_360_enabled": SCAN_360_ENABLED,
            "scan_360": self._scan_360_result if self._scan_360_result else {
                "scan_360_completed": False,
                "scan_360_screenshots_kept": 0,
                "scan_360_best_direction_degrees": None,
                "scan_360_best_evidence_score": 0.0,
                "reactive_exploration_needed": True,
            },
            "analysis_roi_crop": {
                "top": ANALYSIS_TOP_CROP_PERCENT,
                "bottom": ANALYSIS_BOTTOM_CROP_PERCENT,
                "left": ANALYSIS_LEFT_CROP_PERCENT,
                "right": ANALYSIS_RIGHT_CROP_PERCENT,
            },
            "attempts_used": len(self.decisions),
            "good_screenshots_collected": len([item for item in self.decisions if item.decision == "keep"]),
            "fallback_used": self.fallback_used,
            "stop_reason": self.stop_reason,
            "reliability_warning": (
                f"Only {len(self.saved_paths)} image(s) collected. Results unreliable."
                if len(self.saved_paths) < 2 else None
            ),
            "kept_screenshots": [
                asdict(item) for item in self.decisions if item.decision in {"keep", "fallback_keep"}
            ],
            "rejected_screenshots": [
                asdict(item) for item in self.decisions if item.decision == "reject"
            ],
            "movement_actions": self.movement_actions,
            "total_recorded_movement_duration": round(
                sum(item["duration"] for item in self.movement_actions),
                3,
            ),
        }
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        for path in [
            self.analysis_dir / "smart_capture_report.json",
            self.analysis_dir / "smart_capture_log.json",
            self.run_folder / "smart_capture_report.json",
        ]:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
