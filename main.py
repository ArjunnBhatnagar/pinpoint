import json
import sys
import threading
from datetime import datetime
from pathlib import Path
import shutil
import time

import keyboard

from terminal_output import (
    always_print,
    debug_print,
    run_header,
    stage_done,
    stage_error,
    stage_skip,
    stage_warn,
)
from adaptive_reexploration import run_adaptive_reexploration_decision
from database_health import run_health_check
from analyze import analyze_images
from final_decision import print_final_decision, run_final_decision
from sub_location_reasoner import get_available_countries, run_sub_location_reasoning
from confidence_verifier import run_confidence_verification
from country_reasoner import run_first_pass_reasoning
from capture import capture_screen
from grounded_reasoner import run_grounded_reasoning
from image_selector import run_image_selector
from movement import (
    FORWARD_MOVE_DURATION,
    MAX_TOTAL_MOVEMENT_DURATION,
    MOVEMENT_COOLDOWN_SECONDS,
    MOVEMENT_INTERVAL,
    STABILIZATION_DELAY,
    TURN_90_DRAG_PIXELS,
    TURN_DURATION,
    move_forward_for_duration,
    reset_movement_counter,
    set_movement_duration_limit,
    turn_around,
    turn_left_90,
    turn_right_90,
    wait_for_stabilization,
)
from ocr import (
    BOTTOM_CROP_PERCENT,
    LEFT_CROP_PERCENT,
    OCR_CONFIDENCE_THRESHOLD,
    OCR_GPU_ENABLED,
    OCR_MAX_IMAGE_WIDTH,
    RIGHT_CROP_PERCENT,
    TOP_CROP_PERCENT,
    extract_text_from_images,
    format_ocr_for_gemini,
    format_ocr_results,
    print_ocr_results,
)
from regional_consistency import run_regional_consistency
from retrieval_engine import build_retrieval_package
from smart_capture import (
    MAX_PREVIEW_ATTEMPTS,
    SCAN_360_ENABLED,
    SMART_CAPTURE_ENABLED,
    SMART_CAPTURE_HIGH_EVIDENCE_SCORE,
    SMART_CAPTURE_MAX_TOTAL_MOVEMENT_DURATION,
    SMART_CAPTURE_MIN_EVIDENCE_SCORE,
    SMART_CAPTURE_MIN_GOOD,
    TARGET_SCREENSHOTS,
    SmartCaptureSession,
)


RUNS_DIR = Path("runs")
ANALYSIS_DIR = Path("analysis")
SELECTED_DIR = Path("selected")
RUN_IN_PROGRESS = False


def _debug_should_show_ocr(ocr_results: list) -> bool:
    """Only show OCR output in debug mode."""
    from terminal_output import is_debug
    return is_debug()

# ── Pipeline feature flags ────────────────────────────────────────────────────
# Set OCR_ENABLED = False to skip EasyOCR entirely.
# Downstream modules handle empty OCR gracefully; no penalties are applied
# when this flag is False.
OCR_ENABLED = False


def create_run_folder() -> Path:
    """Create a timestamped folder for screenshots, logs, and responses."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_folder = RUNS_DIR / timestamp
    run_folder.mkdir(parents=True, exist_ok=True)
    return run_folder


def log_status(log_path: Path, message: str) -> None:
    """Write a status message to the run log; only print when debug mode is on."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_line = f"[{timestamp}] {message}"

    debug_print(log_line)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(log_line + "\n")


def pipeline_log(log_path: Path, message: str) -> None:
    """Write clean high-level pipeline logs (debug only)."""
    log_status(log_path, f"[Pipeline] {message}")


def capture_view(run_folder: Path, log_path: Path, label: str) -> Path:
    """Wait briefly, capture one view, and record where it was saved."""
    log_status(
        log_path,
        "Capture: waiting "
        f"{STABILIZATION_DELAY:.2f}s for {label} view.",
    )
    wait_for_stabilization()

    screenshot_path = capture_screen(run_folder, label)
    log_status(log_path, f"Capture: saved {label} view to {screenshot_path}.")

    return screenshot_path


def capture_fixed_route(
    run_folder: Path,
    log_path: Path,
    movement_history_path: Path,
) -> list[Path]:
    """Capture the original fixed route as a safe fallback."""
    screenshots = []

    log_status(log_path, "Capture fallback: using original fixed route.")
    screenshots.append(capture_view(run_folder, log_path, "initial"))

    run_movement_action(
        log_path,
        f"moving forward continuously for {FORWARD_MOVE_DURATION:.2f}s",
        lambda: move_forward_for_duration(FORWARD_MOVE_DURATION),
    )
    record_movement_history(
        movement_history_path,
        "move_forward_for_duration",
        FORWARD_MOVE_DURATION,
    )
    screenshots.append(capture_view(run_folder, log_path, "forward_duration"))

    run_movement_action(log_path, "turning approximately 90 degrees right", turn_right_90)
    record_movement_history(movement_history_path, "turn_right_90", TURN_DURATION)

    run_movement_action(
        log_path,
        f"moving forward continuously for {FORWARD_MOVE_DURATION:.2f}s after right turn",
        lambda: move_forward_for_duration(FORWARD_MOVE_DURATION),
    )
    record_movement_history(
        movement_history_path,
        "move_forward_for_duration_after_right_turn",
        FORWARD_MOVE_DURATION,
    )
    screenshots.append(capture_view(run_folder, log_path, "right_forward_duration"))

    run_movement_action(log_path, "turning approximately 180 degrees left", turn_around)
    record_movement_history(movement_history_path, "turn_around_left", TURN_DURATION * 2)

    run_movement_action(
        log_path,
        f"moving forward continuously for {FORWARD_MOVE_DURATION:.2f}s after turn around",
        lambda: move_forward_for_duration(FORWARD_MOVE_DURATION),
    )
    record_movement_history(
        movement_history_path,
        "move_forward_for_duration_after_turn_around",
        FORWARD_MOVE_DURATION,
    )
    screenshots.append(capture_view(run_folder, log_path, "opposite_forward_duration"))

    return screenshots


def collect_smart_screenshots(
    run_folder: Path,
    log_path: Path,
    movement_history_path: Path,
) -> list[Path]:
    """Collect high-evidence views with a bounded local exploration policy."""
    session = SmartCaptureSession(run_folder, ANALYSIS_DIR)
    movement_seconds = 0.0
    next_action = None
    stop_reason = "max_preview_attempts_reached"
    short_move_duration = min(1.5, FORWARD_MOVE_DURATION)
    longer_move_duration = 2.5  # used by escalation when stuck with 0 good shots

    policy_actions = {
        "move_forward_short": (
            f"moving forward briefly for {short_move_duration:.2f}s toward visible evidence",
            lambda: move_forward_for_duration(short_move_duration),
            short_move_duration,
        ),
        "move_forward_long": (
            f"moving forward for {FORWARD_MOVE_DURATION:.2f}s to search a new area",
            lambda: move_forward_for_duration(FORWARD_MOVE_DURATION),
            FORWARD_MOVE_DURATION,
        ),
        "move_forward_longer": (
            f"moving forward for {longer_move_duration:.2f}s (escalation — stuck with 0 good shots)",
            lambda: move_forward_for_duration(longer_move_duration),
            longer_move_duration,
        ),
        "turn_left": ("turning approximately 90 degrees left", turn_left_90, TURN_DURATION),
        "turn_right": ("turning approximately 90 degrees right", turn_right_90, TURN_DURATION),
        "turn_around": ("turning approximately 180 degrees", turn_around, TURN_DURATION * 2),
        "turn_around_180": ("turning 180 degrees to try opposite direction (escalation)", turn_around, TURN_DURATION * 2),
        "rotate_scan_left": ("rotating left to scan for stronger evidence", turn_left_90, TURN_DURATION),
        "rotate_scan_right": ("rotating right to scan for stronger evidence", turn_right_90, TURN_DURATION),
    }

    # ── 360-degree scan phase ─────────────────────────────────────────────────
    # Runs before reactive exploration to guarantee directional coverage.
    _skip_reactive = False
    if SCAN_360_ENABLED:
        pipeline_log(log_path, "360 degree scan started")
        scan_result = session.run_360_scan()
        movement_seconds += scan_result["movement_seconds_used"]
        record_movement_history(
            movement_history_path,
            "scan_360_complete",
            scan_result["movement_seconds_used"],
        )
        pipeline_log(
            log_path,
            f"360 degree scan completed: "
            f"{scan_result['scan_360_screenshots_kept']} kept, "
            f"best direction {scan_result['scan_360_best_direction_degrees']}° "
            f"(evidence {scan_result['scan_360_best_evidence_score']:.2f})",
        )
        if scan_result["skip_reactive"]:
            _skip_reactive = True
            stop_reason = "scan_360_sufficient"
            log_status(
                log_path,
                f"[SmartCapture] 360 scan sufficient "
                f"({scan_result['scan_360_screenshots_kept']} good screenshots). "
                "Skipping reactive exploration.",
            )

    # ── Reactive exploration phase ────────────────────────────────────────────
    if not _skip_reactive:
        for attempt in range(1, MAX_PREVIEW_ATTEMPTS + 1):
            if next_action:
                description, action, duration = policy_actions.get(
                    next_action,
                    policy_actions["rotate_scan_right"],
                )
                if movement_seconds + duration > SMART_CAPTURE_MAX_TOTAL_MOVEMENT_DURATION:
                    stop_reason = "max_total_movement_duration_reached"
                    break

                try:
                    run_movement_action(log_path, description, action)
                    record_movement_history(movement_history_path, next_action, duration)
                    session.record_movement(next_action, duration)
                    movement_seconds += duration
                except RuntimeError as error:
                    log_status(log_path, f"[SmartCapture] Movement limit reached: {error}")
                    stop_reason = "movement_module_safety_limit_reached"
                    break

            log_status(
                log_path,
                f"[SmartCapture] Waiting {STABILIZATION_DELAY:.2f}s before preview {attempt}.",
            )
            wait_for_stabilization()
            decision = session.capture_preview(f"attempt_{attempt:02d}", attempt)
            next_action = decision.recommended_next_action
            should_stop, reason = session.should_stop(movement_seconds)
            if should_stop:
                stop_reason = reason or "quality_gate_stop"
                break

    screenshots = session.finalize(stop_reason)
    log_status(
        log_path,
        f"[SmartCapture] Stop reason: {stop_reason}.",
    )
    log_status(
        log_path,
        f"[SmartCapture] Collected {len(screenshots)} screenshots "
        f"({min(len(session.saved_paths), SMART_CAPTURE_MIN_GOOD)}/{SMART_CAPTURE_MIN_GOOD} minimum).",
    )
    return screenshots


def run_movement_action(log_path: Path, description: str, action) -> None:
    """Run a movement action and log its timing."""
    start_time = time.monotonic()
    log_status(log_path, f"Movement: starting {description}.")

    action()

    elapsed = time.monotonic() - start_time
    log_status(log_path, f"Movement: finished {description} in {elapsed:.2f}s.")


def record_movement_history(history_path: Path, action: str, duration: float) -> None:
    """Save a plain movement history for debugging exploration paths."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    with history_path.open("a", encoding="utf-8") as history_file:
        history_file.write(f"[{timestamp}] {action} | duration={duration:.2f}s\n")


def copy_selected_screenshots(selected_images: list[Path], run_folder: Path) -> None:
    """Copy selected screenshots into selected/ for quick visual inspection."""
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)

    for index, image_path in enumerate(selected_images, start=1):
        destination = SELECTED_DIR / f"{run_folder.name}_{index:02d}_{image_path.name}"
        shutil.copy2(image_path, destination)


def serialize_ocr_results(ocr_results) -> list[dict]:
    """Convert OCR dataclasses into JSON-safe dictionaries."""
    serialized = []

    for result in ocr_results:
        serialized.append(
            {
                "image_name": result.image_name,
                "error": result.error,
                "processing_time": result.processing_time,
                "original_size": result.original_size,
                "cropped_size": result.cropped_size,
                "ocr_size": result.ocr_size,
                "cropped_image_path": (
                    str(result.cropped_image_path)
                    if result.cropped_image_path
                    else None
                ),
                "skipped_duplicate": result.skipped_duplicate,
                "duplicate_of": result.duplicate_of,
                "detections": [
                    {
                        "text": detection.text,
                        "confidence": detection.confidence,
                        "relevance": detection.relevance,
                    }
                    for detection in result.detections
                ],
                "removed_detections": [
                    {
                        "text": detection.text,
                        "confidence": detection.confidence,
                        "reason": detection.removal_reason,
                    }
                    for detection in result.removed_detections
                ],
            }
        )

    return serialized


def save_json(path: Path, payload: dict | list) -> None:
    """Save JSON using a consistent format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def get_selector_rejection_count() -> int:
    """Read image selector rejection count from analysis/image_ranking.json."""
    ranking_path = ANALYSIS_DIR / "image_ranking.json"

    if not ranking_path.exists():
        return 0

    try:
        payload = json.loads(ranking_path.read_text(encoding="utf-8"))
        screenshots = payload.get("all_screenshots", [])
        return sum(1 for item in screenshots if item.get("rejected"))
    except Exception:
        return 0


def _write_empty_ocr_artifacts(run_folder: Path) -> None:
    """Write placeholder OCR files so downstream modules never crash on missing files."""
    ocr_txt = run_folder / "ocr.txt"
    ocr_txt.write_text(
        "OCR Results\n"
        "GPU enabled setting: False\n"
        "OCR disabled by config.\n"
        "Filtered OCR text: none\n",
        encoding="utf-8",
    )
    # Empty results list — same schema as a real run with no detections.
    save_json(ANALYSIS_DIR / "ocr_results.json", [])
    # Status sentinel so confidence_verifier and adaptive can detect disabled OCR.
    save_json(
        ANALYSIS_DIR / "ocr_status.json",
        {"attempted": False, "reason": "OCR disabled by config (OCR_ENABLED = False)"},
    )


def run_exploration_analysis() -> None:
    """Capture a broader fixed route, then analyze all views together."""
    global RUN_IN_PROGRESS

    if RUN_IN_PROGRESS:
        print("A run is already in progress. Please wait.")
        return

    RUN_IN_PROGRESS = True
    run_folder = create_run_folder()
    log_path = run_folder / "run.log"
    movement_history_path = run_folder / "movement_history.txt"
    exploration_path = run_folder / "exploration_path.txt"
    pipeline_summary = {
        "run_folder": str(run_folder),
        "screenshots_captured": 0,
        "selected_image_count": 0,
        "rejected_image_count": 0,
        "ocr_runtime_seconds": None,
        "gemini_runtime_seconds": None,
        "first_pass_reasoning_runtime_seconds": None,
        "retrieval_runtime_seconds": None,
        "grounded_reasoning_runtime_seconds": None,
        "regional_consistency_runtime_seconds": None,
        "confidence_verification_runtime_seconds": None,
        "adaptive_decision_runtime_seconds": None,
        "final_decision_runtime_seconds": None,
        "sub_location_runtime_seconds": None,
        "selector_runtime_seconds": None,
        "fallback_used": False,
        "error": None,
    }

    # ── Stage numbering ───────────────────────────────────────────────────────
    # 1  Smart Capture       6  Grounded Comparison
    # 2  Image Selection     7  Regional Consistency
    # 3  OCR                 8  Confidence Check
    # 4  Environmental Scan  9  Adaptive Decision (internal)
    # 5  First-Pass          10 Sub-Location
    _T = 10  # total stages

    set_movement_duration_limit(
        SMART_CAPTURE_MAX_TOTAL_MOVEMENT_DURATION
        if SMART_CAPTURE_ENABLED
        else MAX_TOTAL_MOVEMENT_DURATION
    )
    reset_movement_counter()

    run_header(run_folder.name)

    log_status(log_path, f"Run started: {run_folder}")
    log_status(
        log_path,
        "Movement settings: "
        f"forward_move_duration={FORWARD_MOVE_DURATION:.2f}s, "
        f"movement_interval={MOVEMENT_INTERVAL:.2f}s, "
        f"cooldown={MOVEMENT_COOLDOWN_SECONDS:.2f}s, "
        f"stabilization={STABILIZATION_DELAY:.2f}s, "
        f"turn_drag={TURN_90_DRAG_PIXELS}px, "
        f"turn_duration={TURN_DURATION:.2f}s, "
        "max_total_movement_duration="
        f"{(SMART_CAPTURE_MAX_TOTAL_MOVEMENT_DURATION if SMART_CAPTURE_ENABLED else MAX_TOTAL_MOVEMENT_DURATION):.2f}s, "
        f"ocr_gpu_enabled={OCR_GPU_ENABLED}, "
        f"ocr_max_width={OCR_MAX_IMAGE_WIDTH}px, "
        f"ocr_threshold={OCR_CONFIDENCE_THRESHOLD:.2f}, "
        f"ocr_crop_top={TOP_CROP_PERCENT:.2f}, "
        f"ocr_crop_bottom={BOTTOM_CROP_PERCENT:.2f}, "
        f"ocr_crop_left={LEFT_CROP_PERCENT:.2f}, "
        f"ocr_crop_right={RIGHT_CROP_PERCENT:.2f}, "
        f"smart_capture_enabled={SMART_CAPTURE_ENABLED}, "
        f"smart_capture_target={TARGET_SCREENSHOTS}, "
        f"smart_capture_min_good={SMART_CAPTURE_MIN_GOOD}, "
        f"smart_capture_max_previews={MAX_PREVIEW_ATTEMPTS}, "
        f"smart_capture_min_evidence={SMART_CAPTURE_MIN_EVIDENCE_SCORE:.2f}, "
        f"smart_capture_high_evidence={SMART_CAPTURE_HIGH_EVIDENCE_SCORE:.2f}, "
        f"smart_capture_movement_budget={SMART_CAPTURE_MAX_TOTAL_MOVEMENT_DURATION:.2f}s.",
    )

    exploration_path.write_text(
        "Exploration path:\n"
        "1. Capture and score an initial temporary preview.\n"
        "2. Keep useful previews; discard low-information previews.\n"
        f"3. Continue bounded movement until {TARGET_SCREENSHOTS} useful screenshots "
        f"or {MAX_PREVIEW_ATTEMPTS} preview attempts.\n"
        "4. Promote the best available previews if strict filtering keeps too few.\n"
        "5. Continue OCR and Gemini reasoning.\n",
        encoding="utf-8",
    )

    try:
        # ── Stage 1: Smart Capture ────────────────────────────────────────────
        if SMART_CAPTURE_ENABLED:
            try:
                pipeline_log(log_path, "Smart preview capture started")
                screenshots = collect_smart_screenshots(
                    run_folder,
                    log_path,
                    movement_history_path,
                )
                if not screenshots:
                    raise RuntimeError("Smart capture did not produce screenshots.")
                pipeline_log(log_path, "Smart preview capture completed")
                stage_done(1, _T, "Smart Capture", f"{len(screenshots)} screenshots collected")
            except Exception as error:
                pipeline_log(log_path, f"Smart capture failed safely: {error}")
                pipeline_summary["fallback_used"] = True
                reset_movement_counter()
                screenshots = capture_fixed_route(
                    run_folder,
                    log_path,
                    movement_history_path,
                )
                stage_warn(1, _T, "Smart Capture",
                           f"fallback to fixed route — {len(screenshots)} screenshots")
        else:
            screenshots = capture_fixed_route(
                run_folder,
                log_path,
                movement_history_path,
            )
            stage_done(1, _T, "Smart Capture", f"{len(screenshots)} screenshots (fixed route)")

        log_status(log_path, f"Movement: history saved to {movement_history_path}.")
        log_status(log_path, f"Movement: exploration path saved to {exploration_path}.")

        pipeline_summary["screenshots_captured"] = len(screenshots)
        pipeline_log(log_path, f"Captured {len(screenshots)} screenshots")

        # ── Stage 2: Image Selection ──────────────────────────────────────────
        pipeline_log(log_path, "Running image selector...")
        selector_start = time.monotonic()
        try:
            selected_screenshots = run_image_selector(
                screenshots,
                analysis_dir=ANALYSIS_DIR,
            )
        except Exception as error:
            pipeline_log(log_path, f"Image selector failed: {error}")
            selected_screenshots = screenshots
            pipeline_summary["fallback_used"] = True

        selector_runtime = time.monotonic() - selector_start
        pipeline_summary["selector_runtime_seconds"] = round(selector_runtime, 3)

        if not selected_screenshots:
            pipeline_log(log_path, "Image selector returned no images; using all screenshots")
            selected_screenshots = screenshots
            pipeline_summary["fallback_used"] = True

        selected_screenshots = [Path(path) for path in selected_screenshots]
        pipeline_summary["selected_image_count"] = len(selected_screenshots)
        rejected = get_selector_rejection_count()
        pipeline_summary["rejected_image_count"] = rejected

        copy_selected_screenshots(selected_screenshots, run_folder)
        pipeline_log(log_path, f"Selected {len(selected_screenshots)} screenshots")
        stage_done(2, _T, "Image Selection",
                   f"Best {len(selected_screenshots)} selected"
                   + (f", {rejected} rejected" if rejected else ""))

        # ── Stage 3: OCR ──────────────────────────────────────────────────────
        if OCR_ENABLED:
            pipeline_log(log_path, "OCR running on selected screenshots")
            ocr_start = time.monotonic()
            ocr_results = extract_text_from_images(selected_screenshots, debug_dir=run_folder)
            ocr_runtime = time.monotonic() - ocr_start
            pipeline_summary["ocr_runtime_seconds"] = round(ocr_runtime, 3)

            formatted_ocr = format_ocr_results(ocr_results)
            ocr_path = run_folder / "ocr.txt"
            ocr_path.write_text(formatted_ocr, encoding="utf-8")
            save_json(ANALYSIS_DIR / "ocr_results.json", serialize_ocr_results(ocr_results))
            save_json(ANALYSIS_DIR / "ocr_status.json", {"attempted": True})

            log_status(log_path, f"OCR: results saved to {ocr_path}.")
            pipeline_log(log_path, f"OCR completed in {ocr_runtime:.2f} seconds")
            if _debug_should_show_ocr(ocr_results):
                print_ocr_results(formatted_ocr)

            total_detections = sum(len(r.detections) for r in ocr_results)
            stage_done(3, _T, "OCR",
                       f"{total_detections} text detection{'s' if total_detections != 1 else ''} "
                       f"in {ocr_runtime:.1f}s")
        else:
            pipeline_log(log_path, "[Pipeline] OCR disabled by config. Skipping.")
            _write_empty_ocr_artifacts(run_folder)
            ocr_results = []
            formatted_ocr = "OCR disabled by config."
            log_status(log_path, "OCR: skipped (OCR_ENABLED = False).")
            stage_skip(3, _T, "OCR", "disabled by config")

        # ── Stage 4: Environmental Analysis ──────────────────────────────────
        log_status(
            log_path,
            "Analysis: sending "
            f"{len(selected_screenshots)} selected screenshots and OCR text to Gemini.",
        )
        pipeline_log(log_path, "Gemini reasoning started")
        gemini_start = time.monotonic()
        analysis = analyze_images(
            selected_screenshots,
            format_ocr_for_gemini(ocr_results),
        )
        gemini_runtime = time.monotonic() - gemini_start
        pipeline_summary["gemini_runtime_seconds"] = round(gemini_runtime, 3)
        pipeline_log(log_path, "Gemini reasoning completed")

        response_path = run_folder / "gemini_response.txt"
        response_path.write_text(analysis, encoding="utf-8")
        summary_path = run_folder / "reasoning_summary.txt"
        summary_path.write_text(analysis, encoding="utf-8")
        log_status(log_path, f"Analysis: response saved to {response_path}.")
        log_status(log_path, f"Analysis: summary saved to {summary_path}.")

        # Show full Gemini text only in debug mode
        if True:  # always gate this — it's very long
            debug_print()
            debug_print("=" * 60)
            debug_print("Gemini Environmental Comparison Analysis")
            debug_print("=" * 60)
            debug_print(analysis)
            debug_print("=" * 60)
            debug_print()

        stage_done(4, _T, "Environmental Scan", f"Gemini analysis complete ({gemini_runtime:.1f}s)")

        # ── Stage 5: First-Pass Reasoning ─────────────────────────────────────
        pipeline_log(log_path, "First-pass structured country reasoning started")
        first_pass_start = time.monotonic()
        first_pass = {}
        try:
            first_pass = run_first_pass_reasoning(ANALYSIS_DIR)
            first_pass_runtime = time.monotonic() - first_pass_start
            pipeline_summary["first_pass_reasoning_runtime_seconds"] = round(first_pass_runtime, 3)
            pipeline_log(log_path, "First-pass structured country reasoning completed")
            log_status(log_path,
                       "Reasoner: first_pass.json saved with "
                       f"{len(first_pass.get('top_candidates', []))} candidates.")
            _candidates = [c.get("country", "") for c in first_pass.get("top_candidates", [])]
            stage_done(5, _T, "First-Pass Reasoning",
                       "Candidates: " + ", ".join(_candidates) if _candidates else "no candidates")
        except Exception as error:
            pipeline_log(log_path, f"First-pass reasoning failed safely: {error}")
            stage_error(5, _T, "First-Pass Reasoning", str(error))

        # ── Stage 6: Reference Retrieval ──────────────────────────────────────
        pipeline_log(log_path, "Reference retrieval package generation started")
        retrieval_start = time.monotonic()
        retrieval_package = {}
        try:
            retrieval_package = build_retrieval_package(
                first_pass_path=ANALYSIS_DIR / "first_pass.json",
                output_path=ANALYSIS_DIR / "retrieval_package.json",
            )
            retrieval_runtime = time.monotonic() - retrieval_start
            pipeline_summary["retrieval_runtime_seconds"] = round(retrieval_runtime, 3)
            pipeline_log(log_path, "Reference retrieval package generation completed")
            _ret_sum = retrieval_package.get("summary", {})
            _top_country = (first_pass.get("top_candidates") or [{}])[0].get("country", "")
            _n_img = _ret_sum.get("total_images", 0)
            _n_txt = _ret_sum.get("total_texts", 0)
            log_status(log_path,
                       f"Retrieval: retrieval_package.json saved with {_n_img} images "
                       f"and {_n_txt} texts.")
            stage_done(6, _T, "Reference Retrieval",
                       f"{_n_img} images, {_n_txt} texts"
                       + (f" for {_top_country}" if _top_country else ""))
        except Exception as error:
            pipeline_log(log_path, f"Reference retrieval failed safely: {error}")
            stage_error(6, _T, "Reference Retrieval", str(error))

        # ── Stage 7: Grounded Comparison ──────────────────────────────────────
        pipeline_log(log_path, "Grounded comparison reasoning started")
        grounded_start = time.monotonic()
        grounded_result = {}
        try:
            grounded_result = run_grounded_reasoning(ANALYSIS_DIR)
            grounded_runtime = time.monotonic() - grounded_start
            pipeline_summary["grounded_reasoning_runtime_seconds"] = round(grounded_runtime, 3)
            pipeline_log(log_path, "Grounded comparison reasoning completed")
            log_status(log_path,
                       f"Grounded: grounded_result.json saved with "
                       f"final_country={grounded_result.get('final_country')}.")
            _gr_country = grounded_result.get("final_country", "unknown")
            _gr_conf = grounded_result.get("confidence", 0.0)
            stage_done(7, _T, "Grounded Comparison",
                       f"{_gr_country} ({int(_gr_conf * 100)}%)")
        except Exception as error:
            pipeline_log(log_path, f"Grounded comparison failed safely: {error}")
            stage_error(7, _T, "Grounded Comparison", str(error))

        # ── Stage 8: Regional Consistency ─────────────────────────────────────
        pipeline_log(log_path, "Regional consistency enforcement started")
        regional_start = time.monotonic()
        regional_result = {}
        try:
            regional_result = run_regional_consistency(ANALYSIS_DIR)
            regional_runtime = time.monotonic() - regional_start
            pipeline_summary["regional_consistency_runtime_seconds"] = round(regional_runtime, 3)
            pipeline_log(log_path, "Regional consistency enforcement completed")
            log_status(log_path,
                       "RegionalConsistency: regional_consistency.json saved with "
                       f"selected_country={regional_result.get('selected_country')}.")
            _override = regional_result.get("override_triggered", False)
            _reg_country = regional_result.get("selected_country", "unknown")
            _reg_summary = (
                f"override → {_reg_country}" if _override
                else f"no override — {_reg_country} consistent"
            )
            stage_done(8, _T, "Regional Consistency", _reg_summary)
        except Exception as error:
            pipeline_log(log_path, f"Regional consistency failed safely: {error}")
            stage_error(8, _T, "Regional Consistency", str(error))

        # ── Stage 9: Confidence Check ─────────────────────────────────────────
        pipeline_log(log_path, "Confidence verification started")
        confidence_start = time.monotonic()
        confidence_report = None
        try:
            confidence_report = run_confidence_verification(ANALYSIS_DIR)
            confidence_runtime = time.monotonic() - confidence_start
            pipeline_summary["confidence_verification_runtime_seconds"] = round(
                confidence_runtime, 3)
            pipeline_log(log_path, "Confidence verification completed")
            log_status(log_path,
                       "Confidence: confidence_report.json saved with "
                       f"trust_level={confidence_report.get('trust_level')} and "
                       f"recommendation={confidence_report.get('recommendation')}.")
            _conf_pct = int(confidence_report.get("verified_confidence", 0.0) * 100)
            _trust = confidence_report.get("trust_level", "")
            stage_done(9, _T, "Confidence Check", f"Verified: {_conf_pct}% ({_trust})")
        except Exception as error:
            confidence_report = None
            pipeline_log(log_path, f"Confidence verification failed safely: {error}")
            stage_error(9, _T, "Confidence Check", str(error))

        # ── Adaptive Decision (internal, no user-facing stage line) ───────────
        pipeline_log(log_path, "Adaptive re-exploration decision started")
        adaptive_start = time.monotonic()
        try:
            adaptive_decision = run_adaptive_reexploration_decision(
                ANALYSIS_DIR,
                run_folder,
            )
            adaptive_runtime = time.monotonic() - adaptive_start
            pipeline_summary["adaptive_decision_runtime_seconds"] = round(adaptive_runtime, 3)
            pipeline_log(log_path, "Adaptive re-exploration decision completed")
            log_status(log_path,
                       "Adaptive: adaptive_reexploration_decision.json saved with "
                       f"decision={adaptive_decision.get('decision')}.")
        except Exception as error:
            adaptive_decision = {
                "decision": "manual_review",
                "recommended_strategy": None,
            }
            pipeline_log(log_path, f"Adaptive decision failed safely: {error}")

        # ── Final Decision (internal) ─────────────────────────────────────────
        pipeline_log(log_path, "Final decision layer started")
        final_decision_start = time.monotonic()
        final_decision_result = None
        try:
            final_decision_result = run_final_decision(ANALYSIS_DIR)
            final_decision_runtime = time.monotonic() - final_decision_start
            pipeline_summary["final_decision_runtime_seconds"] = round(final_decision_runtime, 3)
            pipeline_log(log_path, "Final decision layer completed")
            log_status(log_path,
                       "FinalDecision: final_decision.json saved with "
                       f"final_country={final_decision_result.get('final_country')} "
                       f"({final_decision_result.get('confidence_label')}, "
                       f"{int(final_decision_result.get('confidence', 0) * 100)}%).")
        except Exception as error:
            final_decision_result = None
            pipeline_log(log_path, f"Final decision failed safely: {error}")

        # ── Stage 10: Sub-Location ────────────────────────────────────────────
        pipeline_log(log_path, "Sub-location reasoning started")
        sub_location_start = time.monotonic()
        sub_location_result = None
        try:
            sub_location_result = run_sub_location_reasoning(ANALYSIS_DIR)
            sub_location_runtime = time.monotonic() - sub_location_start
            pipeline_summary["sub_location_runtime_seconds"] = round(sub_location_runtime, 3)
            pipeline_log(log_path, "Sub-location reasoning completed")
            log_status(log_path,
                       "SubLocation: sub_location_result.json saved with "
                       f"final_state={sub_location_result.get('final_state')} "
                       f"broad_region={sub_location_result.get('broad_region')}.")
            _attempted = sub_location_result.get("sub_location_attempted", False)
            _kb_used = sub_location_result.get("knowledge_base_used", False)
            if not _attempted:
                stage_skip(10, _T, "Sub-Location", "confidence below threshold")
            elif not _kb_used:
                _fd_country = (final_decision_result or {}).get("final_country", "")
                stage_skip(10, _T, "Sub-Location",
                           f"no knowledge base for {_fd_country}")
            else:
                _broad = sub_location_result.get("broad_region", "")
                _state = sub_location_result.get("final_state")
                _s_conf = int((sub_location_result.get("state_confidence") or 0) * 100)
                if _state:
                    stage_done(10, _T, "Sub-Location",
                               f"{_state} ({_s_conf}%) — {_broad}")
                else:
                    stage_done(10, _T, "Sub-Location",
                               f"Region: {_broad} (state undetermined)")
        except Exception as error:
            pipeline_log(log_path, f"Sub-location reasoning failed safely: {error}")
            stage_error(10, _T, "Sub-Location", str(error))

        log_status(log_path, "Run completed.")
        save_json(ANALYSIS_DIR / "pipeline_summary.json", pipeline_summary)
    except KeyboardInterrupt as error:
        pipeline_summary["error"] = str(error)
        save_json(ANALYSIS_DIR / "pipeline_summary.json", pipeline_summary)
        log_status(log_path, f"Emergency stop: {error}")
        return
    except Exception as error:
        pipeline_summary["error"] = str(error)
        save_json(ANALYSIS_DIR / "pipeline_summary.json", pipeline_summary)
        log_status(log_path, f"Run failed: {error}")
        return
    finally:
        RUN_IN_PROGRESS = False

    # Rate-limit warning — always visible.
    grounded_path = ANALYSIS_DIR / "grounded_result.json"
    if grounded_path.exists():
        try:
            _gr = json.loads(grounded_path.read_text(encoding="utf-8"))
            if _gr.get("rate_limited"):
                always_print()
                always_print("=" * 55)
                always_print("WARNING: Gemini rate limit hit during this run.")
                always_print("  Reference images were not used for comparison.")
                always_print("  Results are less reliable than normal.")
                always_print("  Try again in a few minutes.")
                always_print("=" * 55)
                always_print()
        except Exception:
            pass

    if final_decision_result:
        print_final_decision(final_decision_result, sub_location=sub_location_result)
    elif confidence_report:
        always_print("Final country:", confidence_report.get("final_country"))
        always_print("Trust level:", confidence_report.get("trust_level"))

    # ── Post-round answer prompt (Part A) ─────────────────────────────────────
    _answer_result: list[str | None] = [None]

    def _read_answer() -> None:
        try:
            _answer_result[0] = input(
                "\n[Accuracy] Press Enter to skip, or type the correct answer: "
            )
        except (EOFError, OSError):
            _answer_result[0] = ""

    _answer_thread = threading.Thread(target=_read_answer, daemon=True)
    _answer_thread.start()
    _answer_thread.join(30)

    _answer = _answer_result[0]
    if _answer and _answer.strip():
        try:
            from accuracy_tracker import log_round_result
            record = log_round_result(_answer.strip(), run_folder, ANALYSIS_DIR)
            outcome = (
                "CORRECT"
                if record.get("country_correct")
                else f"WRONG (predicted: {record.get('predicted_country')})"
            )
            always_print(f"[Accuracy] Round logged — {outcome}")
        except Exception as _acc_err:
            always_print(f"[Accuracy] Logging failed (non-fatal): {_acc_err}")


def main() -> None:
    # ── --stats mode (Part C) ─────────────────────────────────────────────────
    if "--stats" in sys.argv:
        from accuracy_tracker import print_accuracy_stats
        print_accuracy_stats()
        sys.exit(0)

    # Run the reference database health check once at startup.
    # This produces analysis/database_health_report.json and
    # reference_download_priority.txt — no side effects on the pipeline.
    try:
        run_health_check(ANALYSIS_DIR)
    except Exception as _health_error:
        always_print(f"[Database] Health check failed (non-fatal): {_health_error}")

    _kb_countries = get_available_countries()
    if _kb_countries:
        debug_print(
            "[SubLocation] Regional knowledge available for: "
            + ", ".join(_kb_countries)
        )

    always_print("Press G to run analysis.  Press ESC to quit.")

    keyboard.add_hotkey("g", run_exploration_analysis)
    keyboard.wait("esc")

    print("Goodbye!")


if __name__ == "__main__":
    main()
