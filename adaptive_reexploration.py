import json
import logging
import os
import re
from pathlib import Path
from typing import Any


# Phase 9C: deterministic post-run decision layer.
# Automatic movement stays disabled by default. This module decides whether one
# additional controlled pass would be useful, but it never creates an open loop.
ANALYSIS_DIR = Path("analysis")
DECISION_PATH = ANALYSIS_DIR / "adaptive_reexploration_decision.json"

ADAPTIVE_REEXPLORE_ENABLED = os.getenv(
    "ADAPTIVE_REEXPLORE_ENABLED",
    "false",
).strip().casefold() in {"1", "true", "yes", "on"}
MAX_REEXPLORATION_CYCLES = 1

UI_OCR_TERMS = {
    "adobe",
    "chrome",
    "creative cloud",
    "facebook",
    "google",
    "guess",
    "maps",
    "return",
    "singleplayer",
    "whatsapp",
    "world",
    "youtube",
}
STRONG_OCR_PATTERNS = [
    re.compile(r"\b[a-z]{1,3}[- ]?\d{1,4}\b", re.I),  # Route markers such as M-36 or A12.
    re.compile(r"\b\d{1,4}\s+(st|street|rd|road|ave|avenue|blvd|boulevard|hwy|highway)\b", re.I),
    re.compile(r"\b(st|street|rd|road|ave|avenue|blvd|boulevard|hwy|highway)\b", re.I),
]

logger = logging.getLogger("adaptive_reexploration")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[Adaptive] %(message)s"))
    logger.addHandler(handler)


def load_json_file(path: Path, default: Any) -> tuple[Any, list[str]]:
    """Load JSON safely and report missing or malformed artifacts."""
    if not path.exists():
        return default, [f"Missing file: {path}"]

    try:
        return json.loads(path.read_text(encoding="utf-8")), []
    except Exception as error:
        return default, [f"Malformed JSON in {path}: {error}"]


def load_run_artifact(
    analysis_dir: Path,
    run_folder: Path | None,
    filename: str,
    default: Any,
) -> tuple[Any, list[str], str | None]:
    """Load an analysis artifact, falling back to the current run folder."""
    candidates = [analysis_dir / filename]
    if run_folder:
        candidates.append(run_folder / filename)

    for path in candidates:
        if path.exists():
            payload, warnings = load_json_file(path, default)
            return payload, warnings, str(path)

    return default, [f"Missing file: {filename}"], None


def normalize_text(value: Any) -> str:
    """Normalize OCR and reasoning text for deterministic checks."""
    return str(value or "").strip().casefold()


def evaluate_ocr_quality(ocr_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify OCR quality while excluding obvious application UI text."""
    detections = []
    useful = []
    ignored_ui = []
    strong_clues = []

    for result in ocr_results if isinstance(ocr_results, list) else []:
        for detection in result.get("detections", []):
            text = str(detection.get("text", "")).strip()
            if not text:
                continue

            confidence = float(detection.get("confidence", 0.0) or 0.0)
            normalized = normalize_text(text)
            item = {"text": text, "confidence": confidence}
            detections.append(item)

            if normalized in UI_OCR_TERMS or any(term in normalized for term in UI_OCR_TERMS):
                ignored_ui.append(item)
                continue

            if len(normalized) <= 2 and not normalized.isalpha():
                continue

            useful.append(item)
            if any(pattern.search(text) for pattern in STRONG_OCR_PATTERNS):
                strong_clues.append(item)

    if strong_clues:
        quality = "strong"
    elif len(useful) >= 3:
        quality = "medium"
    elif useful:
        quality = "weak"
    else:
        quality = "useless"

    logger.info("OCR quality: %s", quality)
    return {
        "quality": quality,
        "total_detections": len(detections),
        "useful_detections": useful,
        "ignored_ui_detections": ignored_ui,
        "strong_clues": strong_clues,
    }


def evaluate_retrieval_quality(
    retrieval_package: dict[str, Any],
    first_pass: dict[str, Any],
) -> dict[str, Any]:
    """Check whether references were retrieved for actual first-pass candidates."""
    summary = retrieval_package.get("summary", {}) if isinstance(retrieval_package, dict) else {}
    total_images = int(summary.get("total_images", 0) or 0)
    total_texts = int(summary.get("total_texts", 0) or 0)
    missing_countries = summary.get("missing_countries", [])
    missing_meta_categories = summary.get("missing_meta_categories", [])
    retrieved = retrieval_package.get("countries_actually_retrieved", [])
    first_pass_countries = [
        str(item.get("country"))
        for item in first_pass.get("top_candidates", [])
        if isinstance(item, dict) and item.get("country")
    ]
    retrieved_normalized = {normalize_text(country) for country in retrieved}
    unmatched_candidates = [
        country
        for country in first_pass_countries
        if normalize_text(country) not in retrieved_normalized
    ]

    if total_images == 0 and total_texts == 0:
        quality = "empty"
    elif missing_countries or unmatched_candidates:
        quality = "weak"
    elif total_images >= 5 and total_texts >= 3:
        quality = "strong"
    elif total_images >= 1 or total_texts >= 2:
        quality = "medium"
    else:
        quality = "weak"

    logger.info("Retrieval quality: %s", quality)
    return {
        "quality": quality,
        "total_images": total_images,
        "total_texts": total_texts,
        "candidate_countries": first_pass_countries,
        "countries_actually_retrieved": retrieved,
        "missing_countries": missing_countries,
        "unmatched_candidates": unmatched_candidates,
        "missing_meta_categories": missing_meta_categories,
    }


def evaluate_smart_capture_quality(report: dict[str, Any]) -> dict[str, Any]:
    """Evaluate evidence quantity, diversity, fallback use, and duplicates."""
    if not isinstance(report, dict) or not report:
        logger.info("Smart capture quality: poor")
        return {
            "quality": "poor",
            "good_screenshots_collected": 0,
            "fallback_used": True,
            "average_evidence_score": 0.0,
            "categories": [],
            "duplicate_rejections": 0,
            "stop_reason": "missing_report",
        }

    target = int(report.get("target_good_screenshots", 5) or 5)
    minimum = int(report.get("min_good_screenshots", 4) or 4)
    good_count = int(report.get("good_screenshots_collected", 0) or 0)
    fallback_used = bool(report.get("fallback_used", False))
    kept = report.get("kept_screenshots", [])
    rejected = report.get("rejected_screenshots", [])
    evidence_scores = [
        float(item.get("evidence_score", 0.0) or 0.0)
        for item in kept
        if isinstance(item, dict)
    ]
    categories = sorted(
        {
            str(category)
            for item in kept
            if isinstance(item, dict)
            for category in item.get("evidence_categories", [])
        }
    )
    duplicate_rejections = sum(
        1
        for item in rejected
        if isinstance(item, dict) and item.get("duplicate_or_near_duplicate")
    )
    average_score = sum(evidence_scores) / max(len(evidence_scores), 1)
    strong_categories = {
        "road_signs_candidate",
        "signposts_candidate",
        "street_names_or_business_text_candidate",
        "road_markings",
        "road_surface_markings",
        "intersection_candidate",
        "utility_poles_candidate",
        "architecture_or_settlement",
    }
    category_count = len(strong_categories & set(categories))

    if good_count >= target and average_score >= 8.5 and category_count >= 3 and not fallback_used:
        quality = "strong"
    elif good_count >= minimum and average_score >= 7.5 and category_count >= 2 and not fallback_used:
        quality = "good"
    elif good_count >= 2 or (kept and average_score >= 6.0):
        quality = "weak"
    else:
        quality = "poor"

    logger.info("Smart capture quality: %s", quality)
    return {
        "quality": quality,
        "target_good_screenshots": target,
        "good_screenshots_collected": good_count,
        "attempts_used": int(report.get("attempts_used", 0) or 0),
        "fallback_used": fallback_used,
        "average_evidence_score": round(average_score, 3),
        "categories": categories,
        "duplicate_rejections": duplicate_rejections,
        "stop_reason": report.get("stop_reason"),
    }


def evaluate_confidence_quality(confidence_report: dict[str, Any]) -> dict[str, Any]:
    """Extract deterministic confidence and score-margin quality."""
    if not isinstance(confidence_report, dict) or not confidence_report:
        logger.info("Verified confidence: 0.00")
        return {
            "quality": "unreliable",
            "verified_confidence": 0.0,
            "trust_level": "unreliable",
            "recommendation": "manual_review",
            "score_margin_quality": "none",
            "warnings": ["confidence_report.json is missing or malformed"],
        }

    verified = float(confidence_report.get("verified_confidence", 0.0) or 0.0)
    trust_level = str(confidence_report.get("trust_level", "unreliable"))
    score_margin = confidence_report.get("score_analysis", {}).get("margin_strength", "none")
    high_specificity = confidence_report.get("high_specificity_evidence", {})
    logger.info("Verified confidence: %.2f", verified)
    return {
        "quality": trust_level,
        "verified_confidence": verified,
        "trust_level": trust_level,
        "recommendation": confidence_report.get("recommendation"),
        "score_margin_quality": score_margin,
        "warnings": confidence_report.get("warnings", []),
        "high_specificity_evidence": high_specificity,
    }


def detect_serious_contradictions(
    grounded_result: dict[str, Any],
    first_pass: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Identify genuine evidence conflicts between near-equal regional candidates.

    A contradiction is SERIOUS only when ALL three conditions hold:
      1. Two candidates have confidence scores within 0.10 of each other.
      2. They belong to completely different geographic regions.
      3. Neither has been flagged region_consistent = False by first_pass.

    A list of alternatives where one country dominates (e.g. USA 0.85 vs
    Canada 0.12) is NOT a contradiction. Contradictions that only explain why
    alternative countries don't match are also NOT serious contradictions.
    """
    if not isinstance(grounded_result, dict) or not grounded_result:
        return {
            "serious": True,
            "contradictions": [],
            "uncertainty_notes": [],
            "matched_serious_terms": [],
            "reason": "grounded_result.json is missing or malformed",
        }

    if grounded_result.get("reasoning_status") != "ok":
        return {
            "serious": True,
            "contradictions": grounded_result.get("contradictions", []),
            "uncertainty_notes": grounded_result.get("uncertainty_notes", []),
            "matched_serious_terms": ["reasoning_status_not_ok"],
            "reason": "grounded reasoning status is not ok",
        }

    contradictions = grounded_result.get("contradictions", [])
    notes = grounded_result.get("uncertainty_notes", [])

    # Build region map from first_pass candidates (Gemini-provided region fields).
    fp_candidates: list[dict] = []
    if isinstance(first_pass, dict):
        fp_candidates = first_pass.get("top_candidates", [])
    region_by_country: dict[str, str | None] = {
        str(c.get("country", "")): c.get("region")
        for c in fp_candidates
        if isinstance(c, dict) and c.get("country")
    }
    region_consistent_by_country: dict[str, bool | None] = {
        str(c.get("country", "")): c.get("region_consistent")
        for c in fp_candidates
        if isinstance(c, dict) and c.get("country")
    }

    # Analyse country_scores for near-equal pairs in different regions.
    country_scores = grounded_result.get("country_scores", {})
    scored_pairs: list[tuple[str, float]] = []
    for country, score in country_scores.items():
        try:
            scored_pairs.append((str(country), float(score)))
        except (TypeError, ValueError):
            pass
    scored_pairs.sort(key=lambda x: x[1], reverse=True)

    serious = False
    contradiction_reason = ""
    for i in range(len(scored_pairs) - 1):
        name_a, score_a = scored_pairs[i]
        name_b, score_b = scored_pairs[i + 1]
        gap = abs(score_a - score_b)
        if gap > 0.10:
            continue  # not near-equal — not a contradiction
        region_a = region_by_country.get(name_a)
        region_b = region_by_country.get(name_b)
        if region_a and region_b and region_a == region_b:
            continue  # same region — not a contradiction
        rc_a = region_consistent_by_country.get(name_a)
        rc_b = region_consistent_by_country.get(name_b)
        if rc_a is False or rc_b is False:
            continue  # one candidate already flagged as region-inconsistent — not a serious conflict
        # All three conditions met: real contradiction.
        serious = True
        contradiction_reason = (
            f"Two near-equal candidates in different regions: "
            f"{name_a} ({score_a:.2f}) vs {name_b} ({score_b:.2f})"
        )
        logger.info("[Adaptive] %s", contradiction_reason)
        break

    return {
        "serious": serious,
        "contradictions": contradictions,
        "uncertainty_notes": notes,
        "matched_serious_terms": [contradiction_reason] if contradiction_reason else [],
        "reason": contradiction_reason or "no near-equal multi-region conflict detected",
    }


def determine_reexploration_strategy(
    ocr_quality: dict[str, Any],
    retrieval_quality: dict[str, Any],
    capture_quality: dict[str, Any],
    confidence_quality: dict[str, Any],
) -> str:
    """Recommend one targeted evidence-gathering strategy."""
    categories = set(capture_quality.get("categories", []))
    # "disabled" means OCR was not run — don't recommend seek_signage for that alone.
    if ocr_quality["quality"] in {"useless", "weak"}:
        return "seek_signage"
    if capture_quality.get("duplicate_rejections", 0) >= 2:
        return "avoid_duplicate_views"
    if "intersection_candidate" not in categories:
        return "seek_intersection"
    if "utility_poles_candidate" not in categories:
        return "seek_utility_poles"
    if "road_markings" not in categories:
        return "seek_road_markings"
    if retrieval_quality["quality"] in {"empty", "weak"}:
        return "rotate_scan"
    if confidence_quality["score_margin_quality"] in {"none", "weak"}:
        return "move_forward_longer"
    return "rotate_scan"


def determine_adaptive_decision(
    confidence_quality: dict[str, Any],
    ocr_quality: dict[str, Any],
    retrieval_quality: dict[str, Any],
    capture_quality: dict[str, Any],
    contradictions: dict[str, Any],
    grounded_result: dict[str, Any],
    selected_country: str | None,
    artifact_warnings: list[str],
) -> dict[str, Any]:
    """Choose accept, caution, one re-exploration pass, or manual review."""
    verified = confidence_quality["verified_confidence"]
    trust_level = confidence_quality["trust_level"]
    recommendation = confidence_quality["recommendation"]
    high_specificity = confidence_quality.get("high_specificity_evidence", {})
    strong_specificity = bool(
        high_specificity.get("present")
        and high_specificity.get("supports_final_country")
        and high_specificity.get("strength") in {"high", "very_high"}
    )
    final_country = selected_country or (
        grounded_result.get("final_country")
        if isinstance(grounded_result, dict)
        else None
    )
    strategy = determine_reexploration_strategy(
        ocr_quality,
        retrieval_quality,
        capture_quality,
        confidence_quality,
    )

    broken_result = not final_country or confidence_quality["quality"] == "unreliable" and artifact_warnings
    if broken_result:
        return {
            "decision": "manual_review",
            "should_reexplore": False,
            "reason": "required reasoning artifacts are missing or no reliable country prediction exists",
            "recommended_strategy": None,
        }

    if (
        verified >= 0.80
        and trust_level in {"high", "very_high"}
        and strong_specificity
        and not contradictions["serious"]
    ):
        return {
            "decision": "accept_prediction",
            "should_reexplore": False,
            "reason": "high verified confidence with strong location-specific visual evidence",
            "recommended_strategy": None,
        }

    accept_ready = (
        verified >= 0.80
        and trust_level in {"high", "very_high"}
        and retrieval_quality["quality"] in {"medium", "strong"}
        and capture_quality["quality"] in {"good", "strong"}
        and not contradictions["serious"]
    )
    if accept_ready:
        return {
            "decision": "accept_prediction",
            "should_reexplore": False,
            "reason": "high verified confidence with useful retrieval references and strong captures",
            "recommended_strategy": None,
        }

    caution_ready = (
        verified >= 0.65
        and trust_level in {"medium", "high", "very_high"}
        and retrieval_quality["quality"] in {"medium", "strong"}
        and capture_quality["quality"] in {"good", "strong"}
        and not contradictions["serious"]
    )
    if caution_ready:
        return {
            "decision": "accept_with_caution",
            "should_reexplore": False,
            "reason": "prediction is plausible but one or more evidence sources remain limited",
            "recommended_strategy": None,
        }

    ocr_active_and_weak = ocr_quality["quality"] in {"useless", "weak"}  # "disabled" is excluded
    rerun_requested = recommendation in {"needs_more_evidence", "rerun_exploration", "manual_review"}
    weak_combination = (
        retrieval_quality["quality"] in {"empty", "weak"} and trust_level not in {"high", "very_high"}
    )
    weak_ocr_and_visual = (
        ocr_active_and_weak
        and retrieval_quality["quality"] in {"empty", "weak"}
        and capture_quality["quality"] in {"weak", "poor"}
    )
    close_scores = confidence_quality["score_margin_quality"] in {"none", "weak"}

    if (
        rerun_requested
        or verified < 0.65
        or weak_combination
        or capture_quality["quality"] in {"weak", "poor"}
        or weak_ocr_and_visual
        or close_scores
        or contradictions["serious"]
    ):
        reasons = []
        if verified < 0.65:
            reasons.append("verified confidence is below 0.65")
        if ocr_active_and_weak:
            reasons.append(f"OCR quality is {ocr_quality['quality']}")
        if close_scores:
            reasons.append("top country scores are too close")
        if capture_quality["quality"] in {"weak", "poor"}:
            reasons.append(f"smart capture quality is {capture_quality['quality']}")
        if contradictions["serious"]:
            # Use the specific contradiction reason instead of generic text.
            specific_reason = (
                contradictions.get("reason")
                or contradictions.get("matched_serious_terms", [""])[0]
                or "serious evidence conflict detected"
            )
            reasons.append(specific_reason)
        return {
            "decision": "reexplore_once",
            "should_reexplore": True,
            "reason": "; ".join(reasons) or "additional evidence would improve reliability",
            "recommended_strategy": strategy,
        }

    return {
        "decision": "manual_review",
        "should_reexplore": False,
        "reason": "evidence remains ambiguous and does not justify an automatic recommendation",
        "recommended_strategy": None,
    }


def save_adaptive_decision(
    decision: dict[str, Any],
    output_path: Path = DECISION_PATH,
) -> None:
    """Save adaptive_reexploration_decision.json."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    logger.info("Saved adaptive_reexploration_decision.json")


def run_controlled_reexploration_once(*_args, **_kwargs) -> dict[str, Any]:
    """Reserved integration point for one explicitly enabled future rerun.

    Automatic execution is intentionally not wired into main.py yet. Calling
    the full pipeline recursively from inside itself would risk overwriting
    analysis artifacts and complicate safety guarantees.
    """
    return {
        "executed": False,
        "reason": "automatic controlled re-exploration is not wired; decision-only mode is active",
        "max_cycles": MAX_REEXPLORATION_CYCLES,
    }


def _run_adaptive_reexploration_decision(
    analysis_dir: Path = ANALYSIS_DIR,
    run_folder: Path | None = None,
) -> dict[str, Any]:
    """Evaluate completed run artifacts and save a safe Phase 9C decision."""
    warnings = []

    logger.info("Loading confidence report")
    confidence_report, file_warnings, _ = load_run_artifact(
        analysis_dir, run_folder, "confidence_report.json", {}
    )
    warnings.extend(file_warnings)
    grounded_result, file_warnings, _ = load_run_artifact(
        analysis_dir, run_folder, "grounded_result.json", {}
    )
    warnings.extend(file_warnings)
    first_pass, file_warnings, _ = load_run_artifact(
        analysis_dir, run_folder, "first_pass.json", {}
    )
    warnings.extend(file_warnings)
    retrieval_package, file_warnings, _ = load_run_artifact(
        analysis_dir, run_folder, "retrieval_package.json", {}
    )
    warnings.extend(file_warnings)
    ocr_results, file_warnings, _ = load_run_artifact(
        analysis_dir, run_folder, "ocr_results.json", []
    )
    warnings.extend(file_warnings)
    _image_ranking, file_warnings, _ = load_run_artifact(
        analysis_dir, run_folder, "image_ranking.json", {}
    )
    warnings.extend(file_warnings)
    smart_capture_report, file_warnings, smart_capture_source = load_run_artifact(
        analysis_dir, run_folder, "smart_capture_report.json", {}
    )
    warnings.extend(file_warnings)

    # Check whether OCR was disabled for this run.
    ocr_status_path = analysis_dir / "ocr_status.json"
    ocr_was_attempted = True
    if ocr_status_path.exists():
        try:
            ocr_status = json.loads(ocr_status_path.read_text(encoding="utf-8"))
            ocr_was_attempted = bool(ocr_status.get("attempted", True))
        except Exception:
            pass
    if not ocr_was_attempted:
        logger.info("[Adaptive] OCR disabled, skipping OCR evidence evaluation")

    ocr_quality = evaluate_ocr_quality(ocr_results)
    if not ocr_was_attempted:
        ocr_quality["quality"] = "disabled"
        ocr_quality["total_detections"] = 0

    retrieval_quality = evaluate_retrieval_quality(retrieval_package, first_pass)
    capture_quality = evaluate_smart_capture_quality(smart_capture_report)
    confidence_quality = evaluate_confidence_quality(confidence_report)
    contradictions = detect_serious_contradictions(grounded_result, first_pass)
    decision_fields = determine_adaptive_decision(
        confidence_quality,
        ocr_quality,
        retrieval_quality,
        capture_quality,
        contradictions,
        grounded_result,
        confidence_report.get("final_country"),
        warnings,
    )

    decision = {
        **decision_fields,
        "final_country": confidence_report.get("final_country")
        or grounded_result.get("final_country"),
        "evidence_summary": {
            "confidence_trust_level": confidence_quality["trust_level"],
            "verified_confidence": confidence_quality["verified_confidence"],
            "ocr_quality": ocr_quality["quality"],
            "retrieval_quality": retrieval_quality["quality"],
            "smart_capture_quality": capture_quality["quality"],
            "score_margin_quality": confidence_quality["score_margin_quality"],
            "serious_contradictions": contradictions["serious"],
            "high_specificity_evidence": confidence_quality.get(
                "high_specificity_evidence",
                {},
            ),
        },
        "details": {
            "ocr": ocr_quality,
            "retrieval": retrieval_quality,
            "smart_capture": capture_quality,
            "confidence": confidence_quality,
            "contradictions": contradictions,
            "smart_capture_report_source": smart_capture_source,
        },
        "automatic_reexploration_enabled": ADAPTIVE_REEXPLORE_ENABLED,
        "automatic_reexploration_executed": False,
        "max_reexploration_cycles": MAX_REEXPLORATION_CYCLES,
        "warnings": warnings,
    }

    logger.info("Decision: %s", decision["decision"])
    if decision.get("recommended_strategy"):
        logger.info("Strategy: %s", decision["recommended_strategy"])
    logger.info("Reason: %s", decision["reason"])
    save_adaptive_decision(decision, analysis_dir / "adaptive_reexploration_decision.json")
    return decision


def run_adaptive_reexploration_decision(
    analysis_dir: Path = ANALYSIS_DIR,
    run_folder: Path | None = None,
) -> dict[str, Any]:
    """Run Phase 9C and always save a clean manual-review fallback on failure."""
    try:
        return _run_adaptive_reexploration_decision(analysis_dir, run_folder)
    except Exception as error:
        logger.warning("Adaptive decision fallback: %s", error)
        decision = {
            "decision": "manual_review",
            "should_reexplore": False,
            "reason": "adaptive decision evaluation failed; manual review is required",
            "recommended_strategy": None,
            "final_country": None,
            "evidence_summary": {
                "confidence_trust_level": "unreliable",
                "verified_confidence": 0.0,
                "ocr_quality": "unknown",
                "retrieval_quality": "unknown",
                "smart_capture_quality": "unknown",
                "score_margin_quality": "unknown",
                "serious_contradictions": True,
            },
            "automatic_reexploration_enabled": ADAPTIVE_REEXPLORE_ENABLED,
            "automatic_reexploration_executed": False,
            "max_reexploration_cycles": MAX_REEXPLORATION_CYCLES,
            "warnings": [str(error)],
        }
        save_adaptive_decision(
            decision,
            analysis_dir / "adaptive_reexploration_decision.json",
        )
        return decision


def main() -> None:
    """Run adaptive decision evaluation from the terminal."""
    decision = run_adaptive_reexploration_decision()
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
