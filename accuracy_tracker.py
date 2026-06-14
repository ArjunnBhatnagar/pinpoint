"""accuracy_tracker.py — Round result logging and accuracy analysis.

Records each round's prediction vs ground truth, determines the failure stage,
and produces summary statistics when requested.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ACCURACY_LOG_DIR = Path("accuracy_log")
ROUNDS_FILE = ACCURACY_LOG_DIR / "rounds.jsonl"

# ── Compact country → broad-region fallback map ───────────────────────────────
# Used when the correct country did not appear in regional_consistency.json.
_COUNTRY_REGION: dict[str, str] = {
    # Africa
    "South Africa": "Sub-Saharan Africa", "Nigeria": "Sub-Saharan Africa",
    "Kenya": "Sub-Saharan Africa", "Ethiopia": "Sub-Saharan Africa",
    "Tanzania": "Sub-Saharan Africa", "Ghana": "Sub-Saharan Africa",
    "Cameroon": "Sub-Saharan Africa", "Uganda": "Sub-Saharan Africa",
    "Zimbabwe": "Sub-Saharan Africa", "Mozambique": "Sub-Saharan Africa",
    "Zambia": "Sub-Saharan Africa", "Botswana": "Sub-Saharan Africa",
    "Eswatini": "Sub-Saharan Africa", "Lesotho": "Sub-Saharan Africa",
    "Namibia": "Sub-Saharan Africa", "Angola": "Sub-Saharan Africa",
    "Senegal": "West Africa", "Ivory Coast": "West Africa",
    "Mali": "West Africa", "Burkina Faso": "West Africa",
    "Egypt": "North Africa", "Morocco": "North Africa",
    "Algeria": "North Africa", "Tunisia": "North Africa",
    "Libya": "North Africa",
    # Middle East
    "Turkey": "Middle East", "Iran": "Middle East",
    "Iraq": "Middle East", "Saudi Arabia": "Middle East",
    "Jordan": "Middle East", "Israel": "Middle East",
    "Lebanon": "Middle East", "Syria": "Middle East",
    "UAE": "Middle East", "Kuwait": "Middle East",
    # Europe
    "Germany": "Western Europe", "France": "Western Europe",
    "Italy": "Western Europe", "Spain": "Western Europe",
    "Portugal": "Western Europe", "Netherlands": "Western Europe",
    "Belgium": "Western Europe", "Switzerland": "Western Europe",
    "Austria": "Western Europe", "Luxembourg": "Western Europe",
    "Monaco": "Western Europe", "San Marino": "Western Europe",
    "Vatican City": "Western Europe",
    "United Kingdom": "Northern Europe", "Ireland": "Northern Europe",
    "Sweden": "Northern Europe", "Norway": "Northern Europe",
    "Finland": "Northern Europe", "Denmark": "Northern Europe",
    "Iceland": "Northern Europe", "Estonia": "Northern Europe",
    "Latvia": "Northern Europe", "Lithuania": "Northern Europe",
    "Poland": "Eastern Europe", "Czech Republic": "Eastern Europe",
    "Slovakia": "Eastern Europe", "Hungary": "Eastern Europe",
    "Romania": "Eastern Europe", "Bulgaria": "Eastern Europe",
    "Serbia": "Eastern Europe", "Croatia": "Eastern Europe",
    "Bosnia": "Eastern Europe", "Slovenia": "Eastern Europe",
    "Albania": "Eastern Europe", "North Macedonia": "Eastern Europe",
    "Montenegro": "Eastern Europe", "Moldova": "Eastern Europe",
    "Ukraine": "Eastern Europe", "Belarus": "Eastern Europe",
    "Greece": "Southern Europe",
    "Russia": "Russia",
    # Asia
    "Japan": "East Asia", "South Korea": "East Asia",
    "China": "East Asia", "Taiwan": "East Asia",
    "Mongolia": "East Asia",
    "India": "South Asia", "Pakistan": "South Asia",
    "Bangladesh": "South Asia", "Sri Lanka": "South Asia",
    "Nepal": "South Asia",
    "Thailand": "Southeast Asia", "Vietnam": "Southeast Asia",
    "Indonesia": "Southeast Asia", "Malaysia": "Southeast Asia",
    "Philippines": "Southeast Asia", "Cambodia": "Southeast Asia",
    "Myanmar": "Southeast Asia", "Laos": "Southeast Asia",
    "Singapore": "Southeast Asia", "Brunei": "Southeast Asia",
    "Kazakhstan": "Central Asia", "Uzbekistan": "Central Asia",
    "Kyrgyzstan": "Central Asia", "Tajikistan": "Central Asia",
    "Turkmenistan": "Central Asia", "Azerbaijan": "Central Asia",
    "Georgia": "Caucasus", "Armenia": "Caucasus",
    # Americas
    "United States": "North America", "Canada": "North America",
    "Mexico": "Central America", "Guatemala": "Central America",
    "Belize": "Central America", "Honduras": "Central America",
    "El Salvador": "Central America", "Nicaragua": "Central America",
    "Costa Rica": "Central America", "Panama": "Central America",
    "Brazil": "South America", "Argentina": "South America",
    "Chile": "South America", "Colombia": "South America",
    "Peru": "South America", "Venezuela": "South America",
    "Bolivia": "South America", "Ecuador": "South America",
    "Paraguay": "South America", "Uruguay": "South America",
    "Cuba": "Caribbean", "Dominican Republic": "Caribbean",
    "Haiti": "Caribbean",
    # Oceania
    "Australia": "Oceania", "New Zealand": "Oceania",
    "Papua New Guinea": "Oceania", "Fiji": "Oceania",
}

logger = logging.getLogger("accuracy_tracker")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[Accuracy] %(message)s"))
    logger.addHandler(handler)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_json(path: Path, default: Any = None) -> Any:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return default


def _region_for(country: str, country_regions: dict[str, str]) -> str | None:
    """Return the broad region for a country, with fallback to built-in map."""
    if country in country_regions:
        return country_regions[country]
    return _COUNTRY_REGION.get(country)


def _normalize(name: str) -> str:
    return name.strip().casefold()


# ── Failure stage logic ───────────────────────────────────────────────────────

def _determine_failure_stage(
    correct_answer: str,
    predicted_country: str,
    country_correct: bool,
    region_correct: bool | None,
    confidence: float,
    screenshots_collected: int,
    correct_state: str | None,
    predicted_state: str | None,
    first_pass_candidates: list[dict],
) -> tuple[str | None, str]:
    """Return (failure_stage, failure_reason)."""
    if country_correct:
        if correct_state and predicted_state and _normalize(correct_state) != _normalize(predicted_state):
            return (
                "sub_location_reasoning",
                f"Country correct ({predicted_country}) but state wrong "
                f"(predicted {predicted_state}, correct {correct_state})",
            )
        return None, "Correct prediction"

    # Country is wrong — determine why
    fp_countries = {_normalize(c.get("country", "")) for c in first_pass_candidates if isinstance(c, dict)}
    correct_in_first_pass = _normalize(correct_answer) in fp_countries

    if screenshots_collected < 3:
        return (
            "insufficient_capture",
            f"Only {screenshots_collected} screenshot(s) collected — "
            "insufficient visual evidence to reason correctly",
        )

    if confidence >= 0.75:
        return (
            "overconfident",
            f"High confidence ({confidence:.0%}) but wrong: predicted "
            f"{predicted_country}, correct was {correct_answer}",
        )

    if region_correct is False:
        return (
            "first_pass_reasoning",
            f"Wrong region predicted — correct answer ({correct_answer}) is in a "
            "different region from predicted region; first-pass reasoning went astray",
        )

    # Region is right but country is wrong
    if not correct_in_first_pass:
        return (
            "first_pass_reasoning",
            f"{correct_answer} was never considered as a candidate in first-pass reasoning",
        )

    return (
        "grounded_reasoning",
        f"{correct_answer} appeared in first-pass candidates but was eliminated "
        f"during grounded comparison or regional consistency, leaving {predicted_country}",
    )


# ── Main logging function ─────────────────────────────────────────────────────

def log_round_result(
    correct_answer: str,
    run_folder: Path,
    analysis_dir: Path,
) -> dict[str, Any]:
    """Build and append a round result record to rounds.jsonl.

    Returns the record dict for immediate display/confirmation.
    """
    ACCURACY_LOG_DIR.mkdir(parents=True, exist_ok=True)

    final_decision = _load_json(analysis_dir / "final_decision.json", {})
    confidence_report = _load_json(analysis_dir / "confidence_report.json", {})
    regional_consistency = _load_json(analysis_dir / "regional_consistency.json", {})
    pipeline_summary = _load_json(analysis_dir / "pipeline_summary.json", {})
    ocr_status = _load_json(analysis_dir / "ocr_status.json", {})
    ocr_results = _load_json(analysis_dir / "ocr_results.json", [])
    sub_location = _load_json(analysis_dir / "sub_location_result.json", {})
    first_pass = _load_json(analysis_dir / "first_pass.json", {})
    smart_capture = _load_json(analysis_dir / "smart_capture_report.json", {})

    predicted_country: str = str(final_decision.get("final_country") or "Unknown")
    confidence: float = float(final_decision.get("confidence", 0.0) or 0.0)
    confidence_label: str = str(final_decision.get("confidence_label") or "")
    decision_source: str = str(final_decision.get("decision_source") or "")
    strongest_region: str = str(final_decision.get("region_cluster") or "")

    # Region lookup
    country_regions: dict[str, str] = regional_consistency.get("country_regions", {})
    correct_region = _region_for(correct_answer, country_regions)
    region_correct: bool | None = None
    if correct_region and strongest_region:
        # Compare broad regions — normalised match
        region_correct = _normalize(correct_region) == _normalize(strongest_region)

    country_correct = _normalize(predicted_country) == _normalize(correct_answer)

    # Screenshots collected
    screenshots_collected: int = int(
        pipeline_summary.get("selected_image_count")
        or smart_capture.get("good_screenshots_collected")
        or 0
    )

    # OCR
    ocr_attempted = bool(ocr_status.get("attempted", True))
    ocr_found_text = False
    if ocr_attempted and isinstance(ocr_results, list):
        ocr_found_text = any(
            len(r.get("detections", [])) > 0
            for r in ocr_results
            if isinstance(r, dict)
        )

    # Definitive identifier
    def_id = confidence_report.get("definitive_identifier", {})
    definitive_identifier_detected = bool(
        isinstance(def_id, dict) and def_id.get("detected")
    )

    # Meta knowledge — always True since meta_knowledge_injector is always active
    meta_knowledge_used = True

    # Sub-location
    sub_location_attempted = bool(
        isinstance(sub_location, dict) and sub_location.get("sub_location_attempted")
    )
    predicted_state: str | None = (
        sub_location.get("final_state") if isinstance(sub_location, dict) else None
    )

    # First-pass candidates for failure stage logic
    first_pass_candidates = first_pass.get("top_candidates", []) if isinstance(first_pass, dict) else []

    failure_stage, failure_reason = _determine_failure_stage(
        correct_answer=correct_answer,
        predicted_country=predicted_country,
        country_correct=country_correct,
        region_correct=region_correct,
        confidence=confidence,
        screenshots_collected=screenshots_collected,
        correct_state=None,  # user did not supply state — extend later
        predicted_state=predicted_state,
        first_pass_candidates=first_pass_candidates,
    )

    record: dict[str, Any] = {
        "run_id": run_folder.name,
        "correct_answer": correct_answer,
        "predicted_country": predicted_country,
        "country_correct": country_correct,
        "predicted_state": predicted_state,
        "correct_state": None,
        "state_correct": None,
        "country_confidence": round(confidence, 3),
        "confidence_label": confidence_label,
        "decision_source": decision_source,
        "strongest_region_predicted": strongest_region,
        "correct_region": correct_region,
        "region_correct": region_correct,
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        "screenshots_collected": screenshots_collected,
        "ocr_found_text": ocr_found_text,
        "definitive_identifier_detected": definitive_identifier_detected,
        "meta_knowledge_used": meta_knowledge_used,
        "sub_location_attempted": sub_location_attempted,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    # Append to JSONL file
    with ROUNDS_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

    outcome = "✓ CORRECT" if country_correct else f"✗ WRONG (was {correct_answer})"
    logger.info(
        "Round logged: %s → %s  %s  [stage: %s]",
        predicted_country,
        correct_answer,
        outcome,
        failure_stage or "success",
    )
    return record


# ── Stats loading ─────────────────────────────────────────────────────────────

def load_all_rounds() -> list[dict[str, Any]]:
    """Load all round records from rounds.jsonl."""
    if not ROUNDS_FILE.exists():
        return []
    rounds = []
    with ROUNDS_FILE.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rounds.append(json.loads(line))
            except Exception:
                pass
    return rounds


# ── Stats printing ────────────────────────────────────────────────────────────

def print_accuracy_stats() -> None:
    """Print the full accuracy summary to stdout."""
    rounds = load_all_rounds()

    if not rounds:
        print("\n[Accuracy] No rounds logged yet. Play some rounds and provide answers.")
        return

    total = len(rounds)
    country_correct_rounds = [r for r in rounds if r.get("country_correct")]
    country_wrong_rounds   = [r for r in rounds if not r.get("country_correct")]
    country_correct = len(country_correct_rounds)

    region_evaluated = [r for r in rounds if r.get("region_correct") is not None]
    region_correct_count = sum(1 for r in region_evaluated if r.get("region_correct"))

    state_rounds = [r for r in rounds if r.get("sub_location_attempted")]
    state_correct_count = sum(
        1 for r in state_rounds
        if r.get("state_correct") is True
    )

    # Confidence averages
    conf_correct = [r["country_confidence"] for r in country_correct_rounds if r.get("country_confidence") is not None]
    conf_wrong   = [r["country_confidence"] for r in country_wrong_rounds   if r.get("country_confidence") is not None]
    avg_conf_correct = sum(conf_correct) / len(conf_correct) if conf_correct else 0.0
    avg_conf_wrong   = sum(conf_wrong)   / len(conf_wrong)   if conf_wrong   else 0.0

    # Failure stages
    stage_counter: Counter = Counter(
        r.get("failure_stage") for r in rounds if r.get("failure_stage")
    )

    # Most confused pairs (predicted → actual, only wrong rounds)
    pair_counter: Counter = Counter(
        (r["predicted_country"], r["correct_answer"])
        for r in country_wrong_rounds
        if r.get("predicted_country") and r.get("correct_answer")
    )

    # Definitive identifier stats
    def_rounds = [r for r in rounds if r.get("definitive_identifier_detected")]
    def_correct = sum(1 for r in def_rounds if r.get("country_correct"))

    # Low-screenshot stats
    low_ss_rounds = [r for r in rounds if (r.get("screenshots_collected") or 0) < 3]
    low_ss_correct = sum(1 for r in low_ss_rounds if r.get("country_correct"))

    # ── Print ──────────────────────────────────────────────────────────────────
    W = 44
    print()
    print("=" * W)
    print("ACCURACY SUMMARY")
    print("=" * W)
    print(f"Total rounds logged:        {total}")
    if total:
        pct = country_correct / total * 100
        print(f"Country accuracy:           {pct:.1f}% ({country_correct}/{total})")
    if region_evaluated:
        rpct = region_correct_count / len(region_evaluated) * 100
        print(f"Region accuracy:            {rpct:.1f}% ({region_correct_count}/{len(region_evaluated)})")
    if state_rounds:
        spct = state_correct_count / len(state_rounds) * 100 if state_rounds else 0
        print(f"State accuracy:             {spct:.1f}% ({state_correct_count}/{len(state_rounds)} attempted)")

    print()
    if pair_counter:
        print("Most common failures:")
        for idx, ((predicted, actual), count) in enumerate(pair_counter.most_common(5), 1):
            times = "time" if count == 1 else "times"
            print(f"  {idx}. {predicted} predicted, {actual} correct ({count} {times})")

    print()
    if stage_counter:
        print("Failure stages:")
        for stage, count in stage_counter.most_common():
            print(f"  - {stage + ':':<32} {count} failure{'s' if count > 1 else ''}")

    print()
    if conf_correct or conf_wrong:
        print(f"Average confidence when correct:   {avg_conf_correct:.2f}")
        print(f"Average confidence when wrong:     {avg_conf_wrong:.2f}")

    print()
    if def_rounds:
        def_pct = def_correct / len(def_rounds) * 100
        print(f"Rounds with definitive identifier: {len(def_rounds)} ({def_pct:.0f}% correct)")
    if low_ss_rounds:
        ls_pct = low_ss_correct / len(low_ss_rounds) * 100
        print(f"Rounds with < 3 screenshots:       {len(low_ss_rounds)} ({ls_pct:.1f}% correct)")

    # Most confused pairs section (Part D)
    if pair_counter:
        print()
        print("Most confused country pairs:")
        for (predicted, actual), count in pair_counter.most_common(8):
            arrow = f"{predicted} → {actual}"
            print(f"  {arrow:<40} {count}x")
        print()
        print("  Tip: add elimination_clues for the top pairs to country_metas.json")

    print("=" * W)
    print()
