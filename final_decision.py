"""final_decision.py — Single authoritative final country decision.

Priority chain:
  1. adaptive_country  (from adaptive_reexploration_decision.json)
  2. region_consistent_country  (from regional_consistency.json)
  3. raw_grounded_country  (from grounded_result.json)
  4. manual_review  (only if all three are missing)

After picking from the chain, a region-consistency guard is applied:
if the chosen country does not belong to strongest_region but
region_consistent_country does, region_consistent_country wins.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger("final_decision")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[FinalDecision] %(message)s"))
    logger.addHandler(handler)

SHOW_DECISION_DEBUG = False

# Minimum state confidence to display a specific state (vs just the region).
_STATE_CONFIDENCE_DISPLAY_THRESHOLD = 0.65

_TRUST_TO_LABEL: dict[str, str] = {
    "locked": "Locked",
    "strong": "Strong",
    "good": "Good",
    "best_guess": "Best guess",
    "weak_guess": "Weak",
    "unreliable": "Weak",
}

_CONFIDENCE_THRESHOLDS: list[tuple[float, str]] = [
    (0.80, "Strong"),
    (0.65, "Good"),
    (0.45, "Best guess"),
    (0.25, "Weak"),
    (0.0, "Very weak"),
]


def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _confidence_label(confidence: float) -> str:
    for threshold, label in _CONFIDENCE_THRESHOLDS:
        if confidence >= threshold:
            return label
    return "Very weak"


def _best_display_confidence(
    verified: float,
    regional_scores: dict[str, float],
    final_country: str,
) -> float:
    """Return a non-zero display confidence when verifier over-penalized."""
    if verified > 0.0:
        return round(verified, 2)
    regional_score = regional_scores.get(final_country, 0.0)
    if regional_score > 0.0:
        # Scale regional score (0–1) to a realistic display range (0.30–0.75)
        scaled = 0.30 + regional_score * 0.45
        return round(min(scaled, 0.75), 2)
    return 0.25


def _alternatives(
    final_country: str,
    regional_scores: dict[str, float],
    first_pass_candidates: list[dict],
    max_count: int = 3,
) -> list[str]:
    candidates: dict[str, float] = {}
    for entry in first_pass_candidates:
        name = entry.get("country") or entry.get("name", "")
        if name and name != final_country:
            candidates[name] = float(entry.get("confidence", 0.0))
    for name, score in regional_scores.items():
        if name != final_country and name not in candidates:
            candidates[name] = score
    sorted_alts = sorted(candidates, key=lambda n: candidates[n], reverse=True)
    return sorted_alts[:max_count]


def run_final_decision(analysis_dir: Path) -> dict:
    """Produce one authoritative final country and save final_decision.json."""
    grounded = _load_json(analysis_dir / "grounded_result.json")
    regional = _load_json(analysis_dir / "regional_consistency.json")
    confidence = _load_json(analysis_dir / "confidence_report.json")
    adaptive = _load_json(analysis_dir / "adaptive_reexploration_decision.json")
    first_pass = _load_json(analysis_dir / "first_pass.json")

    raw_grounded_country: str | None = grounded.get("final_country")
    # Read the new field name; fall back to legacy "selected_country" for
    # regional_consistency.json files written before this refactor.
    region_consistent_country: str | None = (
        regional.get("region_consistent_country")
        or regional.get("selected_country")
    )
    adaptive_country: str | None = adaptive.get("final_country")

    # Country → region lookup from regional_consistency output
    country_regions: dict[str, str | None] = regional.get("country_regions", {})
    strongest_region: str = regional.get("strongest_region", "")

    def _region_of(country: str | None) -> str | None:
        if not country:
            return None
        return country_regions.get(country)

    # Priority chain (initial pick)
    if adaptive_country:
        final_country = adaptive_country
        decision_source = "adaptive_country"
        logger.info("Priority chain: adaptive_country = %s", adaptive_country)
    elif region_consistent_country:
        final_country = region_consistent_country
        decision_source = "region_consistent_country"
        logger.info("Priority chain: region_consistent_country = %s", region_consistent_country)
    elif raw_grounded_country:
        final_country = raw_grounded_country
        decision_source = "raw_grounded_country"
        logger.info("Priority chain: raw_grounded_country = %s", raw_grounded_country)
    else:
        final_country = "Unknown"
        decision_source = "manual_review"
        logger.info("Priority chain: no valid country found — manual_review")

    # Region-consistency guard: if the chosen country's region conflicts with
    # strongest_region AND region_consistent_country does belong to
    # strongest_region, override to region_consistent_country.
    if strongest_region and region_consistent_country:
        chosen_region = _region_of(final_country)
        rc_region = _region_of(region_consistent_country)
        if chosen_region != strongest_region and rc_region == strongest_region and final_country != region_consistent_country:
            logger.info(
                "Region guard triggered: %s is in %s, not %s — overriding with %s (%s)",
                final_country,
                chosen_region or "no region",
                strongest_region,
                region_consistent_country,
                strongest_region,
            )
            final_country = region_consistent_country
            decision_source = "region_consistent_country (region guard)"
        else:
            logger.info(
                "Region guard: %s accepted (region=%s, strongest=%s)",
                final_country,
                chosen_region or "no region",
                strongest_region,
            )

    logger.info("Final country: %s [source=%s]", final_country, decision_source)

    # Confidence
    verified = float(confidence.get("verified_confidence", 0.0))
    trust_level: str = confidence.get("trust_level", "unreliable")
    first_pass_candidates: list[dict] = first_pass.get("top_candidates", [])

    # Build a score map from first_pass candidates (works with both old and new
    # regional_consistency.json formats — no longer reads regional_consistency_scores).
    fp_scores: dict[str, float] = {
        c["country"]: float(c.get("confidence", 0.0))
        for c in first_pass_candidates
        if isinstance(c, dict) and c.get("country")
    }
    regional_scores: dict[str, float] = fp_scores  # used by _alternatives below

    display_confidence = _best_display_confidence(verified, fp_scores, final_country)
    confidence_label = _confidence_label(display_confidence)

    # Prefer the human-readable trust label if confidence is verified above 0
    if verified > 0.0 and trust_level in _TRUST_TO_LABEL:
        confidence_label = _TRUST_TO_LABEL[trust_level]

    recommended_strategy = adaptive.get("recommended_strategy") or confidence.get("recommendation", "")
    needs_more_evidence = display_confidence < 0.65

    alternatives = _alternatives(final_country, regional_scores, first_pass_candidates)

    reason_parts: list[str] = []
    if decision_source == "adaptive_country":
        reason_parts.append(
            f"Adaptive decision selected {final_country} as the most consistent answer."
        )
    if decision_source == "region_consistent_country":
        reason_parts.append(
            f"Regional consistency overrode raw grounded country ({raw_grounded_country}) "
            f"because the strongest region ({strongest_region}) aligns better with {final_country}."
        )
    if raw_grounded_country and raw_grounded_country != final_country:
        reason_parts.append(
            f"Raw grounded country was {raw_grounded_country}; "
            f"regional consistency raised {final_country} to the top."
        )
    reason = " ".join(reason_parts) or f"Selected {final_country} from available evidence."

    result: dict = {
        "final_country": final_country,
        "decision_source": decision_source,
        "confidence": display_confidence,
        "confidence_label": confidence_label,
        "region_cluster": strongest_region,
        "alternatives": alternatives,
        "needs_more_evidence": needs_more_evidence,
        "recommended_strategy": recommended_strategy,
        "reason": reason,
        "debug": {
            "raw_grounded_country": raw_grounded_country,
            "region_consistent_country": region_consistent_country,
            "adaptive_country": adaptive_country,
            "strongest_region": strongest_region,
            "verified_confidence": verified,
            "trust_level": trust_level,
        },
    }

    output_path = analysis_dir / "final_decision.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


def print_final_decision(
    result: dict,
    sub_location: dict | None = None,
) -> None:
    """Print the single clean user-facing output, optionally with sub-location."""
    country = result.get("final_country", "Unknown")
    label = result.get("confidence_label", "")
    confidence = result.get("confidence", 0.0)
    region = result.get("region_cluster", "")
    alternatives = result.get("alternatives", [])
    strategy = result.get("recommended_strategy", "")

    print()
    print("=" * 54)
    print(f"Final country guess: {country} ({label}, {int(confidence * 100)}%)")
    if region:
        print(f"Region cluster:      {region}")
    if alternatives:
        print(f"Alternatives:        {', '.join(alternatives)}")

    # ── Sub-location block ────────────────────────────────────────────────────
    if sub_location and sub_location.get("sub_location_attempted"):
        kb_used = sub_location.get("knowledge_base_used", False)
        broad_region = sub_location.get("broad_region")
        final_state = sub_location.get("final_state")
        state_conf = float(sub_location.get("state_confidence", 0.0) or 0.0)
        region_conf = float(sub_location.get("region_confidence", 0.0) or 0.0)
        candidates = sub_location.get("candidate_states", [])
        state_evidence = sub_location.get("state_evidence", [])

        print("-" * 54)
        if kb_used and broad_region:
            if final_state is None:
                # Gemini could not pick a state at all
                print(
                    f"State/Region:        region only -- {broad_region}"
                    f" ({int(region_conf * 100)}%)"
                )
                print("                     State could not be determined")
            elif state_conf >= 0.75:
                # High confidence — show cleanly
                print(
                    f"State/Region:        {final_state} ({int(state_conf * 100)}%)"
                    f" -- {broad_region}"
                )
                alts = [c for c in candidates if c != final_state][:2]
                if alts:
                    print(f"State alternatives:  {', '.join(alts)}")
                if state_evidence:
                    evidence_str = ", ".join(state_evidence[:3])
                    print(f"State evidence:      {evidence_str}")
            elif state_conf >= 0.50:
                # Moderate confidence — show with warning
                print(
                    f"State/Region:        {final_state} ({int(state_conf * 100)}%)"
                    f" -- {broad_region}"
                )
                print("                     ⚠ moderate confidence -- treat as best guess")
                alts = [c for c in candidates if c != final_state][:2]
                if alts:
                    print(f"State alternatives:  {', '.join(alts)}")
                if state_evidence:
                    evidence_str = ", ".join(state_evidence[:3])
                    print(f"State evidence:      {evidence_str}")
            else:
                # Low confidence — show candidates, not a single state
                cand_str = ", ".join(candidates[:3]) if candidates else "unknown"
                print(
                    f"State/Region:        uncertain between {cand_str}"
                )
                print("                     -- not reliable enough to act on")
                print(
                    f"                     ({broad_region}, {int(region_conf * 100)}% region confidence)"
                )
        elif not kb_used:
            print(f"State/Region:        Sub-location not available for {country}")
        else:
            print("State/Region:        Could not determine sub-location")
    elif sub_location and not sub_location.get("knowledge_base_used", True):
        # Knowledge base not available at all
        print("-" * 54)
        print(f"State/Region:        Sub-location not available for {country}")

    print("-" * 54)
    if strategy:
        print(f"Recommendation:      {strategy}")
    print("=" * 54)
    print()

    if SHOW_DECISION_DEBUG:
        debug = result.get("debug", {})
        print("[Debug] raw_grounded_country:", debug.get("raw_grounded_country"))
        print("[Debug] region_consistent_country:", debug.get("region_consistent_country"))
        print("[Debug] adaptive_country:", debug.get("adaptive_country"))
        print("[Debug] strongest_region:", debug.get("strongest_region"))
        print("[Debug] verified_confidence:", debug.get("verified_confidence"))
        print("[Debug] trust_level:", debug.get("trust_level"))
        print()
