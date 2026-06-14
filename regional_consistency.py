import json
import logging
import re
from pathlib import Path
from typing import Any


# Phase 10A (simplified): read Gemini-generated region fields from first_pass
# and grounded_result instead of doing independent region inference in Python.
# region_clusters.json is retained as a fallback for old runs that pre-date the
# region fields in first_pass.json.
ANALYSIS_DIR = Path("analysis")
GROUNDING_FILE = "grounded_result.json"
FIRST_PASS_FILE = "first_pass.json"
REGION_CLUSTERS_PATH = Path("region_clusters.json")
REGIONAL_CONSISTENCY_PATH = ANALYSIS_DIR / "regional_consistency.json"

logger = logging.getLogger("regional_consistency")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[RegionalConsistency] %(message)s"))
    logger.addHandler(handler)


def load_json_file(path: Path, default: Any) -> Any:
    """Load JSON with a safe default."""
    if not path.exists():
        logger.warning("Missing file: %s", path)
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        logger.warning("Malformed JSON in %s: %s", path, error)
        return default


def normalize_country(value: Any) -> str:
    """Normalize country labels for stable matching."""
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


# ---------------------------------------------------------------------------
# Fallback: region_clusters.json lookup (used only when first_pass candidates
# have no region fields, i.e. from a run before the prompt was updated).
# ---------------------------------------------------------------------------

def _load_cluster_lookup(path: Path = REGION_CLUSTERS_PATH) -> dict[str, str]:
    """Return a normalized-country → region dict from region_clusters.json."""
    clusters = load_json_file(path, {})
    if not isinstance(clusters, dict):
        return {}
    lookup: dict[str, str] = {}
    for region, countries in clusters.items():
        for country in (countries if isinstance(countries, list) else []):
            lookup[normalize_country(country)] = str(region)
    return lookup


def _region_from_clusters(country: str | None, lookup: dict[str, str]) -> str | None:
    if not country:
        return None
    return lookup.get(normalize_country(country))


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _candidates_have_region_fields(candidates: list[dict]) -> bool:
    """True if at least one candidate carries a non-None region field."""
    return any(c.get("region") for c in candidates if isinstance(c, dict))


def _infer_strongest_region_from_clusters(
    candidates: list[dict],
    lookup: dict[str, str],
) -> str | None:
    """Fallback: pick region with highest total first-pass confidence."""
    region_scores: dict[str, float] = {}
    for c in candidates:
        if not isinstance(c, dict):
            continue
        country = c.get("country", "")
        region = _region_from_clusters(country, lookup)
        if region:
            conf = float(c.get("confidence", 0.0) or 0.0)
            region_scores[region] = max(region_scores.get(region, 0.0), conf)
    return max(region_scores, key=region_scores.__getitem__) if region_scores else None


def save_regional_consistency(
    payload: dict[str, Any],
    output_path: Path = REGIONAL_CONSISTENCY_PATH,
) -> None:
    """Save regional_consistency.json."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved regional_consistency.json")


def run_regional_consistency(analysis_dir: Path = ANALYSIS_DIR) -> dict[str, Any]:
    """Determine the region-consistent country from Gemini-provided region data."""
    first_pass = load_json_file(analysis_dir / FIRST_PASS_FILE, {})
    grounded = load_json_file(analysis_dir / GROUNDING_FILE, {})

    candidates: list[dict] = first_pass.get("top_candidates", [])
    grounded_country: str | None = grounded.get("final_country")
    grounded_region_consistent: bool | None = grounded.get("region_consistent")

    # ------------------------------------------------------------------
    # Determine strongest_region
    # Prefer the value Gemini wrote into first_pass (reflects the image).
    # Fall back to region_clusters.json derivation for old runs.
    # ------------------------------------------------------------------
    use_cluster_fallback = False
    strongest_region: str | None = first_pass.get("strongest_region")

    # Build country → region map from candidates (Gemini-provided)
    country_regions: dict[str, str | None] = {}
    for c in candidates:
        if isinstance(c, dict) and c.get("country"):
            country_regions[c["country"]] = c.get("region") or None

    # Also include grounded country's region if it came back from grounded_result
    if grounded_country and grounded.get("region"):
        country_regions.setdefault(grounded_country, grounded["region"])

    if not strongest_region or not _candidates_have_region_fields(candidates):
        use_cluster_fallback = True
        cluster_lookup = _load_cluster_lookup()
        strongest_region = _infer_strongest_region_from_clusters(candidates, cluster_lookup)
        # Fill missing region fields from clusters
        for c in candidates:
            if isinstance(c, dict) and c.get("country") and not c.get("region"):
                country_regions[c["country"]] = _region_from_clusters(c["country"], cluster_lookup)
        if grounded_country and not country_regions.get(grounded_country):
            country_regions[grounded_country] = _region_from_clusters(grounded_country, cluster_lookup)
        logger.info(
            "Fallback: no Gemini region fields found, inferred strongest_region=%s from clusters",
            strongest_region,
        )
    else:
        logger.info("Using Gemini-provided strongest_region=%s", strongest_region)

    # ------------------------------------------------------------------
    # Build region_candidate_list: candidates whose region == strongest_region
    # ------------------------------------------------------------------
    region_candidate_list: list[str] = [
        c["country"]
        for c in candidates
        if isinstance(c, dict)
        and c.get("country")
        and country_regions.get(c["country"]) == strongest_region
    ]
    logger.info("Region candidate list for %s: %s", strongest_region, region_candidate_list)

    # ------------------------------------------------------------------
    # Decide region_consistent_country
    # ------------------------------------------------------------------
    override_triggered = False
    source: str

    # grounded_region_consistent == None means old run (field absent): treat
    # as "unknown" — apply region guard only if the grounded country is clearly
    # NOT in the strongest region.
    grounded_in_strongest_region = country_regions.get(grounded_country) == strongest_region

    if grounded_region_consistent is False:
        # Gemini explicitly flagged the grounded country as region-inconsistent.
        logger.info(
            "Grounded country %s flagged region_consistent=false",
            grounded_country,
        )
        # Find best candidate: region_consistent=True AND region==strongest_region
        region_true_candidates = [
            c for c in candidates
            if isinstance(c, dict)
            and c.get("region_consistent") is True
            and country_regions.get(c.get("country")) == strongest_region
        ]
        region_true_candidates.sort(
            key=lambda c: float(c.get("confidence", 0.0) or 0.0),
            reverse=True,
        )

        if region_true_candidates:
            region_consistent_country = region_true_candidates[0]["country"]
            override_triggered = region_consistent_country != grounded_country
            source = "first_pass_region_consistent_override"
            logger.info(
                "Override triggered: %s replaced by %s (region_consistent=true in strongest_region=%s)",
                grounded_country,
                region_consistent_country,
                strongest_region,
            )
        elif region_candidate_list:
            # Fall back to any candidate in strongest_region by confidence
            best = max(
                (c for c in candidates if c.get("country") in region_candidate_list),
                key=lambda c: float(c.get("confidence", 0.0) or 0.0),
                default=None,
            )
            region_consistent_country = best["country"] if best else grounded_country
            override_triggered = region_consistent_country != grounded_country
            source = "first_pass_region_fallback"
            logger.info(
                "No region_consistent=true candidate found; using best in region: %s",
                region_consistent_country,
            )
        else:
            # No candidate in strongest_region at all — keep grounded country
            region_consistent_country = grounded_country
            source = "no_region_candidates_keep_grounded"
            logger.info(
                "No candidates in %s; keeping grounded country %s",
                strongest_region,
                grounded_country,
            )

    elif grounded_region_consistent is None and not grounded_in_strongest_region and region_candidate_list:
        # Old run or ambiguous: grounded country is not in strongest_region but
        # there is a better region-aligned candidate — apply a soft override.
        best = max(
            (c for c in candidates if c.get("country") in region_candidate_list),
            key=lambda c: float(c.get("confidence", 0.0) or 0.0),
            default=None,
        )
        if best and float(best.get("confidence", 0.0) or 0.0) >= float(
            next(
                (c.get("confidence", 0.0) for c in candidates if c.get("country") == grounded_country),
                0.0,
            )
        ) * 0.85:
            region_consistent_country = best["country"]
            override_triggered = region_consistent_country != grounded_country
            source = "soft_region_override_no_gemini_flag"
            logger.info(
                "Soft override (no Gemini flag): %s -> %s (grounded not in %s)",
                grounded_country,
                region_consistent_country,
                strongest_region,
            )
        else:
            region_consistent_country = grounded_country
            source = "grounded_result_accepted"
            logger.info("No override: grounded %s accepted", grounded_country)

    else:
        # grounded_region_consistent is True, or it's None but grounded is
        # already in strongest_region — no override needed.
        region_consistent_country = grounded_country
        source = "grounded_result_consistent"
        logger.info(
            "No override needed: %s is region-consistent (flag=%s)",
            grounded_country,
            grounded_region_consistent,
        )

    payload: dict[str, Any] = {
        "status": "ok",
        "region_consistent_country": region_consistent_country,
        # selected_country kept as alias so main.py log and confidence_verifier
        # continue to work without changes.
        "selected_country": region_consistent_country,
        "strongest_region": strongest_region,
        "override_triggered": override_triggered,
        "region_candidate_list": region_candidate_list,
        "country_regions": country_regions,
        "source": source,
        "fallback_used": use_cluster_fallback,
        "grounded_region_consistent": grounded_region_consistent,
        "raw_grounded_country": grounded_country,
    }
    save_regional_consistency(payload, analysis_dir / "regional_consistency.json")
    return payload


def main() -> None:
    """Run regional consistency from the terminal."""
    print(json.dumps(run_regional_consistency(), indent=2))


if __name__ == "__main__":
    main()
