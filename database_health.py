"""database_health.py — Reference database coverage scanner.

Runs once at pipeline startup to audit Country_Data/ and produce:
  analysis/database_health_report.json   — machine-readable coverage data
  reference_download_priority.txt        — manual download guide

Does NOT call Gemini and does NOT modify any reference files.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Config ────────────────────────────────────────────────────────────────────
COUNTRY_DATA_ROOT = Path("Country_Data")
GENERAL_INFO_ROOT = COUNTRY_DATA_ROOT / "General_info"
METAS_ROOT = COUNTRY_DATA_ROOT / "metas"
ANALYSIS_DIR = Path("analysis")
HEALTH_REPORT_PATH = ANALYSIS_DIR / "database_health_report.json"
PRIORITY_LIST_PATH = Path("reference_download_priority.txt")

# Canonical meta categories the pipeline uses, mapped to actual folder names.
CANONICAL_META_FOLDERS: dict[str, list[str]] = {
    "poles":         ["poles"],
    "road_markings": ["street_markings"],
    "vegetation":    ["flora"],
    "road_signs":    ["street_sign", "signposts"],
    "terrain":       [],           # no folder exists yet in any country
    "architecture":  [],           # no folder exists yet in any country
}
ALL_CANONICAL_METAS = list(CANONICAL_META_FOLDERS.keys())

# Meta priority order for the download list (highest impact first).
META_PRIORITY_ORDER = ["poles", "road_signs", "vegetation", "road_markings", "terrain", "architecture"]

# GeoGuessr frequency priority order (from spec).
GEOGUESSR_PRIORITY = [
    "Brazil", "India", "Japan", "Indonesia", "Nigeria", "South Africa",
    "Mexico", "Thailand", "Turkey", "Poland", "Romania", "Russia",
    "Colombia", "Peru", "Chile", "United States", "Canada", "Australia",
    "Argentina", "Ukraine", "France", "Germany", "Spain", "Portugal",
    "Italy", "Netherlands", "Sweden", "Norway", "Finland", "Denmark",
    "New Zealand", "Kenya", "Ghana", "Bangladesh", "Pakistan",
    "Mongolia", "Kyrgyzstan", "Kazakhstan", "Senegal", "Tunisia",
]

# Search hint templates per meta category.
SEARCH_HINTS: dict[str, str] = {
    "poles":         '"{country} utility poles street view"',
    "road_signs":    '"{country} road signs street view"',
    "vegetation":    '"{country} landscape vegetation typical"',
    "road_markings": '"{country} road markings pavement"',
    "terrain":       '"{country} rural terrain landscape"',
    "architecture":  '"{country} typical buildings architecture"',
}

IGNORED_NAMES = {"__pycache__", ".git", ".ds_store", "thumbs.db", "geometas_logo", "unknown"}

logger = logging.getLogger("database_health")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[Database] %(message)s"))
    logger.addHandler(handler)


# ── Normalization helpers ─────────────────────────────────────────────────────

def _norm(value: str) -> str:
    """Normalize to lowercase ASCII with underscores."""
    value = unicodedata.normalize("NFKD", str(value))
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


# Map common GeoGuessr name variants to metas folder normalized names.
_COUNTRY_NORM_OVERRIDES: dict[str, str] = {
    "united_states":         "united_states_of_america_usa",
    "usa":                   "united_states_of_america_usa",
    "united_kingdom":        "united_kingdom_uk",
    "uk":                    "united_kingdom_uk",
    "uae":                   "united_arab_emirates_uae",
    "united_arab_emirates":  "united_arab_emirates_uae",
    "south_korea":           "south_korea",
    "republic_of_korea":     "south_korea",
    "czechia":               "czech_republic",
}


def _norm_country(name: str) -> str:
    key = _norm(name)
    return _COUNTRY_NORM_OVERRIDES.get(key, key)


# ── Scanning helpers ──────────────────────────────────────────────────────────

def _scan_metas_countries() -> dict[str, dict[str, Any]]:
    """Return per-country coverage data from Country_Data/metas/."""
    result: dict[str, dict[str, Any]] = {}

    if not METAS_ROOT.exists():
        return result

    for country_dir in sorted(METAS_ROOT.iterdir()):
        if not country_dir.is_dir():
            continue
        if _norm(country_dir.name) in IGNORED_NAMES:
            continue

        present: list[str] = []
        missing: list[str] = []
        present_folders = {_norm(sub.name) for sub in country_dir.iterdir() if sub.is_dir()}

        for canonical, folder_aliases in CANONICAL_META_FOLDERS.items():
            if not folder_aliases:
                # No folder exists in the dataset yet for this canonical category.
                missing.append(canonical)
                continue
            if any(_norm(alias) in present_folders for alias in folder_aliases):
                present.append(canonical)
            else:
                missing.append(canonical)

        total = len(ALL_CANONICAL_METAS)
        score = round(len(present) / max(total, 1), 2)
        if score == 0.0:
            risk = "critical"
        elif score < 0.35:
            risk = "high"
        elif score < 0.67:
            risk = "medium"
        else:
            risk = "low"

        result[country_dir.name] = {
            "display_name": country_dir.name.replace("_", " "),
            "folder": str(country_dir),
            "has_general_info": False,  # filled in next step
            "meta_categories_present": present,
            "meta_categories_missing": missing,
            "coverage_score": score,
            "retrieval_risk": risk,
        }

    return result


def _scan_general_info_countries() -> set[str]:
    """Return normalized names of countries that have a General_info folder."""
    found: set[str] = set()
    if not GENERAL_INFO_ROOT.exists():
        return found
    for d in GENERAL_INFO_ROOT.iterdir():
        if d.is_dir() and _norm(d.name) not in IGNORED_NAMES:
            found.add(_norm(d.name))
    return found


# ── Main health check ─────────────────────────────────────────────────────────

def run_health_check(
    analysis_dir: Path = ANALYSIS_DIR,
    priority_list_path: Path = PRIORITY_LIST_PATH,
) -> dict[str, Any]:
    """Scan Country_Data/ and write database_health_report.json + priority list."""
    logger.info("Scanning Country_Data/ for reference coverage…")

    metas_coverage = _scan_metas_countries()
    general_info_countries = _scan_general_info_countries()

    # Merge general_info presence into coverage entries.
    for folder_name, data in metas_coverage.items():
        gi_key = _norm(folder_name)
        # Also check hyphen variants (General_info uses hyphen slugs).
        hyphen_key = gi_key.replace("_", "-")
        data["has_general_info"] = (
            gi_key in general_info_countries
            or hyphen_key in general_info_countries
        )

    # Countries in General_info that have no metas folder at all.
    metas_norm_keys = {_norm(k) for k in metas_coverage}
    gi_only_countries = general_info_countries - metas_norm_keys

    # Summary counts.
    total = len(metas_coverage)
    full_coverage = sum(
        1 for d in metas_coverage.values()
        if not d["meta_categories_missing"]
    )
    no_metas = sum(
        1 for d in metas_coverage.values()
        if not d["meta_categories_present"]
    )
    partial = total - full_coverage - no_metas

    high_risk = [
        d["display_name"]
        for d in metas_coverage.values()
        if d["retrieval_risk"] in {"critical", "high"}
    ]

    report: dict[str, Any] = {
        "total_countries_in_metas": total,
        "countries_with_full_coverage": full_coverage,
        "countries_with_partial_coverage": partial,
        "countries_with_no_metas": no_metas,
        "countries_in_general_info_only": len(gi_only_countries),
        "coverage_by_country": {
            folder_name: {
                "has_general_info": data["has_general_info"],
                "meta_categories_present": data["meta_categories_present"],
                "meta_categories_missing": data["meta_categories_missing"],
                "coverage_score": data["coverage_score"],
                "retrieval_risk": data["retrieval_risk"],
            }
            for folder_name, data in sorted(metas_coverage.items())
        },
        "high_risk_countries": high_risk,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "database_health_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    logger.info(
        "Health check: %d countries, %d full coverage, %d no metas. "
        "See analysis/database_health_report.json for details.",
        total, full_coverage, no_metas,
    )

    # Generate the priority download list.
    _write_priority_list(metas_coverage, priority_list_path)

    return report


# ── Priority list generation ──────────────────────────────────────────────────

def _write_priority_list(
    metas_coverage: dict[str, dict[str, Any]],
    output_path: Path = PRIORITY_LIST_PATH,
) -> None:
    """Write reference_download_priority.txt for manual collection."""

    # Build a lookup: normalized GeoGuessr name → metas coverage entry.
    def _find_coverage(geoguessr_name: str) -> dict[str, Any] | None:
        target = _norm_country(geoguessr_name)
        # Direct match.
        for folder_name, data in metas_coverage.items():
            if _norm(folder_name) == target:
                return data
        # Compact match (strip underscores).
        compact_target = target.replace("_", "")
        for folder_name, data in metas_coverage.items():
            if _norm(folder_name).replace("_", "") == compact_target:
                return data
        return None

    high_lines: list[str] = []
    medium_lines: list[str] = []
    counter = 1

    for country_name in GEOGUESSR_PRIORITY:
        coverage = _find_coverage(country_name)
        if coverage is None:
            # Country not in metas at all — all categories missing.
            for meta in META_PRIORITY_ORDER:
                hint = SEARCH_HINTS.get(meta, f'"{country_name} {meta}"').format(country=country_name)
                high_lines.append(
                    f"{counter}. {country_name}/{meta} — search: {hint}"
                )
                counter += 1
            continue

        missing = coverage["meta_categories_missing"]
        present = coverage["meta_categories_present"]

        # Prioritise by meta order.
        for meta in META_PRIORITY_ORDER:
            if meta not in missing:
                continue
            hint = SEARCH_HINTS.get(meta, f'"{country_name} {meta}"').format(country=country_name)
            line = f"{counter}. {country_name}/{meta} — search: {hint}"
            counter += 1
            if not present:  # nothing at all → high priority
                high_lines.append(line)
            else:            # partial → medium priority
                medium_lines.append(line)

    lines: list[str] = [
        "=== REFERENCE DOWNLOAD PRIORITY LIST ===",
        f"Generated: {datetime.now().strftime('%Y-%m-%d')}",
        "",
        "How to use this list:",
        "  1. Use the search hint to find representative Street View images.",
        "  2. Save images to Country_Data/metas/<Country>/<category>/",
        "  3. Name files descriptively — see retrieval_engine.py for naming guide.",
        "",
        "Naming convention reminder:",
        "  GOOD: Brazil_poles_wooden_rural_01.jpg",
        "  GOOD: India_road_signs_hindi_01.jpg",
        "  BAD:  img001.jpg  photo.jpg  reference1.jpg",
        "",
    ]

    if high_lines:
        lines.append("HIGH PRIORITY (country has no metas at all):")
        lines.extend(high_lines)
        lines.append("")

    if medium_lines:
        lines.append("MEDIUM PRIORITY (country has partial coverage — missing these metas):")
        lines.extend(medium_lines)
        lines.append("")

    lines.append(
        "NOTE: terrain and architecture folders do not exist in any country yet. "
        "Add Country_Data/metas/<Country>/terrain/ and /architecture/ manually."
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Priority list written to %s", output_path)


def main() -> None:
    """Run health check from the terminal."""
    report = run_health_check()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
