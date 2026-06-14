"""meta_knowledge_injector.py — GeoGuessr meta knowledge injection for Gemini prompts.

Loads knowledge_base/country_metas.json and formats country-specific visual
identifiers for injection into country_reasoner.py and grounded_reasoner.py.

Two injection modes:
  - First pass (country_reasoner): inject ALL known country metas as a
    'what to look for' guide, since no candidates exist yet.
  - Grounded comparison (grounded_reasoner): inject only the candidate
    countries, including pairwise elimination clues.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path

logger = logging.getLogger("meta_knowledge_injector")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[MetaKnowledge] %(message)s"))
    logger.addHandler(handler)

_KNOWLEDGE_BASE_PATH = Path("knowledge_base/country_metas.json")

# Module-level cache — loaded once per process.
_meta_db: dict | None = None


def _load_db() -> dict:
    """Load and cache country_metas.json.  Returns empty dict on failure."""
    global _meta_db
    if _meta_db is not None:
        return _meta_db
    if not _KNOWLEDGE_BASE_PATH.exists():
        logger.warning("country_metas.json not found at %s", _KNOWLEDGE_BASE_PATH)
        _meta_db = {}
        return _meta_db
    try:
        _meta_db = json.loads(_KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8"))
        logger.info(
            "Loaded country_metas.json v%s — %d countries",
            _meta_db.get("version", "?"),
            len(_meta_db.get("countries", {})),
        )
    except Exception as exc:
        logger.warning("Failed to load country_metas.json: %s", exc)
        _meta_db = {}
    return _meta_db


def _normalize_key(name: str) -> str:
    """Normalize a country display name to a JSON key.

    'South Korea'  → 'South_Korea'
    'new zealand'  → 'New_Zealand'
    'United_States'→ 'United_States'
    """
    nfkd = unicodedata.normalize("NFKD", str(name))
    ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")
    # Title-case each word, join with underscore
    words = re.split(r"[\s_]+", ascii_name.strip())
    return "_".join(w.capitalize() for w in words if w)


def _lookup(country: str) -> dict | None:
    """Return the meta entry for a country, trying both normalized and raw keys."""
    db = _load_db()
    countries = db.get("countries", {})
    if not countries:
        return None
    # Try normalized key first, then original, then title-case variants
    key = _normalize_key(country)
    return countries.get(key) or countries.get(country) or countries.get(country.title())


def get_country_hint(country: str) -> dict | None:
    """Return the full meta entry for a single country, or None if not in DB."""
    return _lookup(country)


def _format_single_country_block(name: str, entry: dict, include_tier2: bool = False) -> list[str]:
    """Format one country's identifiers as a list of text lines."""
    lines: list[str] = []
    display = name.replace("_", " ")
    lines.append(f"{display} — look for:")
    for item in entry.get("tier1_identifiers", []):
        lines.append(f"  - {item}")
    if include_tier2:
        for item in entry.get("tier2_identifiers", []):
            lines.append(f"  + {item}  (supporting evidence)")
    return lines


def _format_elimination_clues(
    candidate_keys: list[str],
    entries: dict[str, dict],
) -> list[str]:
    """Format elimination clues for pairs of candidates that appear together."""
    seen_pairs: set[frozenset] = set()
    lines: list[str] = []

    for key in candidate_keys:
        entry = entries.get(key)
        if not entry:
            continue
        elim = entry.get("elimination_clues", {})
        confusions = entry.get("common_confusions", [])

        for confusion in confusions:
            confusion_key = _normalize_key(confusion)
            if confusion_key not in candidate_keys:
                continue
            pair = frozenset([key, confusion_key])
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            # Look for the clue under the key that has it
            clue_key = f"vs_{confusion_key}"
            clue = elim.get(clue_key) or elim.get(f"vs_{confusion}")
            if not clue:
                # Try from the other side
                other_entry = entries.get(confusion_key, {})
                other_elim = other_entry.get("elimination_clues", {})
                clue = other_elim.get(f"vs_{key}") or other_elim.get(
                    f"vs_{key.replace('_', ' ')}"
                )

            if clue:
                a = key.replace("_", " ")
                b = confusion_key.replace("_", " ")
                lines.append(f"{a} vs {b}: {clue}")

    return lines


def get_meta_hint_for_prompt(
    candidate_countries: list[str],
    *,
    include_tier2: bool = False,
) -> str:
    """Build a formatted meta-knowledge block for Gemini prompt injection.

    If candidate_countries is empty, returns hints for ALL countries in the DB
    (used by country_reasoner where no candidates are known yet).

    If candidate_countries is non-empty, returns hints only for those countries
    plus elimination clues for any pairs that share common-confusion entries.

    Returns an empty string if no meta knowledge is available for any candidate.
    """
    db = _load_db()
    all_entries: dict[str, dict] = db.get("countries", {})
    if not all_entries:
        return ""

    all_db_keys = list(all_entries.keys())

    if not candidate_countries:
        # First-pass mode: show all countries as general guidance
        target_keys = all_db_keys
        mode = "all"
    else:
        target_keys = [_normalize_key(c) for c in candidate_countries]
        mode = "candidates"

    # Split into found / not found
    found: dict[str, dict] = {}
    missing: list[str] = []

    for i, key in enumerate(target_keys):
        entry = all_entries.get(key)
        if entry:
            found[key] = entry
        else:
            # Report original name for missing
            original = (
                candidate_countries[i]
                if mode == "candidates" and i < len(candidate_countries)
                else key.replace("_", " ")
            )
            missing.append(original)

    if mode == "candidates":
        if found:
            logger.info(
                "Injected meta hints for: %s",
                ", ".join(k.replace("_", " ") for k in found),
            )
        if missing:
            logger.info("No meta knowledge for: %s", ", ".join(missing))
    else:
        logger.info("Injecting general meta knowledge for all %d countries", len(found))

    if not found:
        return ""

    lines: list[str] = [
        "--- GEOGUESSR META KNOWLEDGE ---",
        "The following are highly reliable visual identifiers for these countries.",
        "Use these to confirm or eliminate candidates based on what you can see:",
        "",
        "COUNTRY-SPECIFIC VISUAL IDENTIFIERS TO CHECK:",
        "",
    ]

    for key, entry in found.items():
        lines.extend(_format_single_country_block(key, entry, include_tier2=include_tier2))
        lines.append("")

    # Elimination clues (only when we have specific candidates to compare)
    if mode == "candidates" and len(found) >= 2:
        elim_lines = _format_elimination_clues(list(found.keys()), found)
        if elim_lines:
            lines.append("ELIMINATION CLUES FOR CANDIDATE PAIRS:")
            for clue in elim_lines:
                lines.append(f"  {clue}")
            lines.append("")

    lines.append("--- END META KNOWLEDGE ---")
    return "\n".join(lines)
