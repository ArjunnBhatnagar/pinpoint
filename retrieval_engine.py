"""retrieval_engine.py — Phase 8B reference retrieval package builder.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART C — REFERENCE IMAGE NAMING CONVENTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The retrieval ranker scores files by matching query terms derived from
visual observations and evidence tags against filenames and folder paths.
Descriptive filenames dramatically improve matching accuracy.

GOOD names — the ranker can extract and match these terms:
  Argentina_poles_wooden_rural_01.jpg
  Brazil_vegetation_cerrado_dry_season_01.jpg
  India_poles_concrete_urban_01.jpg
  Japan_poles_wooden_clustered_01.jpg
  Turkey_road_signs_directional_01.jpg
  USA_road_signs_interstate_shield_01.jpg
  Colombia_poles_thin_wooden_rural_01.jpg

BAD names — the ranker cannot extract useful terms:
  img001.jpg
  photo.jpg
  reference1.jpg
  DSC_4832.jpg
  IMG_20240315.jpg

Naming template:
  <Country>_<meta_category>_<descriptor1>_<descriptor2>_<NN>.jpg

Useful descriptor words (use as many as are accurate):
  Poles:         wooden, concrete, thin, thick, clustered, single, rural, urban
  Road signs:    directional, warning, route, highway, interstate, shield, bilingual
  Vegetation:    tropical, cerrado, dry, rainforest, savanna, pine, palm, scrub
  Road markings: dashed, solid, yellow, white, double, single, center
  Terrain:       flat, rolling, hilly, mountainous, steppe, desert, coastal

Store files in:
  Country_Data/metas/<Country>/<meta_category>/filename.jpg

Example:
  Country_Data/metas/Brazil/poles/Brazil_poles_wooden_rural_01.jpg
  Country_Data/metas/India/street_sign/India_road_signs_hindi_01.jpg
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

from meta_normalizer import (
    CANONICAL_RETRIEVAL_CATEGORIES,
    build_meta_normalization,
)


# Phase 8B: reference retrieval package builder.
# This module does not call Gemini and does not perform grounded comparison.
# It only reads first_pass.json and selects a bounded, relevant set of local
# reference files for the next phase.
ANALYSIS_DIR = Path("analysis")
FIRST_PASS_PATH = ANALYSIS_DIR / "first_pass.json"
RETRIEVAL_PACKAGE_PATH = ANALYSIS_DIR / "retrieval_package.json"

COUNTRY_DATA_ROOT = Path("Country_Data")
GENERAL_INFO_ROOT = COUNTRY_DATA_ROOT / "General_info"
METAS_ROOT = COUNTRY_DATA_ROOT / "metas"
REFERENCE_ROOTS = [COUNTRY_DATA_ROOT]

MAX_COUNTRIES = 5
MAX_IMAGES_PER_META = 2     # 2 per category × 13 categories = up to 26 per country (breadth over depth)
MAX_TEXT_FILES_PER_META = 2
MAX_GENERAL_INFO_FILES = 10
TEXT_PREVIEW_CHARS = 500
RANKING_TEXT_CHARS = 6000
MAX_RANKING_DEBUG_ITEMS = 50

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
TEXT_EXTENSIONS = {".txt", ".md", ".json", ".html", ".htm"}
IGNORED_NAMES = {"__pycache__", ".git", ".gitkeep", ".ds_store", "thumbs.db"}

# All 13 database categories attempted for every candidate country.
# Order determines retrieval priority within each country.
RETRIEVAL_CATEGORIES = [
    "bollards",        # bollards/       — 106 countries
    "poles",           # poles/          — 101 countries
    "traffic_lights",  # traffic_lights/ — 104 countries
    "terrain",         # terrain/        —  95 countries
    "street_sign",     # street_sign/    —  56 countries
    "license_plate",   # license_plate/  —  51 countries
    "language",        # language/       —  47 countries
    "flora",           # flora/          —  40 countries
    "architecture",    # architecture/   —  38 countries
    "road_surface",    # road_surface/   —  44 countries
    "street_markings", # street_markings/—  20 countries
    "street_name",     # street_name/    —  22 countries
    "signposts",       # signposts/      —  14 countries
]

# Maps canonical category names → actual folder names on disk.
# IMPORTANT: Empty lists meant "no folder" before. All folders now exist in the
# database and every entry must map to a real folder name.
META_CATEGORY_ALIASES = {
    # ── Direct 1-to-1 folder mappings (match exact disk names) ──────────────
    "bollards":        ["bollards"],
    "poles":           ["poles"],
    "traffic_lights":  ["traffic_lights"],
    "terrain":         ["terrain"],         # was [] — now maps correctly
    "street_sign":     ["street_sign"],
    "license_plate":   ["license_plate"],
    "language":        ["language"],
    "flora":           ["flora"],
    "architecture":    ["architecture"],    # was [] — now maps correctly
    "road_surface":    ["road_surface"],    # was [] — now maps correctly
    "street_markings": ["street_markings"],
    "street_name":     ["street_name"],
    "signposts":       ["signposts"],
    # ── Legacy canonical aliases (kept for backward compatibility) ────────────
    "poles_legacy":    ["poles"],
    "road_signs":      ["street_sign", "signposts", "street_name"],
    "road_markings":   ["street_markings"],
    "vegetation":      ["flora"],
    "license_plates":  ["license_plate"],
    "street_names":    ["street_name"],
    # ── Intentionally empty — no folder exists for these ────────────────────
    "vehicles":        [],
    "guardrails":      [],
    "road_edge_posts": [],
    "street_lights":   [],
}

COUNTRY_ALIASES = {
    "czech_republic": ["czechia"],
    "czechia": ["czech_republic"],
    "united_states": ["united_states_of_america_usa", "united_states"],
    "usa": ["united_states_of_america_usa", "united_states"],
    "united_states_of_america": ["united_states_of_america_usa", "united_states"],
    "united_kingdom": ["united_kingdom_uk", "united_kingdom"],
    "uk": ["united_kingdom_uk", "united_kingdom"],
    "uae": ["united_arab_emirates_uae", "united_arab_emirates"],
    "united_arab_emirates": [
        "united_arab_emirates_uae",
        "united_arab_emirates",
    ],
    "south_korea": ["south_korea"],
    "republic_of_korea": ["south_korea"],
    "north_korea": ["north_korea"],
}

logger = logging.getLogger("retrieval_engine")
logging.basicConfig(level=logging.INFO, format="[Retrieval] %(message)s")

ranker_logger = logging.getLogger("retrieval_ranker")
ranker_logger.setLevel(logging.INFO)
ranker_logger.propagate = False

if not ranker_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[RetrievalRanker] %(message)s"))
    ranker_logger.addHandler(handler)


RANKING_STOP_WORDS = {
    "and",
    "are",
    "basic",
    "beside",
    "buildings",
    "candidate",
    "climate",
    "common",
    "ground",
    "houses",
    "infrastructure",
    "likely",
    "meta",
    "multiple",
    "on",
    "or",
    "possible",
    "simple",
    "some",
    "style",
    "the",
    "trees",
    "visible",
    "with",
}


def load_first_pass(path: Path = FIRST_PASS_PATH) -> dict[str, Any]:
    """Load first_pass.json with graceful fallback."""
    logger.info("Loading first_pass.json")

    if not path.exists():
        return {
            "top_candidates": [],
            "detected_meta_categories": [],
            "strongest_evidence": [],
            "OCR_analysis": {},
            "_warnings": [f"Missing first_pass file: {path}"],
        }

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        return {
            "top_candidates": [],
            "detected_meta_categories": [],
            "strongest_evidence": [],
            "OCR_analysis": {},
            "_warnings": [f"Malformed first_pass file: {error}"],
        }


def normalize_name(value: str) -> str:
    """Normalize country, region, and folder names into safe comparable forms."""
    value = unicodedata.normalize("NFKD", str(value))
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower()
    value = re.sub(r"['’`]", "", value)
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def country_lookup_keys(country_name: str) -> list[str]:
    """Return normalized country names plus known dataset aliases."""
    normalized = normalize_name(country_name)
    keys = [normalized]

    for alias in COUNTRY_ALIASES.get(normalized, []):
        alias_key = normalize_name(alias)
        if alias_key and alias_key not in keys:
            keys.append(alias_key)

    return keys


def is_supported_file(path: Path) -> bool:
    """Ignore empty, cache, system, and unsupported files."""
    if not path.is_file():
        return False

    if path.name.lower() in IGNORED_NAMES or path.name.startswith("."):
        return False

    if path.suffix.lower() not in IMAGE_EXTENSIONS | TEXT_EXTENSIONS:
        return False

    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def discover_country_folders(reference_roots: list[Path] | None = None) -> dict[str, list[Path]]:
    """Discover country folders only from Country_Data/General_info and /metas.

    General_info uses lowercase hyphen slugs such as ``south-korea``.
    metas uses capitalized underscore slugs such as ``South_Korea`` and can
    include suffixes like ``United_States_of_America_USA``. Normalization keeps
    both layouts comparable without loading unrelated reference folders.
    """
    folders: dict[str, list[Path]] = {}
    roots = [GENERAL_INFO_ROOT, METAS_ROOT]

    for root in roots:
        if not root.exists():
            logger.warning("Country_Data root missing: %s", root)
            continue

        for directory in root.iterdir():
            if not directory.is_dir() or directory.name.lower() in IGNORED_NAMES:
                continue

            normalized = normalize_name(directory.name)
            if normalized:
                folders.setdefault(normalized, []).append(directory)

    return folders


def match_country_folder(country_name: str, country_folders: dict[str, list[Path]]) -> Path | None:
    """Match first-pass country names to discovered local folders."""
    matched_folders = match_country_folders(country_name, country_folders)
    return matched_folders[0] if matched_folders else None


def match_country_folders(
    country_name: str,
    country_folders: dict[str, list[Path]],
) -> list[Path]:
    """Return folders that exactly match a first-pass country or known alias.

    This intentionally avoids broad substring fallback. If Portugal is missing,
    the retriever must report Portugal as missing rather than substituting an
    alphabetically nearby or partially matching country.
    """
    lookup_keys = country_lookup_keys(country_name)

    matches = []

    for key in lookup_keys:
        if key in country_folders:
            matches.extend(country_folders[key])

    if matches:
        return sorted(set(matches), key=lambda path: str(path).lower())

    for key in lookup_keys:
        compact_country = key.replace("_", "")
        for folder_name, folder_paths in country_folders.items():
            compact_folder = folder_name.replace("_", "")
            if compact_country == compact_folder:
                matches.extend(folder_paths)

    if matches:
        return sorted(set(matches), key=lambda path: str(path).lower())

    return []


def normalize_meta_category(meta_category: str) -> list[str]:
    """Map canonical retrieval intent to real dataset folder names."""
    normalized = normalize_name(meta_category)
    aliases = META_CATEGORY_ALIASES.get(normalized, [])
    return [normalize_name(alias) for alias in aliases]


def discover_available_meta_categories(root: Path = METAS_ROOT) -> list[str]:
    """Discover the actual category vocabulary present in Country_Data/metas."""
    categories = set()

    if not root.exists():
        return []

    for country_folder in root.iterdir():
        if not country_folder.is_dir():
            continue

        for category_folder in country_folder.iterdir():
            if category_folder.is_dir():
                categories.add(normalize_name(category_folder.name))

    return sorted(categories)


def resolve_dataset_meta_categories(
    meta_categories: list[str],
    available_categories: list[str],
) -> tuple[dict[str, list[str]], list[str]]:
    """Resolve canonical intent into folder names that exist in the dataset."""
    available = set(available_categories)
    resolved = {}
    unsupported = []

    for meta_category in meta_categories:
        folder_names = [
            folder_name
            for folder_name in normalize_meta_category(meta_category)
            if folder_name in available
        ]
        if folder_names:
            resolved[meta_category] = folder_names
            logger.info(
                "Resolved dataset folders: %s -> %s",
                meta_category,
                ", ".join(folder_names),
            )
        else:
            unsupported.append(meta_category)
            logger.info(
                "No dataset folder mapping for canonical meta: %s",
                meta_category,
            )

    return resolved, unsupported


def discover_meta_folders(country_folder: Path, meta_category: str) -> list[Path]:
    """Find meta folders under a matched country/region folder."""
    possible_names = set(normalize_meta_category(meta_category))
    matches = []

    if not country_folder.exists():
        return matches

    if normalize_name(country_folder.name) in possible_names:
        matches.append(country_folder)

    if normalize_name(country_folder.parent.name) in possible_names:
        matches.append(country_folder)

    for directory in country_folder.rglob("*"):
        if not directory.is_dir():
            continue

        normalized = normalize_name(directory.name)
        if normalized in possible_names:
            matches.append(directory)

    return sorted(set(matches), key=lambda path: str(path).lower())


def discover_meta_folders_from_country_folders(
    country_folders: list[Path],
    meta_category: str,
) -> list[Path]:
    """Find meta folders across all matched folders for one country."""
    matches = []

    for folder in country_folders:
        matches.extend(discover_meta_folders(folder, meta_category))

    return sorted(set(matches), key=lambda path: str(path).lower())


def read_text_preview(path: Path, max_chars: int = TEXT_PREVIEW_CHARS) -> str:
    """Read a short text preview safely."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return ""

    if path.suffix.lower() in {".html", ".htm"}:
        text = re.sub(r"<script\b[^<]*(?:(?!</script>)<[^<]*)*</script>", " ", text, flags=re.I)
        text = re.sub(r"<style\b[^<]*(?:(?!</style>)<[^<]*)*</style>", " ", text, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", text)

    text = re.sub(r"\s+", " ", text)
    return text[:max_chars]


def extract_query_terms(values: list[str]) -> list[str]:
    """Extract stable, useful words for deterministic reference ranking."""
    terms = []

    for value in values:
        normalized = normalize_name(value)
        for term in normalized.split("_"):
            if (
                len(term) >= 3
                and term not in RANKING_STOP_WORDS
                and term not in terms
            ):
                terms.append(term)

    return terms


def build_ranking_context(
    first_pass: dict[str, Any],
    meta_normalization: dict[str, Any],
) -> dict[str, Any]:
    """Build weighted query terms from current observations and evidence tags."""
    visual_observations = meta_normalization.get("visual_observations", [])
    evidence_tags = meta_normalization.get("evidence_tags", [])
    canonical_categories = meta_normalization.get(
        "canonical_retrieval_categories",
        [],
    )
    strongest_evidence = first_pass.get("strongest_evidence", [])
    term_weights: dict[str, float] = {}

    weighted_sources = [
        (visual_observations, 3.0),
        (evidence_tags, 2.5),
        (canonical_categories, 1.5),
        (strongest_evidence, 1.0),
    ]
    for values, weight in weighted_sources:
        for term in extract_query_terms([str(value) for value in values]):
            term_weights[term] = max(term_weights.get(term, 0.0), weight)

    query_terms = sorted(term_weights)
    ranker_logger.info("Query terms: %s", ", ".join(query_terms) or "none")
    return {
        "query_terms": query_terms,
        "term_weights": term_weights,
        "visual_observations_used": visual_observations,
        "evidence_tags_used": evidence_tags,
        "canonical_categories_used": canonical_categories,
    }


def get_paired_text(path: Path) -> tuple[str, Path | None]:
    """Read a nearby same-stem text description when one exists."""
    for extension in sorted(TEXT_EXTENSIONS):
        paired_path = path.with_suffix(extension)
        if paired_path.exists() and paired_path.is_file():
            return read_text_preview(paired_path, RANKING_TEXT_CHARS), paired_path

    return "", None


def score_reference_file(
    path: Path,
    meta_category: str,
    ranking_context: dict[str, Any],
    country_confidence: float = 0.0,
    description_text: str = "",
) -> dict[str, Any]:
    """Score one local reference using filename, path, text, and intent."""
    term_weights = ranking_context.get("term_weights", {})
    filename_text = normalize_name(path.stem)
    path_text = normalize_name(str(path.parent))
    description = normalize_name(description_text)
    matched_terms = []
    filename_matches = []
    path_matches = []
    text_matches = []
    score = 3.0

    for term, weight in term_weights.items():
        matched = False
        if term in filename_text:
            score += 1.5 * float(weight)
            filename_matches.append(term)
            matched = True
        if term in path_text:
            score += 0.5 * float(weight)
            path_matches.append(term)
            matched = True
        if description and term in description:
            score += 1.0 * float(weight)
            text_matches.append(term)
            matched = True
        if matched:
            matched_terms.append(term)

    try:
        parsed_confidence = max(0.0, min(1.0, float(country_confidence or 0.0)))
    except (TypeError, ValueError):
        parsed_confidence = 0.0

    score += parsed_confidence * 2.0
    matched_terms = sorted(set(matched_terms))
    reasons = [f"exact meta category: {meta_category}"]

    if filename_matches:
        reasons.append("filename matched: " + ", ".join(sorted(set(filename_matches))))
    if text_matches:
        reasons.append("description matched: " + ", ".join(sorted(set(text_matches))))
    if path_matches and not filename_matches:
        reasons.append("folder path matched: " + ", ".join(sorted(set(path_matches))))
    if parsed_confidence:
        reasons.append(f"candidate confidence: {parsed_confidence:.2f}")

    return {
        "reference_relevance_score": round(score, 3),
        "matched_terms": matched_terms,
        "selection_reason": "; ".join(reasons),
    }


def rank_reference_items(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Rank scored references deterministically and apply the existing limit."""
    return sorted(
        items,
        key=lambda item: (
            -float(item.get("reference_relevance_score", 0.0) or 0.0),
            str(item.get("path", "")).lower(),
        ),
    )[:limit]


def summarize_ranked_items(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Keep ranking diagnostics compact while preserving selection rationale."""
    return [
        {
            "country": item.get("country"),
            "path": item.get("path"),
            "filename": item.get("filename"),
            "meta_category": item.get("meta_category"),
            "reference_relevance_score": item.get("reference_relevance_score"),
            "matched_terms": item.get("matched_terms", []),
            "selection_reason": item.get("selection_reason", ""),
        }
        for item in rank_reference_items(items, limit)
    ]


def collect_files(
    root: Path,
    extensions: set[str],
    limit: int | None = None,
) -> list[Path]:
    """Collect supported files below root with a hard limit."""
    files = []

    if not root.exists():
        return files

    for path in sorted(root.rglob("*"), key=lambda item: str(item).lower()):
        if limit is not None and len(files) >= limit:
            break

        if is_supported_file(path) and path.suffix.lower() in extensions:
            files.append(path)

    return files


def retrieve_general_info(
    country_folder: Path | list[Path],
    ranking_context: dict[str, Any] | None = None,
    country_name: str = "",
    country_confidence: float = 0.0,
    ranking_debug: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve the most relevant bounded general country info text files."""
    country_folders = country_folder if isinstance(country_folder, list) else [country_folder]
    candidate_roots = [
        folder
        for folder in country_folders
        if GENERAL_INFO_ROOT in folder.parents or folder == GENERAL_INFO_ROOT
    ]

    results = []
    seen = set()

    try:
        for root in candidate_roots:
            for path in collect_files(root, TEXT_EXTENSIONS):
                resolved = str(path.resolve())
                if resolved in seen:
                    continue

                seen.add(resolved)
                ranking_text = read_text_preview(path, RANKING_TEXT_CHARS)
                if not ranking_text:
                    continue

                score = score_reference_file(
                    path,
                    "general_info",
                    ranking_context or {},
                    country_confidence,
                    ranking_text,
                )
                results.append(
                    {
                        "path": str(path),
                        "filename": path.name,
                        "preview": ranking_text[:TEXT_PREVIEW_CHARS],
                        "country": country_name,
                        "meta_category": "general_info",
                        **score,
                    }
                )

        selected = rank_reference_items(results, MAX_GENERAL_INFO_FILES)
        if ranking_debug is not None:
            ranking_debug.extend(selected)
        return selected
    except Exception as error:
        ranker_logger.warning("General info ranking failed; using fallback: %s", error)
        fallback = []
        for root in candidate_roots:
            for path in collect_files(root, TEXT_EXTENSIONS, MAX_GENERAL_INFO_FILES):
                preview = read_text_preview(path)
                if preview:
                    fallback.append(
                        {
                            "path": str(path),
                            "filename": path.name,
                            "preview": preview,
                            "country": country_name,
                            "meta_category": "general_info",
                            "reference_relevance_score": 0.0,
                            "matched_terms": [],
                            "selection_reason": "ranking fallback: alphabetical order",
                        }
                    )
        return fallback[:MAX_GENERAL_INFO_FILES]


def retrieve_meta_references(
    country_folder: Path | list[Path],
    meta_category: str,
    ranking_context: dict[str, Any] | None = None,
    country_name: str = "",
    country_confidence: float = 0.0,
    ranking_debug: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Retrieve ranked images and texts for a single canonical meta category."""
    all_country_folders = country_folder if isinstance(country_folder, list) else [country_folder]
    country_folders = [
        folder
        for folder in all_country_folders
        if METAS_ROOT in folder.parents or folder == METAS_ROOT
    ]
    meta_folders = discover_meta_folders_from_country_folders(
        country_folders,
        meta_category,
    )
    source_category = normalize_name(meta_category)
    images = []
    texts = []
    seen_images = set()
    seen_texts = set()

    try:
        for folder in meta_folders:
            for image_path in collect_files(folder, IMAGE_EXTENSIONS):
                resolved = str(image_path.resolve())
                if resolved in seen_images:
                    continue

                seen_images.add(resolved)
                paired_text, paired_path = get_paired_text(image_path)
                score = score_reference_file(
                    image_path,
                    meta_category,
                    ranking_context or {},
                    country_confidence,
                    paired_text,
                )
                image_item = {
                    "path": str(image_path),
                    "filename": image_path.name,
                    "country": country_name,
                    "source_category": folder.name,
                    "meta_category": meta_category,
                    "paired_text_path": str(paired_path) if paired_path else None,
                    **score,
                }
                images.append(image_item)
                ranker_logger.info(
                    "Scored %s/%s/%s = %.3f",
                    country_name or "unknown",
                    meta_category,
                    image_path.name,
                    image_item["reference_relevance_score"],
                )

            for text_path in collect_files(folder, TEXT_EXTENSIONS):
                resolved = str(text_path.resolve())
                if resolved in seen_texts:
                    continue

                preview = read_text_preview(text_path)
                if not preview:
                    continue

                seen_texts.add(resolved)
                score = score_reference_file(
                    text_path,
                    meta_category,
                    ranking_context or {},
                    country_confidence,
                    preview,
                )
                texts.append(
                    {
                        "path": str(text_path),
                        "filename": text_path.name,
                        "country": country_name,
                        "preview": preview,
                        "source_category": folder.name,
                        "meta_category": meta_category,
                        **score,
                    }
                )

        images = rank_reference_items(images, MAX_IMAGES_PER_META)
        texts = rank_reference_items(texts, MAX_TEXT_FILES_PER_META)
        if ranking_debug is not None:
            ranking_debug.extend(images)
            ranking_debug.extend(texts)
        ranker_logger.info(
            "Selected top %s images and %s texts for %s/%s",
            len(images),
            len(texts),
            country_name or "unknown",
            meta_category,
        )
        matched_terms = sorted(
            {
                term
                for item in [*images, *texts]
                for term in item.get("matched_terms", [])
            }
        )
        ranker_logger.info("Matched terms: %s", ", ".join(matched_terms) or "none")
    except Exception as error:
        ranker_logger.warning(
            "Reference ranking failed for %s/%s; using fallback: %s",
            country_name or "unknown",
            meta_category,
            error,
        )
        images = []
        texts = []
        for folder in meta_folders:
            for image_path in collect_files(folder, IMAGE_EXTENSIONS, MAX_IMAGES_PER_META):
                images.append(
                    {
                        "path": str(image_path),
                        "filename": image_path.name,
                        "country": country_name,
                        "source_category": folder.name,
                        "meta_category": meta_category,
                        "reference_relevance_score": 0.0,
                        "matched_terms": [],
                        "selection_reason": "ranking fallback: alphabetical order",
                    }
                )
            for text_path in collect_files(folder, TEXT_EXTENSIONS, MAX_TEXT_FILES_PER_META):
                preview = read_text_preview(text_path)
                if preview:
                    texts.append(
                        {
                            "path": str(text_path),
                            "filename": text_path.name,
                            "country": country_name,
                            "preview": preview,
                            "source_category": folder.name,
                            "meta_category": meta_category,
                            "reference_relevance_score": 0.0,
                            "matched_terms": [],
                            "selection_reason": "ranking fallback: alphabetical order",
                        }
                    )
        images = images[:MAX_IMAGES_PER_META]
        texts = texts[:MAX_TEXT_FILES_PER_META]

    return {
        source_category: {
            "images": images,
            "texts": texts,
        }
    }


def get_candidate_countries(first_pass: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract top candidate countries with confidence from first_pass."""
    candidates = []

    for item in first_pass.get("top_candidates", [])[:MAX_COUNTRIES]:
        if not isinstance(item, dict):
            continue

        country = item.get("country")
        if not country:
            continue

        candidates.append(
            {
                "country": str(country),
                "normalized": normalize_name(str(country)),
                "confidence": item.get("confidence"),
                "reasons": item.get("reasons", []),
            }
        )

    return candidates


def get_detected_meta_categories(first_pass: dict[str, Any]) -> list[str]:
    """Return canonical folder lookup categories only."""
    normalization = build_meta_normalization(first_pass)
    return normalization["canonical_retrieval_categories"]


def extract_ocr_clues(first_pass: dict[str, Any]) -> list[str]:
    """Extract high-value OCR clues from first_pass."""
    ocr_analysis = first_pass.get("OCR_analysis", {})
    clues = []

    if isinstance(ocr_analysis, dict):
        for key in ["high_value_clues", "language_patterns", "weak_or_ignored_clues"]:
            value = ocr_analysis.get(key, [])
            if isinstance(value, list):
                clues.extend(str(item) for item in value)

        merged_text = ocr_analysis.get("merged_text")
        if merged_text:
            clues.append(str(merged_text))

    return clues


def validate_retrieved_countries(
    retrieved: dict[str, Any],
    allowed_country_keys: set[str],
) -> tuple[dict[str, Any], list[str]]:
    """Remove any country not explicitly listed in first_pass candidates."""
    validated = {}
    unexpected = []

    for country_key, payload in retrieved.items():
        if country_key not in allowed_country_keys:
            unexpected.append(country_key)
            logger.warning("Skipping unexpected country not in first_pass: %s", country_key)
            continue

        validated[country_key] = payload

    return validated, unexpected


def build_retrieval_package(
    first_pass_path: Path = FIRST_PASS_PATH,
    output_path: Path = RETRIEVAL_PACKAGE_PATH,
) -> dict[str, Any]:
    """Build the retrieval package for later grounded comparison."""
    first_pass = load_first_pass(first_pass_path)
    warnings = list(first_pass.get("_warnings", []))
    candidate_countries = get_candidate_countries(first_pass)
    candidate_country_names = [item["country"] for item in candidate_countries]
    allowed_country_keys = {item["normalized"] for item in candidate_countries}
    meta_normalization = build_meta_normalization(first_pass)
    detected_metas = meta_normalization["canonical_retrieval_categories"]
    available_meta_folders = discover_available_meta_categories()
    resolved_meta_folders, unsupported_meta_categories = resolve_dataset_meta_categories(
        detected_metas,
        available_meta_folders,
    )
    retrieval_metas = list(resolved_meta_folders)
    meta_normalization["dataset_available_meta_folders"] = available_meta_folders
    meta_normalization["resolved_dataset_folders"] = resolved_meta_folders
    meta_normalization["unsupported_retrieval_categories"] = unsupported_meta_categories
    strongest_evidence = first_pass.get("strongest_evidence", [])
    ocr_clues = extract_ocr_clues(first_pass)
    ranking_context = build_ranking_context(first_pass, meta_normalization)
    ranking_debug_items: list[dict[str, Any]] = []

    logger.info(
        "Candidate countries from first_pass.json: %s",
        ", ".join(candidate_country_names) or "none",
    )
    logger.info("Canonical retrieval categories: %s", ", ".join(detected_metas) or "none")
    logger.info(
        "Dataset-backed retrieval categories: %s",
        ", ".join(retrieval_metas) or "none",
    )
    logger.info(
        "Evidence tags kept for reasoning: %s",
        ", ".join(meta_normalization["evidence_tags"]) or "none",
    )
    for skipped_meta in meta_normalization["skipped_non_retrieval_metas"]:
        logger.info("Skipping non-canonical meta phrase: %s", skipped_meta)

    country_folders = discover_country_folders()
    retrieved: dict[str, Any] = {}
    countries_actually_retrieved = []
    country_folder_matching_debug: dict[str, Any] = {}
    missing_countries = []
    missing_meta_categories = []
    total_images = 0
    total_texts = 0

    if not candidate_countries:
        warnings.append("No candidate countries found in first_pass.json.")

    for candidate in candidate_countries:
        country_name = candidate["country"]
        country_key = candidate["normalized"]
        lookup_keys = country_lookup_keys(country_name)
        matched_folders = match_country_folders(country_name, country_folders)
        country_folder_matching_debug[country_name] = {
            "candidate_key": country_key,
            "lookup_keys": lookup_keys,
            "matched_folders": [str(path) for path in matched_folders],
        }

        if not matched_folders:
            logger.warning(
                "WARNING: No references found for %s. "
                "Grounded reasoning will be visual-only for this candidate.",
                country_name,
            )
            missing_countries.append(country_name)
            warnings.append(f"No country folder found for {country_name}.")
            visual_note = build_visual_only_note(country_name)
            retrieved[country_key] = {
                "matched_folder": None,
                "general_info": [],
                "metas": {},
                "reference_coverage": "none",
                "visual_only_note": visual_note,
            }
            # Inject the note into the candidate dict so it appears in the Gemini prompt.
            for cand in candidate_countries:
                if cand["country"] == country_name:
                    cand["visual_only_note"] = visual_note
                    break
            continue

        logger.info(
            "Matched %s -> %s",
            country_name,
            ", ".join(str(path) for path in matched_folders),
        )
        general_info = retrieve_general_info(
            matched_folders,
            ranking_context,
            country_name,
            candidate.get("confidence", 0.0),
            ranking_debug_items,
        )
        total_texts += len(general_info)
        countries_actually_retrieved.append(country_name)

        country_payload: dict[str, Any] = {
            "matched_folder": [str(path) for path in matched_folders],
            "general_info": general_info,
            "metas": {},
            "reference_coverage": "partial",  # updated after meta loop
        }
        country_missing_metas: list[str] = []
        # Per-category image counts for retrieval_coverage report
        category_image_counts: dict[str, int] = {}

        # Always attempt ALL 13 categories for every candidate country.
        # This replaces the old approach of only attempting what Gemini detected.
        for meta in RETRIEVAL_CATEGORIES:
            meta_payload = retrieve_meta_references(
                matched_folders,
                meta,
                ranking_context,
                country_name,
                candidate.get("confidence", 0.0),
                ranking_debug_items,
            )
            meta_key = next(iter(meta_payload.keys()))

            image_count = len(meta_payload[meta_key]["images"])
            text_count = len(meta_payload[meta_key]["texts"])
            total_images += image_count
            total_texts += text_count
            category_image_counts[meta] = image_count

            if image_count == 0 and text_count == 0:
                missing_meta_categories.append(
                    {"country": country_name, "meta": meta}
                )
                country_missing_metas.append(meta)
                logger.info("Missing meta folder: %s/%s", country_name, meta)
            else:
                country_payload["metas"].update(meta_payload)
                logger.info(
                    "Retrieved %s images and %s texts for %s/%s",
                    image_count,
                    text_count,
                    country_name,
                    meta_key,
                )

        # Build per-country retrieval coverage report
        cats_found = sum(1 for n in category_image_counts.values() if n > 0)
        coverage_score = round(cats_found / max(len(RETRIEVAL_CATEGORIES), 1), 2)
        country_payload["retrieval_coverage"] = {
            **category_image_counts,
            "total": sum(category_image_counts.values()),
            "coverage_score": coverage_score,
        }
        logger.info(
            "%s retrieval coverage: %d/%d categories, %d images total",
            country_name,
            cats_found,
            len(RETRIEVAL_CATEGORIES),
            sum(category_image_counts.values()),
        )

        # Finalise coverage flag and emit warning when effectively visual-only.
        coverage = compute_reference_coverage(
            general_info,
            country_payload["metas"],
            country_missing_metas,
            len(RETRIEVAL_CATEGORIES),
        )
        country_payload["reference_coverage"] = coverage
        if coverage == "none":
            visual_note = build_visual_only_note(country_name)
            country_payload["visual_only_note"] = visual_note
            logger.warning(
                "WARNING: No references found for %s. "
                "Grounded reasoning will be visual-only for this candidate.",
                country_name,
            )
            for cand in candidate_countries:
                if cand["country"] == country_name:
                    cand["visual_only_note"] = visual_note
                    break
        elif coverage == "partial":
            logger.info(
                "%s has partial coverage — missing metas: %s",
                country_name,
                ", ".join(country_missing_metas),
            )

        retrieved[country_key] = country_payload

    retrieved, unexpected_countries_skipped = validate_retrieved_countries(
        retrieved,
        allowed_country_keys,
    )

    package = {
        "source_first_pass": str(first_pass_path),
        "candidate_countries": candidate_countries,
        "candidate_countries_from_first_pass": candidate_country_names,
        "countries_actually_retrieved": countries_actually_retrieved,
        "unexpected_countries_skipped": unexpected_countries_skipped,
        "country_folder_matching_debug": country_folder_matching_debug,
        "detected_meta_categories": detected_metas,
        "retrieval_meta_categories_used": RETRIEVAL_CATEGORIES,  # all 13 always attempted
        "retrieval_categories_attempted": RETRIEVAL_CATEGORIES,
        "retrieval_coverage": {
            country: data.get("retrieval_coverage", {})
            for country, data in retrieved.items()
        },
        "meta_normalization": meta_normalization,
        "strongest_evidence": strongest_evidence,
        "ocr_clues": ocr_clues,
        "retrieval_ranking_debug": {
            "ranking_enabled": True,
            "query_terms": ranking_context["query_terms"],
            "visual_observations_used": ranking_context["visual_observations_used"],
            "evidence_tags_used": ranking_context["evidence_tags_used"],
            "top_reference_scores": summarize_ranked_items(
                ranking_debug_items,
                MAX_RANKING_DEBUG_ITEMS,
            ),
        },
        "retrieved": retrieved,
        "summary": {
            "countries_processed": len(candidate_countries),
            "total_images": total_images,
            "total_texts": total_texts,
            "missing_countries": missing_countries,
            "missing_meta_categories": missing_meta_categories,
            "warnings": warnings,
        },
    }

    save_retrieval_package(package, output_path)
    return package


def compute_reference_coverage(
    general_info: list[dict],
    metas: dict[str, Any],
    missing_metas: list[str],
    total_retrieval_metas: int,
) -> str:
    """Return 'none' | 'partial' | 'full' for a single candidate country."""
    has_general = bool(general_info)
    has_any_meta = bool(metas)
    if not has_general and not has_any_meta:
        return "none"
    if missing_metas and total_retrieval_metas > 0:
        return "partial"
    return "full"


def build_visual_only_note(country_name: str) -> str:
    """Return the grounded-reasoner prompt note for a zero-reference country."""
    return (
        f"Note: No reference images or texts are available for {country_name}. "
        f"Base your assessment of {country_name} on visual evidence only. "
        f"Do not penalize {country_name} for lack of a reference match."
    )


def save_retrieval_package(
    package: dict[str, Any],
    output_path: Path = RETRIEVAL_PACKAGE_PATH,
) -> None:
    """Save retrieval_package.json."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(package, indent=2), encoding="utf-8")
    logger.info("Saved retrieval_package.json")


def main() -> None:
    """Run retrieval package generation from the terminal."""
    package = build_retrieval_package()
    print(json.dumps(package, indent=2))


if __name__ == "__main__":
    main()
