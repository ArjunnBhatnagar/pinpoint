import logging
import re
import unicodedata
from typing import Any


# Phase 9D: retrieval intent normalization.
# Visual observations describe what the model saw. Canonical retrieval
# categories describe which curated reference folders may be queried.
CANONICAL_RETRIEVAL_CATEGORIES = {
    # Original categories
    "poles",
    "road_signs",
    "signposts",
    "road_markings",
    "road_surface",
    "street_markings",
    "architecture",
    "vegetation",
    "terrain",
    "license_plates",
    "vehicles",
    "guardrails",
    "bollards",
    "road_edge_posts",
    "street_lights",
    "language",
    "street_names",
    # Direct folder-name aliases (match actual disk folder names)
    "traffic_lights",
    "street_sign",
    "street_name",
    "license_plate",
    "flora",
}

EXACT_META_MAPPINGS = {
    # ── Poles ────────────────────────────────────────────────────────────────
    "wooden_utility_poles": "poles",
    "concrete_utility_poles": "poles",
    "utility_poles_candidate": "poles",
    "utility_poles": "poles",
    # ── Road markings ────────────────────────────────────────────────────────
    "road_surface_markings": "road_markings",
    "lane_markings": "road_markings",
    "road_paint": "road_markings",
    "street_markings": "street_markings",
    "road_marking": "street_markings",
    "road_markings": "street_markings",
    "lane_marking": "street_markings",
    "center_line": "street_markings",
    "painted_lines": "street_markings",
    "road_lines": "street_markings",
    "yellow_line": "street_markings",
    "white_line": "street_markings",
    # ── Road signs ───────────────────────────────────────────────────────────
    "road_signs_candidate": "road_signs",
    "street_sign": "street_sign",
    "street_signs": "street_sign",
    "signage": "road_signs",
    "signage_systems": "road_signs",
    "highway_markers": "road_signs",
    "road_sign": "street_sign",
    # ── Signposts ────────────────────────────────────────────────────────────
    "signposts_candidate": "signposts",
    "signpost": "signposts",
    "direction_sign": "signposts",
    "wayfinding_sign": "signposts",
    "distance_sign": "signposts",
    # ── Street names ─────────────────────────────────────────────────────────
    "street_names": "street_name",
    "street_name": "street_name",
    "street_name_sign": "street_name",
    "road_name": "street_name",
    "road_name_sign": "street_name",
    # ── Road surface ─────────────────────────────────────────────────────────
    "narrow_paved_rural_road": "road_surface",
    "unpaved_road": "road_surface",
    "dirt_road": "road_surface",
    "asphalt_road": "road_surface",
    "pavement": "road_surface",
    "road_quality": "road_surface",
    "asphalt": "road_surface",
    "concrete_road": "road_surface",
    "gravel_road": "road_surface",
    "paved_road": "road_surface",
    "road_texture": "road_surface",
    "sidewalk": "road_surface",
    # ── Terrain ──────────────────────────────────────────────────────────────
    "dry_dusty_ground": "terrain",
    "flat_farmland": "terrain",
    "mountainous_terrain": "terrain",
    "rolling_hills": "terrain",
    # ── Vegetation / Flora ───────────────────────────────────────────────────
    "sparse_scrubby_vegetation": "vegetation",
    "deciduous_trees_with_green_leaves": "vegetation",
    "tropical_vegetation": "vegetation",
    "dry_grassland": "vegetation",
    "flora": "flora",
    "vegetation": "flora",
    # ── Architecture ─────────────────────────────────────────────────────────
    "simple_single_story_rural_houses": "architecture",
    "reddish_brown_tiled_roofs_on_some_buildings": "architecture",
    "concrete_block_houses": "architecture",
    "modern_apartment_blocks": "architecture",
    "building_style": "architecture",
    "house_style": "architecture",
    "building_architecture": "architecture",
    "residential_architecture": "architecture",
    "commercial_architecture": "architecture",
    "building_design": "architecture",
    "facade": "architecture",
    "roof_style": "architecture",
    # ── Bollards ─────────────────────────────────────────────────────────────
    "bollard": "bollards",
    "traffic_bollard": "bollards",
    "road_bollard": "bollards",
    "parking_bollard": "bollards",
    "concrete_bollard": "bollards",
    "metal_bollard": "bollards",
    "yellow_bollard": "bollards",
    "white_bollard": "bollards",
    "plastic_bollard": "bollards",
    # ── Traffic lights ───────────────────────────────────────────────────────
    "traffic_light": "traffic_lights",
    "traffic_signal": "traffic_lights",
    "stoplight": "traffic_lights",
    "semaphore": "traffic_lights",
    "signal_light": "traffic_lights",
    # ── License plate ────────────────────────────────────────────────────────
    "license_plate": "license_plate",
    "licence_plate": "license_plate",
    "number_plate": "license_plate",
    "vehicle_plate": "license_plate",
    "car_plate": "license_plate",
    "registration_plate": "license_plate",
    "license_plates": "license_plate",
    # ── Language / Script ────────────────────────────────────────────────────
    "language_scripts": "language",
    "visible_languages": "language",
    "script": "language",
    "text_script": "language",
    "writing_system": "language",
    "alphabet": "language",
    "cyrillic": "language",
    "arabic_script": "language",
    "latin_script": "language",
    "chinese_characters": "language",
    "japanese_script": "language",
    "korean_script": "language",
    "devanagari": "language",
    "thai_script": "language",
    # ── Vehicles (kept for backward compat) ──────────────────────────────────
    "distinctive_vehicles": "vehicles",
    "tricycles": "vehicles",
    "tuk_tuks": "vehicles",
    "motorbikes": "vehicles",
    # ── Other legacy entries ──────────────────────────────────────────────────
    "road_edge_post": "road_edge_posts",
    "street_light": "street_lights",
}

EVIDENCE_TAG_RULES = {
    "rural": ("rural", "village", "farmland"),
    "dry_climate": ("dry", "dusty", "arid"),
    "tropical_dry": ("tropical", "scrubby", "dry_grassland"),
    "narrow_paved_road": ("narrow_paved",),
    "basic_infrastructure": ("basic_infrastructure", "simple_infrastructure"),
}

logger = logging.getLogger("meta_normalizer")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[MetaNormalizer] %(message)s"))
    logger.addHandler(handler)


def normalize_meta_name(value: str) -> str:
    """Normalize model text into a stable snake-case key."""
    normalized = unicodedata.normalize("NFKD", str(value))
    normalized = normalized.encode("ascii", "ignore").decode("ascii").lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return re.sub(r"_+", "_", normalized).strip("_")


def normalize_meta_for_retrieval(raw_meta: str) -> dict[str, str]:
    """Classify an observation as a canonical folder lookup or evidence tag."""
    raw = normalize_meta_name(raw_meta)
    canonical = EXACT_META_MAPPINGS.get(raw)

    if not canonical and raw in CANONICAL_RETRIEVAL_CATEGORIES:
        canonical = raw

    if not canonical:
        keyword_rules = [
            ("poles",          ("utility_pole", "wooden_pole", "concrete_pole")),
            ("traffic_lights", ("traffic_light", "traffic_signal", "stoplight", "semaphore", "signal_light")),
            ("bollards",       ("bollard",)),
            ("road_markings",  ("road_marking", "lane_marking", "road_paint")),
            ("street_markings",("street_marking", "painted_line", "road_line", "center_line")),
            ("street_sign",    ("street_sign", "road_sign")),
            ("road_signs",     ("highway_marker", "route_marker")),
            ("signposts",      ("signpost", "direction_sign", "wayfinding", "distance_sign")),
            ("street_name",    ("street_name", "road_name")),
            ("road_surface",   ("paved_road", "unpaved_road", "dirt_road", "asphalt_road", "pavement", "sidewalk")),
            ("terrain",        ("dusty_ground", "farmland", "mountain", "rolling_hill", "terrain", "landscape")),
            ("flora",          ("flora", "scrub", "deciduous_tree", "grassland")),
            ("vegetation",     ("vegetation",)),
            ("architecture",   ("house", "roof", "building", "apartment", "architecture", "facade")),
            ("license_plate",  ("license_plate", "licence_plate", "number_plate", "registration_plate")),
            ("license_plates", ("license_plates",)),
            ("language",       ("language", "script", "alphabet", "cyrillic", "devanagari",
                                "arabic_script", "thai_script", "writing_system")),
            ("vehicles",       ("vehicle", "tricycle", "tuk_tuk", "motorbike")),
            ("guardrails",     ("guardrail",)),
            ("road_edge_posts",("road_edge_post", "reflector_post")),
            ("street_lights",  ("street_light", "streetlamp")),
        ]
        for category, keywords in keyword_rules:
            if any(keyword in raw for keyword in keywords):
                canonical = category
                break

    if canonical:
        logger.info("%s -> %s", raw, canonical)
        return {"raw": raw, "canonical": canonical, "type": "retrieval_category"}

    return {"raw": raw, "canonical": raw, "type": "evidence_tag"}


def deduplicate(values: list[str]) -> list[str]:
    """Remove empty duplicate strings while preserving order."""
    seen = set()
    results = []

    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        results.append(normalized)

    return results


def as_list(value: Any) -> list[Any]:
    """Treat malformed model output as an empty list."""
    return value if isinstance(value, list) else []


def derive_evidence_tags(values: list[str]) -> list[str]:
    """Extract compact semantic tags that help reasoning but never folder lookup."""
    normalized_values = [normalize_meta_name(value) for value in values]
    tags = []

    for tag, keywords in EVIDENCE_TAG_RULES.items():
        if any(keyword in value for value in normalized_values for keyword in keywords):
            tags.append(tag)

    return tags


def build_meta_normalization(first_pass: dict[str, Any]) -> dict[str, list[str]]:
    """Build retrieval categories and diagnostics from first-pass output."""
    raw_metas = deduplicate(
        [
            *as_list(first_pass.get("retrieval_meta_categories")),
            *as_list(first_pass.get("detected_meta_categories")),
            *as_list(first_pass.get("visible_specific_metas")),
        ]
    )
    visual_observations = deduplicate(
        [
            *as_list(first_pass.get("visual_observations")),
            *as_list(first_pass.get("visible_specific_metas")),
        ]
    )
    existing_tags = [
        normalize_meta_name(tag) for tag in as_list(first_pass.get("evidence_tags"))
    ]
    categories = []
    skipped = []

    for raw_meta in raw_metas:
        normalized = normalize_meta_for_retrieval(raw_meta)
        if normalized["type"] == "retrieval_category":
            categories.append(normalized["canonical"])
        else:
            skipped.append(normalized["raw"])

    evidence_tags = deduplicate(
        [
            *existing_tags,
            *derive_evidence_tags([*raw_metas, *visual_observations]),
            *skipped,
        ]
    )
    return {
        "raw_metas": raw_metas,
        "canonical_retrieval_categories": deduplicate(categories),
        "visual_observations": visual_observations,
        "evidence_tags": evidence_tags,
        "skipped_non_retrieval_metas": deduplicate(skipped),
    }


def apply_meta_normalization(first_pass: dict[str, Any]) -> dict[str, Any]:
    """Apply the normalized schema to a first-pass reasoning result."""
    normalization = build_meta_normalization(first_pass)
    first_pass["visual_observations"] = normalization["visual_observations"]
    first_pass["retrieval_meta_categories"] = normalization[
        "canonical_retrieval_categories"
    ]
    first_pass["evidence_tags"] = normalization["evidence_tags"]
    first_pass["detected_meta_categories"] = normalization[
        "canonical_retrieval_categories"
    ]
    return first_pass
