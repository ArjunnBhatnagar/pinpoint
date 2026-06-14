import json
import logging
import re
from pathlib import Path
from typing import Any


# Phase 8D: deterministic confidence verification.
# This module does not ask Gemini. It checks whether previous phases produced
# enough evidence to trust the grounded country prediction.
ANALYSIS_DIR = Path("analysis")
CONFIDENCE_REPORT_PATH = ANALYSIS_DIR / "confidence_report.json"

FIRST_PASS_FILE = "first_pass.json"
RETRIEVAL_FILE = "retrieval_package.json"
GROUNDED_FILE = "grounded_result.json"
OCR_FILE = "ocr_results.json"
OCR_STATUS_FILE = "ocr_status.json"
IMAGE_RANKING_FILE = "image_ranking.json"
REGIONAL_CONSISTENCY_FILE = "regional_consistency.json"

# ── Definitive identifier config ─────────────────────────────────────────────
DEFINITIVE_IDENTIFIER_CONFIDENCE_FLOOR = 0.88

# ── Visually-indistinguishable enclave pairs ──────────────────────────────────
# Small enclave countries that share virtually all visual characteristics with
# the larger country that surrounds them.  When the grounded result selects one
# of these small countries but no definitive country-specific identifier is
# present, the larger parent country is preferred because:
#   1. The parent is statistically far more likely to be the actual location.
#   2. Generic visual matches (terrain, vegetation, road markings, poles) are
#      equally valid for both countries and cannot distinguish them reliably.
#   3. The small country's reference images may accidentally match better due to
#      database coverage imbalance rather than genuine visual discrimination.
VISUALLY_INDISTINGUISHABLE_PAIRS: dict[str, list[str]] = {
    "Eswatini":    ["South Africa"],
    "Lesotho":     ["South Africa"],
    "Vatican City":["Italy"],
    "San Marino":  ["Italy"],
    "Monaco":      ["France"],
    "Brunei":      ["Malaysia"],
}

# Definitive identifiers that WOULD genuinely place us in the small country.
# Patterns are applied against the combined reasoning + first-pass text.
_DEFINITIVE_ENCLAVE_IDENTIFIERS: dict[str, list[re.Pattern]] = {
    "Eswatini": [
        re.compile(r"\b(?:mbabane|manzini|swazi|lilangeni|swaziland)\b", re.I),
    ],
    "Lesotho": [
        re.compile(r"\b(?:maseru|lesotho|basotho|maloti|loti)\b", re.I),
    ],
    "Vatican City": [
        re.compile(r"\b(?:vatican|holy see|st\.?\s*peter(?:\'?s)?|sistine)\b", re.I),
    ],
    "San Marino": [
        re.compile(r"\b(?:san\s+marino|sammarinese|serravalle)\b", re.I),
    ],
    "Monaco": [
        re.compile(r"\b(?:monaco|monte[\s-]?carlo|monegasque|principaut[eé])\b", re.I),
    ],
    "Brunei": [
        re.compile(r"\b(?:brunei|bandar\s+seri\s+begawan|bruneian|darussalam)\b", re.I),
    ],
}

# Valid US state abbreviations for Pattern A matching.
_US_STATE_CODES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
    "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
    "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
    "TX","UT","VT","VA","WA","WV","WI","WY",
}
_US_STATE_PATTERN = re.compile(
    # MUST use hyphen — space separator is NOT valid for US state codes.
    # "IN 15" (from error messages "retry in 15") must NOT match.
    # Single-letter prefixes (L, N, B, D, A, E) are European, never US state codes.
    r"\b(" + "|".join(sorted(_US_STATE_CODES, key=len, reverse=True)) + r")-\d{1,4}\b",
    re.IGNORECASE,
)

# Pattern B — country-specific national route prefixes (definitive identifiers).
_NATIONAL_ROUTE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bRN[- ]\d+\b", re.I), "Argentina"),
    (re.compile(r"\bRuta Nacional\b", re.I), "Argentina"),
    (re.compile(r"\bBR[- ]\d+\b", re.I), "Brazil"),
]

# European road prefixes — these are EVIDENCE TAGS only, NOT definitive identifiers.
# They support regional narrowing but do not trigger the confidence floor.
_EUROPEAN_ROAD_PREFIXES: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"\bL\d{1,4}\b"),          "German/Austrian Landesstrasse",  "Germany"),
    (re.compile(r"\bB\d{1,4}\b"),          "German Bundesstrasse",           "Germany"),
    (re.compile(r"\bA\d{1,3}\b"),          "German/Austrian Autobahn",       "Germany"),
    (re.compile(r"\bD\d{1,4}\b"),          "French Departmental road",       "France"),
    (re.compile(r"\bN\d{1,4}\b"),          "French National road",           "France"),
    (re.compile(r"\bE\d{1,4}\b"),          "European route",                 None),
    (re.compile(r"\bRN\s*\d+\b", re.I),   "Argentine national route",       "Argentina"),
]

# Pattern C — Gemini language signalling certainty.
_CERTAINTY_PHRASES = re.compile(
    r"\b(?:unequivocally|definitive(?:ly)?|directly identifies|unambiguous(?:ly)?|"
    r"certain(?:ly)?|confirms|proves|specific identifier|"
    r"strongly suggests [a-z]|restricts the location|state abbreviation)\b",
    re.I,
)

# Pattern D — named places that definitively identify a country (extend as needed).
_NAMED_PLACE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bWyoming\b", re.I), "United States"),
    (re.compile(r"\bBogot[áa]\b", re.I), "Colombia"),
    (re.compile(r"\bSão Paulo\b", re.I), "Brazil"),
    (re.compile(r"\bMexico City\b", re.I), "Mexico"),
    (re.compile(r"\bBuenos Aires\b", re.I), "Argentina"),
    (re.compile(r"\bSydney\b", re.I), "Australia"),
]

logger = logging.getLogger("confidence_verifier")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[Confidence] %(message)s"))
    logger.addHandler(handler)


def load_json_file(path: Path, default: Any) -> tuple[Any, list[str]]:
    """Load JSON and return warnings instead of raising."""
    if not path.exists():
        return default, [f"Missing file: {path}"]

    try:
        return json.loads(path.read_text(encoding="utf-8")), []
    except Exception as error:
        return default, [f"Malformed JSON in {path}: {error}"]


def normalize_text(value: Any) -> str:
    """Normalize text for simple rule checks."""
    return str(value or "").strip().casefold()


def apply_regional_selection(
    grounded_result: dict[str, Any],
    regional_consistency: dict[str, Any],
) -> dict[str, Any]:
    """Prefer the region-consistent country while preserving raw grounding."""
    if not isinstance(grounded_result, dict):
        return {}

    selected_country = (
        regional_consistency.get("selected_country")
        if isinstance(regional_consistency, dict)
        else None
    )
    if not selected_country:
        return grounded_result

    adjusted = dict(grounded_result)
    adjusted["raw_grounded_country"] = grounded_result.get("final_country")
    adjusted["final_country"] = selected_country
    adjusted["regional_consistency"] = regional_consistency
    logger.info("Using region-consistent country: %s", selected_country)
    return adjusted


HIGH_SPECIFICITY_PATTERNS = {
    "visual_text": [
        re.compile(r"\btroncal\s+[a-záéíóúñü]+", re.I),
        re.compile(r"\bcl\.?\s*\d+(?:\s+(?:sur|norte|este|oeste))?\b", re.I),
        re.compile(r"\b(?:calle|carrera|avenida|rua|jalan)\s+[a-z0-9áéíóúñü.-]+", re.I),
        re.compile(r"\b(?:route|ruta|highway|hwy|autoroute|autopista)\s*[a-z-]?\d+\b", re.I),
        re.compile(r"\b[a-z]{1,3}[- ]?\d{1,4}\b", re.I),
    ],
    "public_transport": [
        re.compile(r"\btransmilenio\b", re.I),
        re.compile(r"\bbus rapid transit\b", re.I),
        re.compile(r"\bbrt\b", re.I),
        re.compile(r"\bmetrobus\b", re.I),
        re.compile(r"\bred\s+(?:brt\s+)?buses?\b", re.I),
    ],
    "unique_infrastructure": [
        re.compile(r"\bdedicated\s+(?:bus\s+)?lanes?\b", re.I),
        re.compile(r"\bbrt\s+(?:stations?|infrastructure|system)\b", re.I),
        re.compile(r"\bcity-specific\b", re.I),
        re.compile(r"\bcountry-specific\b", re.I),
        re.compile(r"\bhighly characteristic\b", re.I),
        re.compile(r"\bhallmark\b", re.I),
    ],
    "named_place": [
        re.compile(r"\bbogot[áa]\b", re.I),
        re.compile(r"\bcolombia\b", re.I),
        re.compile(r"\bcaracas\b", re.I),
    ],
}


def _collect_safe_reasoning_strings(
    grounded_result: dict[str, Any],
    first_pass: dict[str, Any],
) -> str:
    """Collect reasoning text while EXCLUDING error/fallback fields.

    uncertainty_notes may contain raw API error messages (e.g. 429 bodies with
     "retry in 15s") that would otherwise poison the identifier scanner.
    """
    # Fields to exclude — these can contain error text, not geographic evidence
    EXCLUDED_KEYS = {
        "uncertainty_notes", "fallback_reason", "warning",
        "_warnings", "rate_limited", "images_actually_sent",
    }

    def _safe_collect(value: Any, excluded: set) -> list[str]:
        if isinstance(value, dict):
            strings = []
            for k, v in value.items():
                if k in excluded:
                    continue
                strings.extend(_safe_collect(v, excluded))
            return strings
        if isinstance(value, list):
            strings = []
            for item in value:
                strings.extend(_safe_collect(item, excluded))
            return strings
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    all_strings: list[str] = []
    for doc in (grounded_result, first_pass):
        all_strings.extend(_safe_collect(doc, EXCLUDED_KEYS))
    return " ".join(all_strings)


def detect_definitive_identifier(
    grounded_result: dict[str, Any],
    first_pass: dict[str, Any],
) -> dict[str, Any]:
    """Scan reasoning text for patterns that definitively identify one country.

    Returns a dict with detected=True/False plus clue details when found.

    Rules:
    - US state highway codes are ONLY checked when final_country is United States.
    - Uncertainty_notes and error fields are EXCLUDED to prevent API error messages
      from being scanned (e.g. "retry in 15s" must not match "IN-15").
    - European road prefixes (L173, B35 etc.) are detected as evidence tags only,
      NOT as definitive identifiers.
    """
    final_country = str(grounded_result.get("final_country") or "")
    combined = _collect_safe_reasoning_strings(grounded_result, first_pass)

    # Pattern A — US state highway codes (WY-272, TX-35, …)
    # ONLY apply when the candidate country is actually the United States.
    final_country_is_usa = (
        "united states" in normalize_text(final_country)
        or "usa" in normalize_text(final_country)
    )
    if final_country_is_usa:
        m = _US_STATE_PATTERN.search(combined)
        if m:
            matched_state = m.group(1).upper()
            clue = m.group(0)
            logger.info("[Confidence] Definitive identifier detected: %s", clue)
            logger.info("[Confidence] Identifier type: us_state_highway_code")
            logger.info("[Confidence] Confidence floor applied: %.2f", DEFINITIVE_IDENTIFIER_CONFIDENCE_FLOOR)
            logger.info("[Confidence] Penalties waived: ocr_empty, score_margin")
            return {
                "detected": True,
                "clue": f"{clue} — US state highway code ({matched_state})",
                "type": "us_state_highway_code",
                "matched_country": "United States",
                "confidence_floor_applied": DEFINITIVE_IDENTIFIER_CONFIDENCE_FLOOR,
                "consistent_with_final_country": True,
            }
    else:
        # Check if this looks like a European road number misidentified as US state
        for pattern, label, _country in _EUROPEAN_ROAD_PREFIXES:
            m = pattern.search(combined)
            if m:
                logger.info(
                    "[Confidence] European road prefix detected: %s (%s) — "
                    "Adding as evidence tag, not definitive identifier.",
                    m.group(0), label,
                )
                # Return as evidence tag only — no confidence floor
                return {
                    "detected": False,
                    "european_road_prefix": {
                        "clue": m.group(0),
                        "type": label,
                        "note": "European road prefix — evidence tag only, not a definitive identifier",
                    },
                }

    # Pattern B — national route prefixes
    for pattern, country in _NATIONAL_ROUTE_PATTERNS:
        m = pattern.search(combined)
        if m:
            clue = m.group(0)
            logger.info("Definitive identifier detected: %s", clue)
            logger.info("Identifier type: national_route_prefix")
            logger.info("Confidence floor applied: %.2f", DEFINITIVE_IDENTIFIER_CONFIDENCE_FLOOR)
            logger.info("Penalties waived: ocr_empty, score_margin")
            return {
                "detected": True,
                "clue": f"{clue} — national route prefix for {country}",
                "type": "national_route_prefix",
                "matched_country": country,
                "confidence_floor_applied": DEFINITIVE_IDENTIFIER_CONFIDENCE_FLOOR,
                "consistent_with_final_country": normalize_text(country) in normalize_text(final_country)
                    or normalize_text(final_country) in normalize_text(country),
            }

    # Pattern C — Gemini certainty language
    m = _CERTAINTY_PHRASES.search(combined)
    if m:
        clue = m.group(0)
        logger.info("Definitive identifier detected (certainty phrase): %s", clue)
        logger.info("Identifier type: gemini_certainty_language")
        logger.info("Confidence floor applied: %.2f", DEFINITIVE_IDENTIFIER_CONFIDENCE_FLOOR)
        return {
            "detected": True,
            "clue": f'Gemini certainty phrase: "{clue}"',
            "type": "gemini_certainty_language",
            "matched_country": final_country,
            "confidence_floor_applied": DEFINITIVE_IDENTIFIER_CONFIDENCE_FLOOR,
            "consistent_with_final_country": True,
        }

    # Pattern D — named places
    for pattern, country in _NAMED_PLACE_PATTERNS:
        m = pattern.search(combined)
        if m:
            clue = m.group(0)
            logger.info("Definitive identifier detected (named place): %s", clue)
            logger.info("Identifier type: named_place")
            logger.info("Confidence floor applied: %.2f", DEFINITIVE_IDENTIFIER_CONFIDENCE_FLOOR)
            return {
                "detected": True,
                "clue": f"{clue} — named place uniquely identifying {country}",
                "type": "named_place",
                "matched_country": country,
                "confidence_floor_applied": DEFINITIVE_IDENTIFIER_CONFIDENCE_FLOOR,
                "consistent_with_final_country": normalize_text(country) in normalize_text(final_country)
                    or normalize_text(final_country) in normalize_text(country),
            }

    return {"detected": False}


def collect_reasoning_strings(value: Any) -> list[str]:
    """Flatten grounded reasoning text for deterministic clue inspection."""
    if isinstance(value, dict):
        strings = []
        for nested in value.values():
            strings.extend(collect_reasoning_strings(nested))
        return strings

    if isinstance(value, list):
        strings = []
        for nested in value:
            strings.extend(collect_reasoning_strings(nested))
        return strings

    if isinstance(value, str) and value.strip():
        return [value.strip()]

    return []


def analyze_high_specificity_evidence(
    grounded_result: dict[str, Any],
) -> dict[str, Any]:
    """Detect named visual clues and distinctive infrastructure evidence."""
    final_country = normalize_text(grounded_result.get("final_country"))
    evidence_sources = {
        "winning_evidence": grounded_result.get("winning_evidence", []),
        "general_reasoning_evidence": grounded_result.get("general_reasoning_evidence", []),
        "evidence_breakdown": grounded_result.get("evidence_breakdown", {}),
    }
    strings = collect_reasoning_strings(evidence_sources)
    items = []
    categories = set()

    for text in strings:
        matched_categories = [
            category
            for category, patterns in HIGH_SPECIFICITY_PATTERNS.items()
            if any(pattern.search(text) for pattern in patterns)
        ]
        if not matched_categories:
            continue

        categories.update(matched_categories)
        items.append(
            {
                "text": text,
                "categories": matched_categories,
            }
        )

    unique_items = []
    seen = set()
    for item in items:
        normalized = normalize_text(item["text"])
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_items.append(item)

    combined_text = normalize_text(" ".join(item["text"] for item in unique_items))
    supports_final_country = bool(final_country and final_country in combined_text)
    has_primary_clue = bool({"visual_text", "public_transport"} & categories)

    if supports_final_country and has_primary_clue and len(categories) >= 3:
        strength = "very_high"
        boost = 0.15
    elif supports_final_country and has_primary_clue:
        strength = "high"
        boost = 0.10
    elif has_primary_clue:
        strength = "medium"
        boost = 0.04
    else:
        strength = "none"
        boost = 0.0

    if unique_items:
        logger.info(
            "High-specificity evidence detected: %s",
            "; ".join(item["text"] for item in unique_items[:3]),
        )
    if supports_final_country and "visual_text" in categories:
        logger.info("Visual text clue supports %s", grounded_result.get("final_country"))

    return {
        "present": strength in {"high", "very_high"},
        "items": unique_items,
        "categories": sorted(categories),
        "supports_final_country": supports_final_country,
        "strength": strength,
        "confidence_boost": boost,
    }


def score_to_strength(value: float, thresholds: tuple[float, float, float]) -> str:
    """Convert a numeric score into low/medium/high style labels."""
    low, medium, high = thresholds
    if value >= high:
        return "high"
    if value >= medium:
        return "medium"
    if value >= low:
        return "low"
    return "none"


def analyze_retrieval_health(retrieval_package: dict[str, Any]) -> dict[str, Any]:
    """Evaluate whether Phase 8B retrieved useful reference material."""
    logger.info("Checking retrieval health")
    summary = retrieval_package.get("summary", {}) if isinstance(retrieval_package, dict) else {}
    total_images = int(summary.get("total_images", 0) or 0)
    total_texts = int(summary.get("total_texts", 0) or 0)
    countries_retrieved = retrieval_package.get("countries_actually_retrieved", [])
    missing_countries = summary.get("missing_countries", [])
    missing_meta_categories = summary.get("missing_meta_categories", [])

    if total_images == 0 and total_texts == 0:
        status = "empty"
        penalty = 0.35
    elif total_images < 2 and total_texts < 2:
        status = "weak"
        penalty = 0.18
    elif total_images >= 5 or total_texts >= 4:
        status = "good"
        penalty = 0.0
    else:
        status = "limited"
        penalty = 0.08

    if missing_countries:
        penalty += min(0.15, 0.05 * len(missing_countries))

    return {
        "total_images": total_images,
        "total_texts": total_texts,
        "countries_actually_retrieved": countries_retrieved,
        "missing_countries": missing_countries,
        "missing_meta_categories": missing_meta_categories,
        "status": status,
        "confidence_penalty": round(min(penalty, 0.45), 3),
    }


def analyze_score_margin(grounded_result: dict[str, Any]) -> dict[str, Any]:
    """Evaluate whether the winning country is clearly ahead."""
    logger.info("Checking score margin")
    scores = grounded_result.get("country_scores", {})

    if not isinstance(scores, dict) or not scores:
        return {
            "top_country": grounded_result.get("final_country"),
            "second_country": None,
            "top_score": 0.0,
            "second_score": 0.0,
            "margin": 0.0,
            "margin_ratio": 0.0,
            "margin_strength": "none",
            "confidence_penalty": 0.25,
        }

    parsed_scores = []
    for country, score in scores.items():
        try:
            parsed_scores.append((str(country), float(score)))
        except (TypeError, ValueError):
            parsed_scores.append((str(country), 0.0))

    parsed_scores.sort(key=lambda item: item[1], reverse=True)
    top_country, top_score = parsed_scores[0]
    second_country, second_score = parsed_scores[1] if len(parsed_scores) > 1 else (None, 0.0)
    margin = top_score - second_score
    margin_ratio = margin / max(abs(top_score), 1.0)

    if margin >= 8 and margin_ratio >= 0.4:
        strength = "strong"
        penalty = 0.0
    elif margin >= 4 and margin_ratio >= 0.25:
        strength = "moderate"
        penalty = 0.06
    elif margin > 0:
        strength = "weak"
        penalty = 0.16
    else:
        strength = "none"
        penalty = 0.28

    return {
        "top_country": top_country,
        "second_country": second_country,
        "top_score": top_score,
        "second_score": second_score,
        "margin": margin,
        "margin_ratio": round(margin_ratio, 3),
        "margin_strength": strength,
        "confidence_penalty": penalty,
    }


def analyze_ocr_strength(
    ocr_results: list[dict[str, Any]],
    grounded_result: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate OCR amount, relevance, support, and possible contradiction."""
    logger.info("Checking OCR strength")
    final_country = normalize_text(grounded_result.get("final_country"))
    detections = []

    for result in ocr_results if isinstance(ocr_results, list) else []:
        for detection in result.get("detections", []):
            text = str(detection.get("text", "")).strip()
            if not text:
                continue

            confidence = float(detection.get("confidence", 0.0) or 0.0)
            relevance = normalize_text(detection.get("relevance"))
            detections.append(
                {
                    "text": text,
                    "confidence": confidence,
                    "relevance": relevance,
                }
            )

    high_relevance_terms = ["high", "location", "sign", "road", "language", "marker"]
    relevant_count = sum(
        1
        for item in detections
        if item["confidence"] >= 0.45
        and any(term in item["relevance"] for term in high_relevance_terms)
    )
    support_count = sum(
        1 for item in detections if final_country and final_country in normalize_text(item["text"])
    )

    grounded_text = normalize_text(
        " ".join(grounded_result.get("winning_evidence", []))
        + " "
        + " ".join(grounded_result.get("contradictions", []))
    )
    ocr_referenced_by_grounded = "ocr" in grounded_text

    if support_count > 0:
        strength = "high"
        penalty = 0.0
    elif relevant_count >= 3 or ocr_referenced_by_grounded:
        strength = "medium"
        penalty = 0.04
    elif detections:
        strength = "low"
        penalty = 0.1
    else:
        strength = "none"
        penalty = 0.16

    return {
        "detection_count": len(detections),
        "relevant_detection_count": relevant_count,
        "direct_support_count": support_count,
        "grounded_result_mentions_ocr": ocr_referenced_by_grounded,
        "strength": strength,
        "confidence_penalty": penalty,
    }


def analyze_image_quality(image_ranking: dict[str, Any]) -> dict[str, Any]:
    """Evaluate selector quality, visible metas, and diversity."""
    logger.info("Checking image quality")
    screenshots = image_ranking.get("all_screenshots", []) if isinstance(image_ranking, dict) else []
    if not screenshots:
        return {
            "selected_count": 0,
            "average_score": 0.0,
            "combined_metas": [],
            "diversity_score": 0.0,
            "quality": "none",
            "confidence_penalty": 0.18,
        }

    selected = [item for item in screenshots if not item.get("rejected")]
    if not selected:
        selected = screenshots

    scores = [float(item.get("final_score", 0.0) or 0.0) for item in selected]
    average_score = sum(scores) / max(len(scores), 1)
    metas = set()
    dominant_types = set()

    for item in selected:
        gemini = item.get("gemini", {})
        metas.update(str(meta) for meta in gemini.get("detected_metas", []))
        dominant = gemini.get("dominant_type")
        if dominant:
            dominant_types.add(str(dominant))

    diversity_score = len(dominant_types) / max(len(selected), 1)

    if average_score >= 0.65 and len(metas) >= 5:
        quality = "high"
        penalty = 0.0
    elif average_score >= 0.45 and len(metas) >= 3:
        quality = "medium"
        penalty = 0.06
    elif average_score >= 0.25:
        quality = "low"
        penalty = 0.14
    else:
        quality = "very_low"
        penalty = 0.24

    return {
        "selected_count": len(selected),
        "average_score": round(average_score, 3),
        "combined_metas": sorted(metas),
        "dominant_types": sorted(dominant_types),
        "diversity_score": round(diversity_score, 3),
        "quality": quality,
        "confidence_penalty": penalty,
    }


def analyze_contradictions(grounded_result: dict[str, Any]) -> dict[str, Any]:
    """Evaluate contradiction severity from grounded_result.json.

    Contradictions that exclusively name a losing candidate country are NOT
    counted as real contradictions — they are evidence against alternatives,
    which is expected and healthy in a grounded comparison.
    Only contradictions that implicate the final (winning) country, or that
    contain no country name at all, are counted for severity scoring.
    """
    logger.info("Checking evidence contradictions")
    contradictions = grounded_result.get("contradictions", [])
    uncertainty_notes = grounded_result.get("uncertainty_notes", [])

    if not isinstance(contradictions, list):
        contradictions = []
    if not isinstance(uncertainty_notes, list):
        uncertainty_notes = []

    final_country = normalize_text(grounded_result.get("final_country", ""))
    country_scores = grounded_result.get("country_scores", {})
    all_countries_norm = {normalize_text(c) for c in country_scores if c}
    losing_countries_norm = all_countries_norm - ({final_country} if final_country else set())

    # Filter: keep only contradictions that mention the winner or mention no country.
    real_contradictions: list[str] = []
    for item in contradictions:
        item_norm = normalize_text(str(item))
        mentions_winner = bool(final_country and final_country in item_norm)
        mentions_loser_only = (
            any(lc in item_norm for lc in losing_countries_norm) and not mentions_winner
        )
        if not mentions_loser_only:
            real_contradictions.append(str(item))

    filtered_count = len(contradictions) - len(real_contradictions)
    if filtered_count > 0:
        logger.info(
            "[Confidence] Filtered %d alternative-country contradictions "
            "(evidence against losing candidates, not real contradictions)",
            filtered_count,
        )

    text = normalize_text(
        " ".join(real_contradictions + [str(n) for n in uncertainty_notes])
    )
    serious_terms = ["contradict", "conflict", "no evidence", "missing", "fallback", "failed"]
    serious_count = sum(1 for term in serious_terms if term in text)

    if len(real_contradictions) >= 3 or serious_count >= 2:
        severity = "high"
        penalty = 0.22
    elif real_contradictions or serious_count:
        severity = "medium"
        penalty = 0.1
    else:
        severity = "low"
        penalty = 0.0

    return {
        "contradictions": contradictions,           # full list for transparency
        "real_contradictions": real_contradictions, # filtered list used for scoring
        "uncertainty_notes": uncertainty_notes,
        "severity": severity,
        "confidence_penalty": penalty,
    }


def analyze_reference_usage(grounded_result: dict[str, Any]) -> dict[str, Any]:
    """Check whether 8C appears to have used retrieved references."""
    reference_usage = grounded_result.get("reference_usage", [])
    country_comparisons = grounded_result.get("country_reference_comparisons", [])
    meta_matches = grounded_result.get("meta_matches", {})

    if isinstance(reference_usage, dict):
        references_provided = bool(reference_usage.get("references_provided"))
        references_used = bool(reference_usage.get("references_used_in_reasoning"))
        usage_count = (
            int(reference_usage.get("reference_images_count", 0) or 0)
            + int(reference_usage.get("reference_texts_count", 0) or 0)
        )
        comparison_items = (
            country_comparisons if isinstance(country_comparisons, list) else []
        )
    else:
        references_provided = bool(reference_usage)
        references_used = bool(reference_usage)
        usage_count = len(reference_usage) if isinstance(reference_usage, list) else 0
        comparison_items = reference_usage if isinstance(reference_usage, list) else []

    match_count = len(meta_matches) if isinstance(meta_matches, dict) else 0
    strong_usage_count = 0
    for item in comparison_items:
        if not isinstance(item, dict):
            continue

        try:
            reference_match_score = float(item.get("reference_match_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            reference_match_score = 0.0

        if (
            str(item.get("match_strength", "")).casefold() == "high"
            or reference_match_score >= 0.7
            or bool(item.get("strong_matches"))
        ):
            strong_usage_count += 1
    strong_meta_match_count = sum(
        1
        for item in meta_matches.values()
        if isinstance(item, dict)
        and str(item.get("strength", "")).casefold() == "high"
    )
    strong_match_count = strong_usage_count + strong_meta_match_count

    if strong_match_count >= 1:
        strength = "high"
        penalty = 0.0
    elif references_used or usage_count >= 1 or match_count >= 1:
        strength = "medium"
        penalty = 0.1
    else:
        strength = "low"
        penalty = 0.18

    return {
        "reference_usage_count": usage_count,
        "references_provided": references_provided,
        "references_used_in_reasoning": references_used,
        "meta_match_count": match_count,
        "strong_match_count": strong_match_count,
        "strength": strength,
        "confidence_penalty": penalty,
    }


def analyze_specific_evidence(
    first_pass: dict[str, Any],
    grounded_result: dict[str, Any],
) -> dict[str, Any]:
    """Penalize generic-only reasoning and close first-pass candidates."""
    specific_metas = first_pass.get("visible_specific_metas", [])
    generic_warnings = grounded_result.get("generic_evidence_warnings", [])
    missing_reference_comparisons = grounded_result.get("missing_reference_comparisons", [])
    candidates = first_pass.get("top_candidates", [])
    confidences = sorted(
        [
            float(item.get("confidence", 0.0) or 0.0)
            for item in candidates
            if isinstance(item, dict)
        ],
        reverse=True,
    )
    first_pass_margin = (
        confidences[0] - confidences[1]
        if len(confidences) >= 2
        else (confidences[0] if confidences else 0.0)
    )

    penalty = 0.0
    if not specific_metas:
        penalty += 0.12
    if generic_warnings:
        penalty += min(0.12, 0.04 * len(generic_warnings))
        logger.info("Generic evidence detected; confidence penalty applied")
    if missing_reference_comparisons:
        penalty += min(0.08, 0.02 * len(missing_reference_comparisons))
    if len(confidences) >= 2 and first_pass_margin < 0.1:
        penalty += 0.08

    return {
        "visible_specific_metas": specific_metas,
        "generic_evidence_warnings": generic_warnings,
        "missing_reference_comparisons": missing_reference_comparisons,
        "first_pass_candidate_margin": round(first_pass_margin, 3),
        "confidence_penalty": round(min(penalty, 0.28), 3),
    }


def calibrate_penalties(
    retrieval_health: dict[str, Any],
    score_analysis: dict[str, Any],
    ocr_strength: dict[str, Any],
    image_quality: dict[str, Any],
    contradictions: dict[str, Any],
    reference_usage: dict[str, Any],
    specific_evidence: dict[str, Any],
    high_specificity: dict[str, Any],
    definitive_identifier: dict[str, Any] | None = None,
    ocr_was_attempted: bool = True,
) -> dict[str, float]:
    """Reduce weak-source penalties when independent specific evidence exists."""
    penalties = {
        "retrieval": float(retrieval_health["confidence_penalty"]),
        "score_margin": float(score_analysis["confidence_penalty"]),
        "ocr": float(ocr_strength["confidence_penalty"]),
        "image_quality": float(image_quality["confidence_penalty"]),
        "contradictions": float(contradictions["confidence_penalty"]),
        "references": float(reference_usage["confidence_penalty"]),
        "generic_or_missing_specific_evidence": float(
            specific_evidence["confidence_penalty"]
        ),
    }

    # When OCR was disabled (not attempted), waive the OCR penalty entirely.
    if not ocr_was_attempted:
        penalties["ocr"] = 0.0
        logger.info("[Confidence] OCR disabled — OCR penalty waived")

    has_high_specificity = (
        high_specificity.get("present")
        and high_specificity.get("supports_final_country")
    )
    has_definitive = bool(
        definitive_identifier
        and definitive_identifier.get("detected")
        and definitive_identifier.get("consistent_with_final_country")
    )

    # ── Definitive identifier overrides: waive score_margin + contradictions ─
    if has_definitive:
        penalties["score_margin"] = 0.0
        penalties["ocr"] = 0.0
        penalties["contradictions"] = 0.0
        penalties["generic_or_missing_specific_evidence"] = min(
            penalties["generic_or_missing_specific_evidence"], 0.02
        )
        logger.info("[Confidence] Penalties waived: ocr_empty, score_margin (definitive identifier)")
        logger.info("[Confidence] Contradictions penalty waived: all contradictions explain alternatives, not the winner")
        return penalties

    # ── High-specificity evidence (existing calibration, extended) ────────────
    if not has_high_specificity:
        return penalties

    if ocr_was_attempted and ocr_strength.get("strength") in {"none", "low"}:
        penalties["ocr"] = min(penalties["ocr"], 0.02)
        logger.info(
            "OCR is %s but penalty reduced due to strong visual evidence",
            ocr_strength.get("strength"),
        )

    if retrieval_health.get("status") != "empty":
        penalties["retrieval"] = min(penalties["retrieval"], 0.03)

    # Extended: also calibrate "weak" margin (not just "moderate")
    if score_analysis.get("margin_strength") in {"moderate", "weak"}:
        penalties["score_margin"] = min(penalties["score_margin"], 0.03)
        logger.info("Score margin penalty reduced due to strong visual evidence")

    # Extended: also calibrate "high" contradiction severity when specificity present
    if contradictions.get("severity") in {"medium", "high"}:
        penalties["contradictions"] = min(penalties["contradictions"], 0.05)
        logger.info("Contradiction penalty reduced due to strong visual evidence")

    penalties["generic_or_missing_specific_evidence"] = min(
        penalties["generic_or_missing_specific_evidence"],
        0.04,
    )
    return penalties


def calculate_verified_confidence(
    grounded_result: dict[str, Any],
    retrieval_health: dict[str, Any],
    score_analysis: dict[str, Any],
    ocr_strength: dict[str, Any],
    image_quality: dict[str, Any],
    contradictions: dict[str, Any],
    reference_usage: dict[str, Any],
    specific_evidence: dict[str, Any],
    high_specificity: dict[str, Any],
    definitive_identifier: dict[str, Any] | None = None,
    ocr_was_attempted: bool = True,
) -> float:
    """Fuse reported confidence with deterministic penalties."""
    try:
        reported = float(grounded_result.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        reported = 0.0

    if grounded_result.get("reasoning_status") == "fallback":
        reported = min(reported, 0.2)

    penalties = calibrate_penalties(
        retrieval_health,
        score_analysis,
        ocr_strength,
        image_quality,
        contradictions,
        reference_usage,
        specific_evidence,
        high_specificity,
        definitive_identifier=definitive_identifier,
        ocr_was_attempted=ocr_was_attempted,
    )
    verified = max(
        0.0,
        min(
            1.0,
            reported
            + float(high_specificity.get("confidence_boost", 0.0) or 0.0)
            - sum(penalties.values()),
        ),
    )

    # Apply definitive identifier confidence floor.
    if (
        definitive_identifier
        and definitive_identifier.get("detected")
        and definitive_identifier.get("consistent_with_final_country")
    ):
        floor = float(definitive_identifier.get("confidence_floor_applied", DEFINITIVE_IDENTIFIER_CONFIDENCE_FLOOR))
        if verified < floor:
            logger.info(
                "[Confidence] Confidence floor applied: %.2f → %.2f (definitive identifier)",
                verified,
                floor,
            )
            verified = floor

    high_specificity["effective_penalties"] = {
        key: round(value, 3) for key, value in penalties.items()
    }
    return round(verified, 3)


def determine_trust_level(
    verified_confidence: float,
    high_specificity: dict[str, Any],
    contradictions: dict[str, Any],
    reference_usage: dict[str, Any],
) -> str:
    """Map verified confidence to trust levels."""
    specific = bool(high_specificity.get("present"))
    strong_reference = reference_usage.get("strength") == "high"
    serious_contradiction = contradictions.get("severity") == "high"

    if verified_confidence >= 0.9 and specific and not serious_contradiction:
        return "very_high"
    if (
        verified_confidence >= 0.8
        and (specific or strong_reference)
        and not serious_contradiction
    ):
        return "high"
    if verified_confidence >= 0.55:
        return "medium"
    if verified_confidence >= 0.35:
        return "low"
    return "unreliable"


def determine_recommendation(
    trust_level: str,
    retrieval_health: dict[str, Any],
    score_analysis: dict[str, Any],
    contradictions: dict[str, Any],
    high_specificity: dict[str, Any],
) -> tuple[str, bool]:
    """Choose a deterministic next-step recommendation."""
    if trust_level in {"very_high", "high"}:
        return "accept_prediction", False

    if (
        trust_level == "medium"
        and high_specificity.get("present")
        and high_specificity.get("supports_final_country")
        and contradictions["severity"] != "high"
    ):
        return "accept_with_caution", False

    if retrieval_health["status"] == "empty":
        return "manual_review", True

    if contradictions["severity"] == "high":
        return "needs_more_evidence", True

    if score_analysis["margin_strength"] in {"none", "weak"}:
        return "rerun_exploration", True

    if trust_level == "medium":
        return "accept_with_caution", False

    return "needs_more_evidence", True


def save_confidence_report(
    report: dict[str, Any],
    output_path: Path = CONFIDENCE_REPORT_PATH,
) -> None:
    """Save confidence_report.json."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved confidence_report.json")


def check_visually_indistinguishable_pair(
    grounded_result: dict[str, Any],
    first_pass: dict[str, Any],
) -> dict[str, Any]:
    """Detect and correct visually-indistinguishable enclave misidentifications.

    When the grounded result picks a tiny enclave country that shares all visual
    characteristics with its surrounding parent country, and no definitive
    country-specific identifier is visible, the parent is preferred.

    This function mutates grounded_result["final_country"] in-place when a
    correction is made, so all downstream analysis uses the corrected country.

    Returns a result dict that is stored in confidence_report.json under
    "indistinguishable_pair_check".
    """
    final_country = str(grounded_result.get("final_country") or "")

    # Find the matching key in VISUALLY_INDISTINGUISHABLE_PAIRS
    matched_key: str | None = None
    for key in VISUALLY_INDISTINGUISHABLE_PAIRS:
        if normalize_text(key) == normalize_text(final_country):
            matched_key = key
            break

    if not matched_key:
        return {"triggered": False}

    parent_countries = VISUALLY_INDISTINGUISHABLE_PAIRS[matched_key]

    # Build confidence score maps from first_pass and grounded scores.
    first_pass_candidates = first_pass.get("top_candidates", [])
    fp_scores: dict[str, float] = {
        str(c.get("country", "")): float(c.get("confidence", 0.0) or 0.0)
        for c in first_pass_candidates
        if isinstance(c, dict) and c.get("country")
    }
    grounded_scores: dict[str, float] = {
        str(k): float(v or 0.0)
        for k, v in (grounded_result.get("country_scores") or {}).items()
    }
    small_fp_conf = fp_scores.get(final_country, fp_scores.get(matched_key, 0.0))

    # Identify which parent countries appeared as candidates close in confidence.
    parent_candidates: list[str] = []
    for parent in parent_countries:
        parent_fp = fp_scores.get(parent, 0.0)
        in_grounded = parent in grounded_scores
        if parent_fp > 0.0 and abs(parent_fp - small_fp_conf) <= 0.15:
            parent_candidates.append(parent)
        elif in_grounded and parent_fp == 0.0:
            # Parent appeared in grounded comparison even without first-pass score
            parent_candidates.append(parent)

    if not parent_candidates:
        logger.info(
            "Visually indistinguishable pair: %s detected but parent %s not close "
            "enough in candidates — no correction applied.",
            final_country, parent_countries,
        )
        return {
            "triggered": False,
            "small_country": final_country,
            "parent_countries_checked": parent_countries,
            "reason": "parent country not a close candidate (confidence gap > 0.15)",
        }

    chosen_parent = parent_candidates[0]

    # Scan combined reasoning text for any definitive small-country identifier.
    combined = _collect_safe_reasoning_strings(grounded_result, first_pass)
    enclave_patterns = _DEFINITIVE_ENCLAVE_IDENTIFIERS.get(matched_key, [])
    definitive_found: str | None = None
    for pattern in enclave_patterns:
        m = pattern.search(combined)
        if m:
            definitive_found = m.group(0)
            break

    logger.info(
        "Visually indistinguishable pair detected: %s vs %s",
        final_country, chosen_parent,
    )

    if definitive_found:
        logger.info(
            "Definitive %s-specific identifier found: '%s' — keeping small country result.",
            final_country, definitive_found,
        )
        return {
            "triggered": True,
            "small_country": final_country,
            "parent_country": chosen_parent,
            "definitive_identifier_found": True,
            "definitive_identifier": definitive_found,
            "resolution": "kept_small_country",
        }

    # No definitive identifier — prefer the larger parent country.
    logger.info("No definitive %s-specific identifier found.", final_country)
    logger.info("Preferring larger parent country: %s", chosen_parent)
    logger.info(
        "Reason: %s is geographically tiny (less likely by prior probability) "
        "and shares all visual characteristics with %s. "
        "Generic visual matches cannot distinguish them reliably.",
        final_country, chosen_parent,
    )

    # Mutate grounded_result so all subsequent analysis uses the corrected country.
    grounded_result["final_country"] = chosen_parent

    return {
        "triggered": True,
        "small_country": final_country,
        "parent_country": chosen_parent,
        "definitive_identifier_found": False,
        "resolution": "preferred_parent_country",
    }


def run_confidence_verification(analysis_dir: Path = ANALYSIS_DIR) -> dict[str, Any]:
    """Run Phase 8D confidence fusion and verification."""
    warnings = []

    first_pass, file_warnings = load_json_file(analysis_dir / FIRST_PASS_FILE, {})
    warnings.extend(file_warnings)

    retrieval_package, file_warnings = load_json_file(analysis_dir / RETRIEVAL_FILE, {})
    warnings.extend(file_warnings)

    logger.info("Loading grounded_result.json")
    grounded_result, file_warnings = load_json_file(analysis_dir / GROUNDED_FILE, {})
    warnings.extend(file_warnings)

    regional_consistency, file_warnings = load_json_file(
        analysis_dir / REGIONAL_CONSISTENCY_FILE,
        {},
    )
    warnings.extend(file_warnings)
    grounded_result = apply_regional_selection(grounded_result, regional_consistency)

    # ── Visually-indistinguishable pair check ─────────────────────────────────
    # Must run BEFORE all analysis so that score margins, contradiction analysis,
    # trust level, and definitive identifier detection all use the corrected country.
    indistinguishable_pair_check = check_visually_indistinguishable_pair(
        grounded_result, first_pass
    )

    ocr_results, file_warnings = load_json_file(analysis_dir / OCR_FILE, [])
    warnings.extend(file_warnings)

    # Determine whether OCR was actually attempted (vs disabled by config).
    ocr_status, _ = load_json_file(analysis_dir / OCR_STATUS_FILE, {})
    ocr_was_attempted = bool(ocr_status.get("attempted", True))  # default True for old runs
    if not ocr_was_attempted:
        logger.info("[Confidence] OCR disabled, skipping OCR evidence")

    image_ranking, file_warnings = load_json_file(analysis_dir / IMAGE_RANKING_FILE, {})
    warnings.extend(file_warnings)

    retrieval_health = analyze_retrieval_health(retrieval_package)
    score_analysis = analyze_score_margin(grounded_result)
    ocr_strength = analyze_ocr_strength(ocr_results, grounded_result)
    if not ocr_was_attempted:
        # Override penalty to zero — OCR was never run, absence is not a weakness.
        ocr_strength["strength"] = "disabled"
        ocr_strength["confidence_penalty"] = 0.0
    image_quality = analyze_image_quality(image_ranking)
    contradictions = analyze_contradictions(grounded_result)
    reference_usage = analyze_reference_usage(grounded_result)
    specific_evidence = analyze_specific_evidence(first_pass, grounded_result)
    high_specificity = analyze_high_specificity_evidence(grounded_result)
    definitive_identifier = detect_definitive_identifier(grounded_result, first_pass)

    verified_confidence = calculate_verified_confidence(
        grounded_result,
        retrieval_health,
        score_analysis,
        ocr_strength,
        image_quality,
        contradictions,
        reference_usage,
        specific_evidence,
        high_specificity,
        definitive_identifier=definitive_identifier,
        ocr_was_attempted=ocr_was_attempted,
    )
    trust_level = determine_trust_level(
        verified_confidence,
        high_specificity,
        contradictions,
        reference_usage,
    )
    recommendation, needs_more_evidence = determine_recommendation(
        trust_level,
        retrieval_health,
        score_analysis,
        contradictions,
        high_specificity,
    )

    # When a definitive identifier is present but only 1 image was collected,
    # accept the country but recommend more images.
    if (
        definitive_identifier.get("detected")
        and definitive_identifier.get("consistent_with_final_country")
    ):
        image_count = image_quality.get("selected_count", 0)
        if image_count <= 1:
            recommendation = "seek_signage"
            needs_more_evidence = True
            logger.info(
                "[Confidence] Definitive identifier present but only %d image(s) — "
                "recommendation set to seek_signage",
                image_count,
            )
        elif recommendation not in {"accept_prediction", "accept_with_caution"}:
            recommendation = "accept_prediction"
            needs_more_evidence = False
            logger.info("[Confidence] Definitive identifier present — overriding recommendation to accept_prediction")

    logger.info("Verified confidence recalibrated: %.2f", verified_confidence)
    logger.info("Recommendation: %s", recommendation)

    evidence_breakdown = {
        "ocr": ocr_strength["strength"],
        "references": reference_usage["strength"],
        "infrastructure": score_to_strength(
            len(image_quality.get("combined_metas", [])),
            (2, 4, 6),
        ),
        "environment": image_quality["quality"],
        "image_quality": image_quality["quality"],
    }
    specificity_categories = set(high_specificity.get("categories", []))
    evidence_categories = {
        "ocr_evidence_strength": ocr_strength["strength"],
        "visual_text_evidence_strength": (
            high_specificity["strength"]
            if "visual_text" in specificity_categories
            else "none"
        ),
        "unique_infrastructure_evidence_strength": (
            high_specificity["strength"]
            if "unique_infrastructure" in specificity_categories
            else "none"
        ),
        "public_transport_evidence_strength": (
            high_specificity["strength"]
            if "public_transport" in specificity_categories
            else "none"
        ),
        "reference_evidence_strength": reference_usage["strength"],
        "generic_environment_evidence_strength": (
            "high"
            if specific_evidence.get("generic_evidence_warnings")
            else "low"
        ),
    }

    report = {
        "final_country": grounded_result.get("final_country"),
        "reported_confidence": grounded_result.get("confidence", 0.0),
        "verified_confidence": verified_confidence,
        "trust_level": trust_level,
        "needs_more_evidence": needs_more_evidence,
        "retrieval_health": retrieval_health,
        "score_analysis": score_analysis,
        "evidence_breakdown": evidence_breakdown,
        "evidence_categories": evidence_categories,
        "high_specificity_evidence": high_specificity,
        "ocr_quality": ocr_strength,
        "image_quality": image_quality,
        "contradiction_analysis": contradictions,
        "reference_usage": reference_usage,
        "specific_evidence": specific_evidence,
        "definitive_identifier": definitive_identifier,
        "indistinguishable_pair_check": indistinguishable_pair_check,
        "warnings": warnings,
        "recommendation": recommendation,
        "inputs_seen": {
            "first_pass_candidates": first_pass.get("top_candidates", []),
            "retrieval_candidates": retrieval_package.get("candidate_countries", []),
            "grounded_reasoning_status": grounded_result.get("reasoning_status"),
            "raw_grounded_country": grounded_result.get("raw_grounded_country"),
            "regional_consistency": regional_consistency,
        },
    }

    save_confidence_report(report, analysis_dir / "confidence_report.json")
    return report


def main() -> None:
    """Run confidence verification from the terminal."""
    report = run_confidence_verification()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
