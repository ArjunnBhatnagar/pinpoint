import json
import logging
import os
import re
import time
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

from gemini_retry import generate_content_with_fallback
from meta_knowledge_injector import get_meta_hint_for_prompt


# Phase 8C: grounded country comparison.
# This module compares live observations against the retrieval package built in
# Phase 8B. It does not perform state/province reasoning, adaptive movement,
# embeddings, or autonomous exploration.
ANALYSIS_DIR = Path("analysis")
FIRST_PASS_PATH = ANALYSIS_DIR / "first_pass.json"
RETRIEVAL_PACKAGE_PATH = ANALYSIS_DIR / "retrieval_package.json"
OCR_RESULTS_PATH = ANALYSIS_DIR / "ocr_results.json"
SELECTED_IMAGES_PATH = ANALYSIS_DIR / "selected_images.json"
GROUNDED_RESULT_PATH = ANALYSIS_DIR / "grounded_result.json"
GROUNDED_REFERENCE_DEBUG_PATH = ANALYSIS_DIR / "grounded_reference_debug.json"

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 1.0
RATE_LIMIT_BUFFER_SECONDS = 2.0   # extra buffer added on top of the API-reported retry delay
RATE_LIMIT_MAX_RETRIES = 2        # max times to retry after a 429 before giving up
MAX_IMAGE_WIDTH = 960
JPEG_QUALITY = 85

MAX_REFERENCE_IMAGES_TOTAL = 18
MAX_REFERENCE_TEXT_CHARS = 9000
MAX_OCR_LINES = 40
SUPPORTED_REFERENCE_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

logger = logging.getLogger("grounded_reasoner")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[Grounded] %(message)s"))
    logger.addHandler(handler)


DEFAULT_GROUNDED_RESULT = {
    "final_country": None,
    "confidence": 0.0,
    "region": None,
    "region_consistent": None,
    "consistency_note": "",
    "country_scores": {},
    "evidence_breakdown": {},
    "winning_evidence": [],
    "contradictions": [],
    "meta_matches": {},
    "uncertainty_notes": [],
    "generic_evidence_warnings": [],
    "missing_reference_comparisons": [],
    "reference_usage": {
        "references_provided": False,
        "reference_images_count": 0,
        "reference_texts_count": 0,
        "references_used_in_reasoning": False,
        "reference_countries": [],
        "reference_meta_categories": [],
        "comparison_summary": "No valid retrieved references were available.",
    },
    "country_reference_comparisons": [],
    "general_reasoning_evidence": [],
    "ocr_evidence": [],
    "retrieved_reference_evidence": [],
    "reference_grounding_status": "no_valid_references_available",
    "reasoning_status": "fallback",
}


def load_api_key() -> str:
    """Load Gemini API key from .env."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY was not found in your .env file.")

    return api_key


def load_json(path: Path, default: Any) -> Any:
    """Load JSON safely."""
    if not path.exists():
        logger.warning("Missing file: %s", path)
        return default

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        logger.warning("Malformed JSON in %s: %s", path, error)
        return default


def load_first_pass(analysis_dir: Path = ANALYSIS_DIR) -> dict[str, Any]:
    """Load first-pass candidate hypotheses."""
    return load_json(analysis_dir / "first_pass.json", {})


def load_retrieval_package(analysis_dir: Path = ANALYSIS_DIR) -> dict[str, Any]:
    """Load the Phase 8B retrieval package."""
    logger.info("Loading retrieval package")
    return load_json(analysis_dir / "retrieval_package.json", {})


def load_ocr_results(analysis_dir: Path = ANALYSIS_DIR) -> list[dict[str, Any]]:
    """Load OCR results, allowing empty OCR as valid input."""
    payload = load_json(analysis_dir / "ocr_results.json", [])
    return payload if isinstance(payload, list) else []


def remove_duplicate_paths(paths: list[Path]) -> list[Path]:
    """Remove duplicate paths while preserving order."""
    seen = set()
    unique_paths = []

    for path in paths:
        resolved = str(path.resolve())
        if resolved in seen:
            continue

        seen.add(resolved)
        unique_paths.append(path)

    return unique_paths


def load_selected_screenshots(analysis_dir: Path = ANALYSIS_DIR) -> list[Path]:
    """Load selected screenshot paths from analysis/selected_images.json."""
    logger.info("Loading screenshots")
    payload = load_json(analysis_dir / "selected_images.json", {})
    raw_items = payload.get("selected_screenshots", [])
    image_paths = []

    for item in raw_items:
        raw_path = item.get("path") if isinstance(item, dict) else item
        if not raw_path:
            continue

        path = Path(raw_path)
        if path.exists():
            image_paths.append(path)
        else:
            logger.warning("Selected screenshot missing: %s", path)

    return remove_duplicate_paths(image_paths)


def merge_ocr_evidence(
    ocr_results: list[dict[str, Any]],
    max_lines: int = MAX_OCR_LINES,
) -> str:
    """Merge OCR detections into concise evidence lines.

    OCR has high geographic value when it contains language, highway numbers,
    place names, or administrative signs. This function deduplicates detections
    so one repeated text fragment does not overweight the final comparison.
    """
    lines = []
    seen = set()

    for result in ocr_results:
        image_name = result.get("image_name", "unknown")
        for detection in result.get("detections", []):
            text = str(detection.get("text", "")).strip()
            if not text:
                continue

            normalized = text.casefold()
            if normalized in seen:
                continue

            seen.add(normalized)
            confidence = detection.get("confidence", 0.0)
            relevance = detection.get("relevance", "unknown")
            lines.append(
                f"- {text} | image={image_name} | "
                f"confidence={confidence} | relevance={relevance}"
            )

            if len(lines) >= max_lines:
                break

        if len(lines) >= max_lines:
            break

    return "\n".join(lines) if lines else "No reliable OCR evidence available."


def image_to_part(image_path: Path) -> types.Part:
    """Resize and compress images before Gemini upload."""
    with Image.open(image_path) as image:
        image = image.convert("RGB")

        if image.width > MAX_IMAGE_WIDTH:
            scale = MAX_IMAGE_WIDTH / image.width
            image = image.resize(
                (MAX_IMAGE_WIDTH, int(image.height * scale)),
                Image.LANCZOS,
            )

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)

    return types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg")


def collect_reference_context(
    retrieval_package: dict[str, Any],
) -> tuple[str, list[dict[str, str]], dict[str, Any]]:
    """Collect bounded text previews and reference image metadata.

    The retrieval package can grow over time. This keeps the grounded request
    bounded while preserving the strongest comparison material for Gemini.
    The debug payload records the exact files that cross the Gemini boundary.
    """
    text_sections = []
    image_refs = []
    text_reference_count = 0
    image_paths_found = []
    missing_image_paths = []
    invalid_image_paths = []
    countries = []
    meta_categories = []

    retrieved = retrieval_package.get("retrieved", {})
    if not isinstance(retrieved, dict):
        return "No retrieved references available.", [], {
            "image_paths_found": 0,
            "valid_image_files": 0,
            "text_references_found": 0,
            "attached_image_paths": [],
            "countries_included": [],
            "meta_categories_included": [],
            "missing_image_paths": [],
            "invalid_image_paths": [],
        }

    candidate_names = {
        str(candidate.get("normalized", "")): str(candidate.get("country", ""))
        for candidate in retrieval_package.get("candidate_countries", [])
        if isinstance(candidate, dict)
    }

    for country_key, country_data in retrieved.items():
        if not isinstance(country_data, dict):
            continue

        country_name = candidate_names.get(country_key, country_key)
        if country_name not in countries:
            countries.append(country_name)
        text_sections.append(f"\nCOUNTRY: {country_name}")

        for info in country_data.get("general_info", []):
            preview = str(info.get("preview", "")).strip()
            if preview:
                text_reference_count += 1
                text_sections.append(
                    f"General info from {info.get('path', 'unknown')}:\n{preview}"
                )

        metas = country_data.get("metas", {})
        if not isinstance(metas, dict):
            continue

        for meta_name, meta_payload in metas.items():
            if not isinstance(meta_payload, dict):
                continue

            if meta_name not in meta_categories:
                meta_categories.append(meta_name)

            for text_ref in meta_payload.get("texts", []):
                preview = str(text_ref.get("preview", "")).strip()
                if preview:
                    text_reference_count += 1
                    text_sections.append(
                        f"Country={country_name} meta={meta_name} text from "
                        f"{text_ref.get('path', 'unknown')}:\n{preview}"
                    )

            for image_ref in meta_payload.get("images", []):
                raw_path = image_ref.get("path")
                if not raw_path:
                    continue

                path = Path(raw_path)
                image_paths_found.append(str(path))
                if not path.exists():
                    logger.warning("Reference image missing: %s", path)
                    missing_image_paths.append(str(path))
                    continue

                if path.suffix.lower() not in SUPPORTED_REFERENCE_IMAGE_EXTENSIONS:
                    logger.warning("Unsupported reference image type: %s", path)
                    invalid_image_paths.append(str(path))
                    continue

                try:
                    with Image.open(path) as image:
                        image.verify()
                except Exception as error:
                    logger.warning("Unreadable reference image %s: %s", path, error)
                    invalid_image_paths.append(str(path))
                    continue

                if len(image_refs) >= MAX_REFERENCE_IMAGES_TOTAL:
                    continue

                image_refs.append(
                    {
                        "country": country_name,
                        "meta": meta_name,
                        "path": str(path),
                        "filename": image_ref.get("filename", path.name),
                        "source_category": image_ref.get("source_category", meta_name),
                    }
                )

    reference_text = "\n\n".join(text_sections).strip()
    if len(reference_text) > MAX_REFERENCE_TEXT_CHARS:
        reference_text = reference_text[:MAX_REFERENCE_TEXT_CHARS] + "\n[truncated]"

    debug_payload = {
        "image_paths_found": len(image_paths_found),
        "valid_image_files": len(image_refs),
        "text_references_found": text_reference_count,
        "attached_image_paths": [reference["path"] for reference in image_refs],
        "countries_included": countries,
        "meta_categories_included": meta_categories,
        "missing_image_paths": missing_image_paths,
        "invalid_image_paths": invalid_image_paths,
        "reference_image_limit": MAX_REFERENCE_IMAGES_TOTAL,
    }
    return (
        reference_text or "No retrieved reference text available.",
        image_refs,
        debug_payload,
    )


def save_reference_debug(
    payload: dict[str, Any],
    output_path: Path = GROUNDED_REFERENCE_DEBUG_PATH,
) -> None:
    """Save the exact retrieved references prepared for grounded reasoning."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_comparison_prompt(
    first_pass: dict[str, Any],
    retrieval_package: dict[str, Any],
    ocr_evidence: str,
    reference_text: str,
    selected_screenshots: list[Path],
    reference_images: list[dict[str, str]],
    reference_debug: dict[str, Any],
) -> str:
    """Build the grounded evidence-comparison prompt."""
    logger.info("Comparing candidate countries")
    candidate_countries = retrieval_package.get("candidate_countries", [])
    detected_metas = retrieval_package.get("detected_meta_categories", [])
    meta_normalization = retrieval_package.get("meta_normalization", {})

    return f"""
You are an evidence-based environmental and infrastructure comparison analyst.

Task:
Determine which candidate country is best supported by the evidence.

Do NOT treat this as open-ended country guessing.
Do NOT invent exact coordinates.
Do NOT use unsupported certainty.
Compare live observations against the retrieved references and first-pass
hypotheses. The final result must be based on evidence support, not intuition.

Candidate countries from first pass:
{json.dumps(candidate_countries, indent=2)}

Canonical retrieval meta categories:
{json.dumps(detected_metas, indent=2)}

Visual observations:
{json.dumps(meta_normalization.get("visual_observations", first_pass.get("visual_observations", [])), indent=2)}

Evidence tags:
{json.dumps(meta_normalization.get("evidence_tags", first_pass.get("evidence_tags", [])), indent=2)}

First-pass evidence and uncertainty:
{json.dumps({
    "strongest_evidence": first_pass.get("strongest_evidence", []),
    "contradictory_evidence": first_pass.get("contradictory_evidence", []),
    "uncertainty_notes": first_pass.get("uncertainty_notes", []),
    "environmental_analysis": first_pass.get("environmental_analysis", {}),
    "infrastructure_analysis": first_pass.get("infrastructure_analysis", {}),
    "road_system_analysis": first_pass.get("road_system_analysis", {}),
}, indent=2)}

OCR evidence:
{ocr_evidence}

Retrieved reference text:
{reference_text}

LIVE SCREENSHOTS:
{json.dumps([f"live_{index}: {path.name}" for index, path in enumerate(selected_screenshots, start=1)], indent=2)}

REFERENCE IMAGES:
{json.dumps([
    f"reference_{index}: {reference['country']} / {reference['meta']} / {reference['filename']}"
    for index, reference in enumerate(reference_images, start=1)
], indent=2)}

REFERENCE AVAILABILITY:
{json.dumps({
    "references_provided": bool(reference_images or reference_debug.get("text_references_found")),
    "reference_images_count": len(reference_images),
    "reference_texts_count": reference_debug.get("text_references_found", 0),
    "reference_countries": reference_debug.get("countries_included", []),
    "reference_meta_categories": reference_debug.get("meta_categories_included", []),
}, indent=2)}

For each candidate country, compare the live screenshots against the provided
reference images across ALL of these 13 categories where references exist:
- BOLLARDS: bollard style, color, material, shape
- UTILITY POLES: pole material (wood/concrete/metal), crossbar design, wire style
- TRAFFIC LIGHTS: housing shape, mounting style, pedestrian signal style
- TERRAIN: landscape type, elevation, soil color, environment
- ROAD SIGNS: sign shape, color scheme, language, font style
- LICENSE PLATES: color scheme, format, alphanumeric pattern
- LANGUAGE/SCRIPT: writing system visible on any text
- VEGETATION/FLORA: plant species, density, climate indicators
- ARCHITECTURE: construction materials, roof style, window style
- ROAD SURFACE: material, quality, color, condition
- STREET MARKINGS: line colors, patterns, arrow styles
- STREET NAMES: sign format, color, font, language
- SIGNPOSTS: color scheme, font, format style

For each category where reference images ARE provided, explicitly state whether
the live screenshot MATCHES, PARTIALLY MATCHES, or DOES NOT MATCH the reference.
A country with more category matches across these 13 dimensions is more likely
correct. Do not score a category match unless reference images for that category
were actually provided.

Scoring guidance:
- Score each candidate independently.
- OCR with highway numbers, place names, country names, or distinctive language
  can be very strong evidence.
- Prioritize route/highway markers, street text, country-specific road
  markings, utility poles, signs, license plates, distinctive vehicles, unique
  architecture, traffic light style, bollard style, and strong retrieved-reference
  matches across all 13 categories.
- Never penalize a candidate merely because reference images are absent for
  a given category.
- Generic vegetation, generic European-style roads, generic architecture, and
  generic sunny climate are weak evidence. They must not carry a high score.
- If references are missing for a meta, state that comparison was unavailable.
  Do not imply a reference match was performed.
- Visual observations are descriptive clues, not guaranteed database
  categories. Retrieved references are the actual comparison evidence. Use
  visual observations to guide reasoning, but do not pretend they were
  reference matches unless references exist.
- If retrieved references do not match strongly, say so clearly and lower
  confidence. Separate internal visual intuition from grounded reference use.
- If reference images/text are provided, you must compare against them
  explicitly. Do not say no references were provided unless the reference list
  is empty.
- If references exist but match weakly, state: "References were provided, but
  weakly matched."
- Identify contradictions explicitly and subtract support where appropriate.
- Explain why matching references match and why non-matching references do not.

REGION SELF-CONSISTENCY CHECK (mandatory):

After determining your final_country, cross-check it against the geographic
region that the visual evidence suggests:
- "region": the broad geographic region your final_country belongs to
  (e.g. "Middle East", "North Africa", "Southern Europe").
- "region_consistent": true if your final_country is genuinely consistent
  with the visual region cues in the live screenshots; false if the country
  is a stretch given what the images show (e.g. the images look Central Asian
  but you chose Morocco, which is North Africa — set region_consistent = false).
- "consistency_note": one sentence explaining why region fits or does not fit.

Do NOT silently output a country that contradicts the visual region evidence.
If region_consistent must be false, say so clearly and explain in
consistency_note. The downstream system will use this flag to correct the
selection. It is better to be honest here than to hide a mismatch.

Return STRICT JSON only:
{{
  "final_country": "Country name or null",
  "confidence": 0.0,
  "region": "Region name",
  "region_consistent": true,
  "consistency_note": "One sentence why the region fits or does not fit.",
  "country_scores": {{
    "Country name": 0
  }},
  "evidence_breakdown": {{
    "Country name": {{
      "ocr": {{"score": 0, "evidence": [], "contradictions": []}},
      "utility_poles": {{"score": 0, "evidence": [], "contradictions": []}},
      "road_markings": {{"score": 0, "evidence": [], "contradictions": []}},
      "signs": {{"score": 0, "evidence": [], "contradictions": []}},
      "environment": {{"score": 0, "evidence": [], "contradictions": []}},
      "infrastructure": {{"score": 0, "evidence": [], "contradictions": []}},
      "total": 0
    }}
  }},
  "winning_evidence": [],
  "contradictions": [],
  "meta_matches": {{
    "utility_poles": {{
      "best_match": "Country name",
      "strength": "low|medium|high",
      "reason": "why this reference evidence matches"
    }}
  }},
  "uncertainty_notes": [],
  "generic_evidence_warnings": [],
  "missing_reference_comparisons": [],
  "reference_usage": {{
    "references_provided": true,
    "reference_images_count": 0,
    "reference_texts_count": 0,
    "references_used_in_reasoning": true,
    "reference_countries": [],
    "reference_meta_categories": [],
    "comparison_summary": "References were provided and compared."
  }},
  "country_reference_comparisons": [
    {{
      "country": "Country name",
      "reference_match_score": 0.0,
      "matched_reference_categories": [],
      "strong_matches": [],
      "weak_matches": [],
      "contradictions": []
    }}
  ],
  "general_reasoning_evidence": [],
  "ocr_evidence": [],
  "retrieved_reference_evidence": [],
  "reference_grounding_status": "references_provided_and_compared|references_provided_but_weakly_matched|no_valid_references_available",
  "reasoning_status": "ok"
}}
""".strip()


def parse_grounded_response(response_text: str) -> dict[str, Any]:
    """Parse Gemini JSON, recovering from markdown fences or surrounding text."""
    text = (response_text or "").strip()

    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1).replace("JSON\n", "", 1)

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in grounded reasoning response.")

    return json.loads(text[start : end + 1])


def validate_grounded_result(
    payload: dict[str, Any],
    retrieval_package: dict[str, Any],
) -> dict[str, Any]:
    """Ensure grounded result is machine-readable and bounded."""
    validated = deepcopy(DEFAULT_GROUNDED_RESULT)
    if isinstance(payload, dict):
        validated.update(payload)

    try:
        validated["confidence"] = max(
            0.0,
            min(1.0, float(validated.get("confidence", 0.0))),
        )
    except (TypeError, ValueError):
        validated["confidence"] = 0.0

    # Normalise region consistency fields added by the updated prompt
    if not isinstance(validated.get("region"), (str, type(None))):
        validated["region"] = None
    if validated.get("region_consistent") not in (True, False, None):
        validated["region_consistent"] = None
    if not isinstance(validated.get("consistency_note"), str):
        validated["consistency_note"] = ""

    if not isinstance(validated.get("country_scores"), dict):
        validated["country_scores"] = {}

    if not isinstance(validated.get("evidence_breakdown"), dict):
        validated["evidence_breakdown"] = {}

    for list_key in [
        "winning_evidence",
        "contradictions",
        "uncertainty_notes",
        "generic_evidence_warnings",
        "missing_reference_comparisons",
        "country_reference_comparisons",
        "general_reasoning_evidence",
        "ocr_evidence",
        "retrieved_reference_evidence",
    ]:
        if not isinstance(validated.get(list_key), list):
            validated[list_key] = []

    if not isinstance(validated.get("reference_usage"), dict):
        validated["reference_usage"] = deepcopy(DEFAULT_GROUNDED_RESULT["reference_usage"])

    if not isinstance(validated.get("meta_matches"), dict):
        validated["meta_matches"] = {}

    if not validated.get("country_scores"):
        for candidate in retrieval_package.get("candidate_countries", []):
            country = candidate.get("country")
            if country:
                validated["country_scores"][country] = 0

    return validated


def apply_reference_grounding_metadata(
    grounded_result: dict[str, Any],
    reference_debug: dict[str, Any],
) -> dict[str, Any]:
    """Make reference availability deterministic instead of model-inferred."""
    image_count = int(reference_debug.get("valid_image_files", 0) or 0)
    text_count = int(reference_debug.get("text_references_found", 0) or 0)
    references_provided = image_count > 0 or text_count > 0
    comparisons = grounded_result.get("country_reference_comparisons", [])
    retrieved_evidence = grounded_result.get("retrieved_reference_evidence", [])
    model_usage = grounded_result.get("reference_usage", {})
    references_used = bool(
        references_provided
        and (
            model_usage.get("references_used_in_reasoning")
            or comparisons
            or retrieved_evidence
        )
    )

    if not references_provided:
        grounded_result["confidence"] = min(grounded_result.get("confidence", 0.0), 0.45)
        status = "no_valid_references_available"
        summary = "No valid retrieved references were available for comparison."
    elif references_used:
        status = "references_provided_and_compared"
        summary = "References were provided and compared."
    else:
        status = "references_provided_but_weakly_matched"
        summary = "References were provided, but weakly matched."
        grounded_result["confidence"] = min(grounded_result.get("confidence", 0.0), 0.7)

    grounded_result["reference_usage"] = {
        "references_provided": references_provided,
        "reference_images_count": image_count,
        "reference_texts_count": text_count,
        "references_used_in_reasoning": references_used,
        "reference_countries": reference_debug.get("countries_included", []),
        "reference_meta_categories": reference_debug.get("meta_categories_included", []),
        "comparison_summary": summary,
    }
    grounded_result["reference_grounding_status"] = status
    return grounded_result


def fallback_grounded_result(
    error: Exception | str,
    retrieval_package: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a safe grounded result if comparison cannot run."""
    payload = validate_grounded_result({}, retrieval_package or {})
    payload["reasoning_status"] = "fallback"
    payload["uncertainty_notes"] = [
        "Grounded comparison failed or had insufficient inputs.",
        str(error),
    ]
    return payload


def save_grounded_result(
    payload: dict[str, Any],
    output_path: Path = GROUNDED_RESULT_PATH,
) -> None:
    """Save grounded_result.json."""
    logger.info("Saving grounded_result.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


_RETRY_DELAY_PATTERN = re.compile(r"retry in (\d+(?:\.\d+)?)s", re.I)


def _is_rate_limit_error(error: Exception) -> bool:
    """Return True if this error is a 429 / RESOURCE_EXHAUSTED."""
    msg = str(error).lower()
    return "429" in msg or "resource_exhausted" in msg or "quota" in msg


def _parse_retry_delay(error: Exception) -> float:
    """Extract the suggested retry delay from a 429 error message."""
    m = _RETRY_DELAY_PATTERN.search(str(error))
    if m:
        return float(m.group(1)) + RATE_LIMIT_BUFFER_SECONDS
    return 30.0 + RATE_LIMIT_BUFFER_SECONDS  # safe fallback


def _call_grounded_with_429_retry(
    client: Any,
    contents: list[Any],
    config: Any,
    n_reference_images: int,
) -> tuple[Any, str, bool]:
    """Call Gemini with explicit 429 handling that preserves the images list.

    Returns (response, model_used, rate_limited).
    The images in `contents` are NEVER dropped on retry.
    """
    import time as _time

    rate_limited = False
    last_error: Exception | None = None

    for retry_num in range(RATE_LIMIT_MAX_RETRIES + 1):
        try:
            response, model_used = generate_content_with_fallback(
                client=client,
                contents=contents,   # same contents every retry
                config=config,
                primary_model=MODEL_NAME,
                attempts_per_model=MAX_RETRIES,
                base_delay_seconds=RETRY_DELAY_SECONDS,
                logger=logger,
                operation_name="Grounded reasoning",
            )
            return response, model_used, rate_limited

        except Exception as error:
            if not _is_rate_limit_error(error):
                raise  # non-429 errors propagate immediately

            rate_limited = True
            last_error = error
            wait = _parse_retry_delay(error)
            logger.warning(
                "[Grounded] Rate limited. Waiting %.1fs before retry "
                "(attempt %d/%d)...",
                wait,
                retry_num + 1,
                RATE_LIMIT_MAX_RETRIES,
            )
            logger.info(
                "[Grounded] Retrying with %d reference images attached.",
                n_reference_images,
            )
            _time.sleep(wait)

    # All retries exhausted
    logger.warning(
        "[Grounded] WARNING: Gemini rate limit exhausted. "
        "Reference images could not be sent. "
        "Results will be based on visual reasoning only."
    )
    raise RuntimeError(
        f"Gemini rate limit exhausted after {RATE_LIMIT_MAX_RETRIES} retries: {last_error}"
    )


def run_grounded_reasoning(analysis_dir: Path = ANALYSIS_DIR) -> dict[str, Any]:
    """Run Phase 8C grounded country comparison."""
    retrieval_package = {}
    reference_debug = {}

    try:
        first_pass = load_first_pass(analysis_dir)
        retrieval_package = load_retrieval_package(analysis_dir)
        ocr_results = load_ocr_results(analysis_dir)
        selected_screenshots = load_selected_screenshots(analysis_dir)

        if not retrieval_package:
            raise FileNotFoundError("retrieval_package.json is missing or empty.")

        if not retrieval_package.get("candidate_countries"):
            raise ValueError("No candidate countries available for grounded comparison.")

        if not selected_screenshots:
            raise FileNotFoundError("No selected screenshots available for grounded comparison.")

        ocr_evidence = merge_ocr_evidence(ocr_results)
        logger.info("Retrieval package loaded")
        reference_text, reference_images, reference_debug = collect_reference_context(
            retrieval_package
        )
        retrieval_summary = retrieval_package.get("summary", {})
        reference_debug["retrieval_summary_reported_images"] = int(
            retrieval_summary.get("total_images", 0) or 0
        )
        reference_debug["retrieval_summary_reported_texts"] = int(
            retrieval_summary.get("total_texts", 0) or 0
        )
        reference_debug["retrieval_package_path"] = str(
            (analysis_dir / "retrieval_package.json").resolve()
        )
        save_reference_debug(
            reference_debug,
            analysis_dir / "grounded_reference_debug.json",
        )
        logger.info("Reference images found: %s", reference_debug["image_paths_found"])
        logger.info("Reference texts found: %s", reference_debug["text_references_found"])
        logger.info("Valid reference images attached: %s", reference_debug["valid_image_files"])
        if (
            reference_debug["retrieval_summary_reported_images"]
            != reference_debug["image_paths_found"]
            or reference_debug["retrieval_summary_reported_texts"]
            != reference_debug["text_references_found"]
        ):
            logger.warning(
                "Retrieval summary count differs from stored entries: "
                "summary=%s images/%s texts, stored=%s images/%s texts",
                reference_debug["retrieval_summary_reported_images"],
                reference_debug["retrieval_summary_reported_texts"],
                reference_debug["image_paths_found"],
                reference_debug["text_references_found"],
            )
        logger.info(
            "Reference countries: %s",
            ", ".join(reference_debug["countries_included"]) or "none",
        )
        logger.info(
            "Reference meta categories: %s",
            ", ".join(reference_debug["meta_categories_included"]) or "none",
        )
        if not reference_images and not reference_debug["text_references_found"]:
            logger.warning("retrieval_package contains no valid references")

        prompt = build_comparison_prompt(
            first_pass,
            retrieval_package,
            ocr_evidence,
            reference_text,
            selected_screenshots,
            reference_images,
            reference_debug,
        )

        # Inject meta knowledge specifically for the candidate countries.
        # Includes tier1 + tier2 identifiers and pairwise elimination clues
        # for any two candidates that are known common confusions of each other.
        candidate_names = [
            c.get("country", "")
            for c in retrieval_package.get("candidate_countries", [])
            if isinstance(c, dict) and c.get("country")
        ]
        if candidate_names:
            meta_hint = get_meta_hint_for_prompt(candidate_names, include_tier2=True)
            if meta_hint:
                prompt = prompt + "\n\n" + meta_hint

        logger.info("Evaluating pole evidence")
        logger.info("Evaluating OCR evidence")
        logger.info("Ranking countries")

        client = genai.Client(api_key=load_api_key())
        contents: list[Any] = [prompt]

        for index, screenshot_path in enumerate(selected_screenshots, start=1):
            contents.append(f"Live screenshot {index}: {screenshot_path.name}")
            contents.append(image_to_part(screenshot_path))

        for index, reference in enumerate(reference_images, start=1):
            reference_path = Path(reference["path"])
            contents.append(
                "Retrieved reference "
                f"{index}: country={reference['country']}, "
                f"meta={reference['meta']}, file={reference_path.name}"
            )
            contents.append(image_to_part(reference_path))

        grounded_config = types.GenerateContentConfig(
            temperature=0.0,
            system_instruction=(
                "You perform grounded environmental comparison. "
                "Compare only the provided candidate countries and references. "
                "When references are provided, compare them explicitly and never "
                "claim that references were absent. "
                "Return strict JSON only."
            ),
        )

        response, model_used, rate_limited = _call_grounded_with_429_retry(
            client=client,
            contents=contents,
            config=grounded_config,
            n_reference_images=len(reference_images),
        )
        logger.info("Grounded reasoning succeeded with model=%s", model_used)
        parsed = parse_grounded_response(response.text or "{}")
        validated = validate_grounded_result(parsed, retrieval_package)
        validated = apply_reference_grounding_metadata(validated, reference_debug)
        validated["rate_limited"] = rate_limited
        validated["images_actually_sent"] = len(reference_images)
        validated["fallback_reason"] = None
        if rate_limited:
            validated["warning"] = (
                "Gemini rate limit hit during grounded comparison. "
                "Reference images may not have been used. Results are less reliable."
            )
            logger.warning(
                "[Grounded] WARNING: Rate limit hit — reference images may not have been used."
            )
        if validated.get("generic_evidence_warnings"):
            logger.info("Generic evidence detected; confidence penalty applied downstream")
        save_grounded_result(validated, analysis_dir / "grounded_result.json")
        return validated
    except Exception as error:
        is_rate_limit = _is_rate_limit_error(error)
        logger.warning("Grounded reasoning fallback: %s", error)
        payload = fallback_grounded_result(error, retrieval_package)
        if reference_debug:
            payload = apply_reference_grounding_metadata(payload, reference_debug)
        payload["rate_limited"] = is_rate_limit
        payload["images_actually_sent"] = len(reference_images) if reference_images else 0
        payload["fallback_reason"] = "rate_limit" if is_rate_limit else "error"
        if is_rate_limit:
            payload["warning"] = (
                "Gemini rate limit hit during grounded comparison. "
                "Reference images were not used. Results are less reliable."
            )
            logger.warning(
                "[Grounded] WARNING: Gemini rate limit exhausted. "
                "Reference images could not be sent. "
                "Results will be based on visual reasoning only."
            )
        save_grounded_result(payload, analysis_dir / "grounded_result.json")
        return payload


def main() -> None:
    """Run grounded comparison from the terminal."""
    result = run_grounded_reasoning()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
