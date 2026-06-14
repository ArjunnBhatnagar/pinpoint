import json
import logging
import os
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
from meta_normalizer import apply_meta_normalization


# Phase 8A: first-pass geographic hypothesis generation.
# This module does not perform retrieval, grounded comparison, state/province
# reasoning, or final country locking. It produces structured, uncertainty-aware
# candidate hypotheses that later systems can consume.
ANALYSIS_DIR = Path("analysis")
FIRST_PASS_PATH = ANALYSIS_DIR / "first_pass.json"

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 1.0
MAX_IMAGE_WIDTH = 960
JPEG_QUALITY = 85

logger = logging.getLogger("country_reasoner")
logging.basicConfig(level=logging.INFO, format="[Reasoner] %(message)s")


DEFAULT_REASONING_OUTPUT = {
    "top_candidates": [],
    "confidence_scores": {},
    "strongest_region": None,
    "region_reasoning": "",
    "strongest_evidence": [],
    "contradictory_evidence": [],
    "uncertainty_notes": [],
    "detected_meta_categories": [],
    "visible_specific_metas": [],
    "visual_observations": [],
    "retrieval_meta_categories": [],
    "evidence_tags": [],
    "weak_or_uncertain_metas": [],
    "missing_high_value_metas": [],
    "environmental_analysis": {
        "terrain": None,
        "climate": None,
        "vegetation": None,
        "environmental_conditions": None,
    },
    "OCR_analysis": {
        "merged_text": "",
        "high_value_clues": [],
        "weak_or_ignored_clues": [],
        "language_patterns": [],
    },
    "infrastructure_analysis": {
        "utility_poles": None,
        "road_quality": None,
        "architecture": None,
        "infrastructure_style": None,
        "vehicle_clues": None,
    },
    "road_system_analysis": {
        "road_markings": None,
        "driving_side_clues": None,
        "highway_markings": None,
        "signage_systems": None,
    },
    "missing_evidence": [],
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
    """Load JSON safely, returning a default on missing or malformed files."""
    if not path.exists():
        return default

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        logger.warning("Could not load %s: %s", path, error)
        return default


def load_selected_images(analysis_dir: Path = ANALYSIS_DIR) -> list[Path]:
    """Load selected screenshots from analysis/selected_images.json."""
    logger.info("Loading selected screenshots")
    payload = load_json(analysis_dir / "selected_images.json", {})
    raw_items = payload.get("selected_screenshots", [])
    image_paths = []

    for item in raw_items:
        if isinstance(item, dict):
            raw_path = item.get("path")
        else:
            raw_path = item

        if not raw_path:
            continue

        path = Path(raw_path)
        if path.exists():
            image_paths.append(path)

    return remove_duplicate_paths(image_paths)


def remove_duplicate_paths(paths: list[Path]) -> list[Path]:
    """Remove duplicate image paths while preserving order."""
    seen = set()
    unique_paths = []

    for path in paths:
        resolved = str(path.resolve())
        if resolved in seen:
            continue

        seen.add(resolved)
        unique_paths.append(path)

    return unique_paths


def load_ocr_results(analysis_dir: Path = ANALYSIS_DIR) -> list[dict]:
    """Load OCR results from analysis/ocr_results.json."""
    logger.info("Loading OCR results")
    payload = load_json(analysis_dir / "ocr_results.json", [])

    if isinstance(payload, list):
        return payload

    return []


def load_image_ranking(analysis_dir: Path = ANALYSIS_DIR) -> dict:
    """Load selector ranking data for detected metas and scores."""
    return load_json(analysis_dir / "image_ranking.json", {})


def merge_ocr_text(ocr_results: list[dict], min_confidence: float = 0.45) -> str:
    """Merge OCR text, ignoring duplicates and low-confidence detections."""
    seen = set()
    merged_lines = []

    for result in ocr_results:
        image_name = result.get("image_name", "unknown_image")

        for detection in result.get("detections", []):
            text = str(detection.get("text", "")).strip()
            confidence = float(detection.get("confidence", 0.0) or 0.0)
            relevance = detection.get("relevance", "unknown")

            if not text or confidence < min_confidence:
                continue

            normalized = text.casefold()
            if normalized in seen:
                continue

            seen.add(normalized)
            merged_lines.append(
                f"- {text} "
                f"(image={image_name}, confidence={confidence:.2f}, "
                f"relevance={relevance})"
            )

    if not merged_lines:
        return "No reliable OCR text was available."

    return "\n".join(merged_lines)


def extract_selector_metas(image_ranking: dict) -> dict:
    """Extract detected metas and scores from selector outputs."""
    selected_payload = load_json(ANALYSIS_DIR / "selected_images.json", {})
    selected_items = selected_payload.get("selected_screenshots", [])
    combined_metas = set(selected_payload.get("combined_visible_metas", []))
    image_summaries = []

    for item in selected_items:
        if not isinstance(item, dict):
            continue

        gemini = item.get("gemini", {})
        metas = gemini.get("detected_metas", [])
        combined_metas.update(metas)
        image_summaries.append(
            {
                "path": item.get("path"),
                "final_score": item.get("final_score"),
                "rejection_reasons": item.get("rejection_reasons", []),
                "detected_metas": metas,
                "ranking_explanation": item.get("ranking_explanation", ""),
            }
        )

    return {
        "combined_metas": sorted(combined_metas),
        "image_summaries": image_summaries,
        "ranking_available": bool(image_ranking),
    }


def analyze_environment(selector_context: dict) -> dict:
    """Prepare environmental context from selector metas for the prompt."""
    return {
        "known_metas": selector_context.get("combined_metas", []),
        "note": "Gemini should verify all metas visually and preserve uncertainty.",
    }


def analyze_infrastructure(selector_context: dict) -> dict:
    """Prepare infrastructure context from selector metas for the prompt."""
    infrastructure_terms = [
        meta
        for meta in selector_context.get("combined_metas", [])
        if any(
            keyword in meta
            for keyword in [
                "pole",
                "road",
                "bollard",
                "sign",
                "lane",
                "paint",
                "architecture",
                "infrastructure",
            ]
        )
    ]

    return {
        "infrastructure_metas": infrastructure_terms,
        "note": "Infrastructure is often geographically informative but not definitive.",
    }


def image_to_part(image_path: Path) -> types.Part:
    """Resize/compress image before Gemini upload."""
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


def build_reasoning_prompt(
    selected_images: list[Path],
    merged_ocr_text: str,
    selector_context: dict,
) -> str:
    """Build a strict first-pass reasoning prompt for Gemini."""
    logger.info("Building structured reasoning prompt")
    environment_context = analyze_environment(selector_context)
    infrastructure_context = analyze_infrastructure(selector_context)

    return f"""
You are a geographic investigator producing FIRST-PASS country hypotheses.

Do NOT finalize a single country. Do NOT hallucinate exact locations.
Do NOT use retrieval, reference matching, state/province reasoning, or grounded
comparison. Preserve multiple plausible countries and uncertainty.

Analyze and report what you observe for each of these 13 visual categories.
For categories not visible or not applicable, say "not visible":

1. BOLLARDS: Any bollard posts visible? Describe color, material, shape, style.
2. UTILITY POLES: Any utility/electric poles? Describe material (wood/concrete/metal),
   crossbar style, wire attachment.
3. TRAFFIC LIGHTS: Any traffic lights? Describe color scheme, mounting style,
   shape of housing, pedestrian signal style.
4. TERRAIN: What is the landscape? Describe flatness, elevation, rock type,
   soil color, overall environment.
5. ROAD SIGNS: Any road signs? Describe shape, color scheme, text language,
   font style, mounting style.
6. LICENSE PLATES: Any vehicle license plates? Describe color scheme, format,
   any visible letters/numbers pattern.
7. LANGUAGE/SCRIPT: What language or script appears on signs, buildings, any text?
   Describe the writing system and any readable text.
8. VEGETATION/FLORA: What plants, trees, or vegetation are visible? Describe
   species if recognizable, density, and climate indicators.
9. ARCHITECTURE: What building styles are visible? Describe construction materials,
   roof styles, window styles, overall architectural character.
10. ROAD SURFACE: What is the road surface like? Describe material, quality,
    color, condition.
11. STREET MARKINGS: Any painted markings on the road? Describe line colors,
    patterns, arrow styles.
12. STREET NAMES: Any street name signs visible? Describe format, color, font,
    language.
13. SIGNPOSTS: Any direction/distance signposts? Describe color scheme, font,
    format style.

Prefer specific visible evidence over broad scenery. Do not claim bollards, signs,
plates, or text unless actually visible. Generic vegetation and sunny climate are
weak evidence unless distinctive. Explicitly list missing high-value clues.

Separate meta output into:
- visual_observations: descriptive clues such as "wooden utility poles" or
  "dry dusty ground".
- retrieval_meta_categories: ONLY canonical reference-folder names selected
  from this exact list:
  bollards, poles, traffic_lights, terrain, street_sign, license_plate,
  language, flora, architecture, road_surface, street_markings, street_name,
  signposts.
  Use these exact names — they map directly to database folders.
- evidence_tags: semantic reasoning tags such as rural, dry_climate,
  tropical_dry, or narrow_paved_road. These are not reference-folder names.

Confidence must be based on evidence strength, uniqueness, consistency between
clues, and ambiguity. Avoid unrealistic certainty.

OCR text:
{merged_ocr_text}

Selector meta context:
{json.dumps(selector_context, indent=2)}

Environmental context:
{json.dumps(environment_context, indent=2)}

Infrastructure context:
{json.dumps(infrastructure_context, indent=2)}

Selected screenshots:
{[path.name for path in selected_images]}

REGION AND SELF-CONSISTENCY RULES (follow strictly):

For every candidate country you output, you MUST also output:
- "region": the broad geographic region that country belongs to
  (e.g. "Middle East", "North Africa", "Southern Europe", "South America",
  "Southeast Asia", "Central Asia", "Eastern Europe", "Sub-Saharan Africa").
  Use a well-known region label, not a continent.
- "region_confidence": 0.0–1.0 how confident you are that this region is
  correct given the visual evidence. If uncertain, give a low value (0.2–0.4).
  Do NOT assign a high region_confidence just to fill the field.
- "region_consistent": true if this country is genuinely consistent with the
  visual region evidence in the images; false if the country is a stretch
  given what the images actually show.
- "consistency_note": one sentence explaining why the region fits or does not
  fit this candidate. Be honest — if the fit is weak, say so.

At the top level you MUST output:
- "strongest_region": the single geographic region most clearly supported by
  the visual evidence (terrain, vegetation, road style, infrastructure style,
  climate). This must reflect what the IMAGE shows, NOT just the region of
  your top country candidate. If the image shows dry steppe/rolling hills
  with a Central Asian feel but your top candidate is Morocco, strongest_region
  should reflect the image, not Morocco.
- "region_reasoning": one or two sentences explaining what visual evidence
  drove the strongest_region choice.

SELF-CONSISTENCY RULE:
  Do not silently output a country whose region contradicts the strongest_region.
  If a candidate country's region does not match strongest_region, set
  region_consistent = false and explain in consistency_note. It is valid to
  list such a candidate if evidence is mixed, but you must flag it honestly.

Return STRICT JSON only:
{{
  "top_candidates": [
    {{
      "country": "Country name",
      "confidence": 0.0,
      "region": "Region name",
      "region_confidence": 0.0,
      "region_consistent": true,
      "consistency_note": "One sentence why this region fits or does not fit.",
      "reasons": ["reason 1"],
      "weaker_because": ["why this might be wrong"]
    }}
  ],
  "confidence_scores": {{"Country name": 0.0}},
  "strongest_region": "Region name",
  "region_reasoning": "What visual evidence supports this region.",
  "strongest_evidence": [],
  "contradictory_evidence": [],
  "uncertainty_notes": [],
  "detected_meta_categories": [],
  "visible_specific_metas": [],
  "visual_observations": [],
  "retrieval_meta_categories": [],
  "evidence_tags": [],
  "weak_or_uncertain_metas": [],
  "missing_high_value_metas": [],
  "environmental_analysis": {{
    "terrain": "",
    "climate": "",
    "vegetation": "",
    "environmental_conditions": ""
  }},
  "OCR_analysis": {{
    "merged_text": "",
    "high_value_clues": [],
    "weak_or_ignored_clues": [],
    "language_patterns": []
  }},
  "infrastructure_analysis": {{
    "utility_poles": "",
    "road_quality": "",
    "architecture": "",
    "infrastructure_style": "",
    "vehicle_clues": ""
  }},
  "road_system_analysis": {{
    "road_markings": "",
    "driving_side_clues": "",
    "highway_markings": "",
    "signage_systems": ""
  }},
  "missing_evidence": [],
  "reasoning_status": "ok"
}}
""".strip()


def parse_reasoning_response(response_text: str) -> dict:
    """Parse Gemini JSON, recovering from markdown fences or surrounding text."""
    logger.info("Parsing Gemini response")
    text = response_text.strip()

    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1).replace("JSON\n", "", 1)

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in first-pass response.")

    return json.loads(text[start : end + 1])


def validate_reasoning_output(payload: dict) -> dict:
    """Ensure the reasoning output has all expected machine-readable keys."""
    validated = deepcopy(DEFAULT_REASONING_OUTPUT)
    validated.update(payload if isinstance(payload, dict) else {})

    if not isinstance(validated.get("top_candidates"), list):
        validated["top_candidates"] = []

    # Normalise top-level region fields
    if not isinstance(validated.get("strongest_region"), (str, type(None))):
        validated["strongest_region"] = None
    if not isinstance(validated.get("region_reasoning"), str):
        validated["region_reasoning"] = ""

    for candidate in validated["top_candidates"]:
        if not isinstance(candidate, dict):
            continue

        try:
            candidate["confidence"] = max(
                0.0,
                min(1.0, float(candidate.get("confidence", 0.0))),
            )
        except (TypeError, ValueError):
            candidate["confidence"] = 0.0

        # Normalise per-candidate region fields added by the new prompt
        if not isinstance(candidate.get("region"), (str, type(None))):
            candidate["region"] = None
        try:
            candidate["region_confidence"] = max(
                0.0,
                min(1.0, float(candidate.get("region_confidence", 0.0))),
            )
        except (TypeError, ValueError):
            candidate["region_confidence"] = 0.0
        if candidate.get("region_consistent") not in (True, False, None):
            candidate["region_consistent"] = None
        if not isinstance(candidate.get("consistency_note"), str):
            candidate["consistency_note"] = ""

    for meta_key in [
        "detected_meta_categories",
        "visible_specific_metas",
        "visual_observations",
        "retrieval_meta_categories",
        "evidence_tags",
        "weak_or_uncertain_metas",
        "missing_high_value_metas",
    ]:
        if not isinstance(validated.get(meta_key), list):
            validated[meta_key] = []

    return apply_meta_normalization(validated)


def save_first_pass_results(payload: dict, output_path: Path = FIRST_PASS_PATH) -> None:
    """Save first-pass reasoning JSON."""
    logger.info("Saved first_pass.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def fallback_reasoning(error: Exception | str) -> dict:
    """Return a safe machine-readable fallback if reasoning fails."""
    payload = validate_reasoning_output({})
    payload["reasoning_status"] = "fallback"
    payload["uncertainty_notes"] = [
        "First-pass country reasoning failed; downstream systems should not rely on candidates."
    ]
    payload["missing_evidence"] = [str(error)]
    return payload


def run_first_pass_reasoning(analysis_dir: Path = ANALYSIS_DIR) -> dict:
    """Run structured first-pass country hypothesis generation."""
    try:
        selected_images = load_selected_images(analysis_dir)
        ocr_results = load_ocr_results(analysis_dir)
        image_ranking = load_image_ranking(analysis_dir)
        selector_context = extract_selector_metas(image_ranking)
        merged_ocr_text = merge_ocr_text(ocr_results)

        if not selected_images:
            raise FileNotFoundError("No selected screenshots available for first-pass reasoning.")

        prompt = build_reasoning_prompt(selected_images, merged_ocr_text, selector_context)

        # Inject meta knowledge for all known countries — no candidates exist yet
        # so we provide general "what to look for" guidance across the full DB.
        meta_hint = get_meta_hint_for_prompt([])
        if meta_hint:
            prompt = prompt + "\n\n" + meta_hint

        logger.info("Sending first-pass reasoning request")
        client = genai.Client(api_key=load_api_key())
        contents = [prompt]

        for index, image_path in enumerate(selected_images, start=1):
            contents.append(f"Selected screenshot {index}: {image_path.name}")
            contents.append(image_to_part(image_path))

        response, model_used = generate_content_with_fallback(
            client=client,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.0,
                system_instruction=(
                    "You generate structured first-pass country hypotheses. "
                    "Preserve uncertainty and return strict JSON only."
                ),
            ),
            primary_model=MODEL_NAME,
            attempts_per_model=MAX_RETRIES,
            base_delay_seconds=RETRY_DELAY_SECONDS,
            logger=logger,
            operation_name="First-pass reasoning",
        )
        logger.info("First-pass reasoning succeeded with model=%s", model_used)
        parsed = parse_reasoning_response(response.text or "{}")
        validated = validate_reasoning_output(parsed)
        logger.info(
            "Visible specific metas: %s",
            ", ".join(str(item) for item in validated.get("visible_specific_metas", [])) or "none",
        )
        save_first_pass_results(validated, analysis_dir / "first_pass.json")
        logger.info("Candidate generation completed")

        # Divergence warning: when top-2 candidates are within 0.15 confidence,
        # later Gemini passes (grounded_reasoner, analyze.py) may reason
        # independently and reach different conclusions — producing inconsistent
        # pipeline output.  Log this so the frequency can be tracked.
        _fp_candidates = validated.get("top_candidates", [])
        if len(_fp_candidates) >= 2:
            _top_conf = float(_fp_candidates[0].get("confidence", 0.0) or 0.0)
            _second_conf = float(_fp_candidates[1].get("confidence", 0.0) or 0.0)
            _gap = _top_conf - _second_conf
            if _gap <= 0.15:
                logger.warning(
                    "WARNING: Top-2 first-pass candidates are very close "
                    "(%s %.2f vs %s %.2f, gap=%.2f). "
                    "Environmental analysis and structured reasoning may diverge — "
                    "pipeline results may be inconsistent across Gemini calls.",
                    _fp_candidates[0].get("country", "?"), _top_conf,
                    _fp_candidates[1].get("country", "?"), _second_conf,
                    _gap,
                )

        return validated
    except Exception as error:
        logger.warning("First-pass reasoning fallback: %s", error)
        payload = fallback_reasoning(error)
        save_first_pass_results(payload, analysis_dir / "first_pass.json")
        return payload


def main() -> None:
    """Run first-pass reasoning from the terminal."""
    result = run_first_pass_reasoning()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
