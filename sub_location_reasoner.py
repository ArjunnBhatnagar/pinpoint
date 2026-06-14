"""sub_location_reasoner.py — State/region identification after country is confirmed.

Runs after final_decision.py has confirmed a country with sufficient confidence.
Uses a regional knowledge base to narrow down the specific state or province.
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from io import BytesIO
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

from gemini_retry import generate_content_with_fallback

# ── Config ────────────────────────────────────────────────────────────────────
ANALYSIS_DIR = Path("analysis")
KNOWLEDGE_BASE_DIR = Path("knowledge_base/regions")
SUB_LOCATION_RESULT_PATH = ANALYSIS_DIR / "sub_location_result.json"

# Minimum country confidence required before attempting sub-location reasoning.
COUNTRY_CONFIDENCE_THRESHOLD = 0.75

# Minimum state confidence to report a specific state (vs just a broad region).
STATE_CONFIDENCE_DISPLAY_THRESHOLD = 0.65

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 1.0
MAX_IMAGE_WIDTH = 960
JPEG_QUALITY = 85

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger("sub_location_reasoner")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[SubLocation] %(message)s"))
    logger.addHandler(handler)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env")
    return api_key


def _country_to_kb_path(country: str) -> Path:
    """Map a country display name to its knowledge base JSON path."""
    nfkd = unicodedata.normalize("NFKD", str(country))
    slug = nfkd.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
    return KNOWLEDGE_BASE_DIR / f"{slug}.json"


def _load_json(path: Path, default: Any = None) -> Any:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def _load_selected_screenshots(analysis_dir: Path) -> list[Path]:
    """Load selected screenshot paths from selected_images.json."""
    payload = _load_json(analysis_dir / "selected_images.json", {})
    paths: list[Path] = []
    for item in payload.get("selected_screenshots", []):
        raw = item.get("path") if isinstance(item, dict) else item
        if raw:
            p = Path(raw)
            if p.exists():
                paths.append(p)
    return paths


def _image_to_part(image_path: Path) -> types.Part:
    """Resize/compress image for Gemini."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        if img.width > MAX_IMAGE_WIDTH:
            scale = MAX_IMAGE_WIDTH / img.width
            img = img.resize(
                (MAX_IMAGE_WIDTH, int(img.height * scale)),
                Image.LANCZOS,
            )
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


def _format_regional_knowledge(kb: dict[str, Any]) -> str:
    """Format the knowledge base into readable text for the Gemini prompt."""
    country = kb.get("country", "Unknown")
    lines: list[str] = [f"REGIONAL KNOWLEDGE BASE FOR {country.upper()}:", ""]

    for region in kb.get("regions", []):
        name = region.get("name", "")
        states = region.get("states", [])
        identifiers = region.get("visual_identifiers", [])
        distinguishers = region.get("state_distinguishers", {})
        route_clues = region.get("route_sign_clues", "")
        hint = region.get("gemini_hint", "")

        lines.append(f"REGION: {name}")
        lines.append(f"  States/Areas: {', '.join(states)}")
        if identifiers:
            lines.append("  Visual identifiers:")
            for item in identifiers:
                lines.append(f"    - {item}")
        if distinguishers:
            lines.append("  How to distinguish between states in this region:")
            for key, val in distinguishers.items():
                lines.append(f"    * {key.replace('_', ' ')}: {val}")
        if route_clues:
            lines.append(f"  Route sign clues: {route_clues}")
        if hint:
            lines.append(f"  KEY HINT: {hint}")
        lines.append("")

    return "\n".join(lines)


def _build_sub_location_prompt(
    country: str,
    kb: dict[str, Any],
    screenshot_names: list[str],
) -> str:
    """Build the Gemini prompt for sub-location reasoning."""
    regional_knowledge = _format_regional_knowledge(kb)
    all_states: list[str] = []
    for region in kb.get("regions", []):
        all_states.extend(region.get("states", []))

    return f"""You are a geographic expert helping identify the specific state or region within {country}.
The country has already been confirmed with high confidence. Your task is ONLY to identify the sub-location.

{regional_knowledge}

Screenshots provided: {screenshot_names}

Looking at these screenshots from {country}, identify:
1. Which broad region this appears to be (from the list above)
2. Which specific state/province if the evidence supports it
3. What specific visual evidence led to this conclusion
4. Which regions/states can be eliminated and why

Be honest about uncertainty. If you can only identify a broad region, say so.
Do not guess a specific state if the evidence is genuinely ambiguous.
A null final_state is better than a wrong specific state guess.

CRITICAL CONFIDENCE RULES — you must follow these:

1. Only assign state_confidence above 0.70 if the evidence is UNIQUE to that
   state and cannot reasonably apply to any other state.

   Examples of evidence that IS unique enough for high confidence:
   - A specific named highway that only exists in one state
     (e.g. 'Great Ocean Road' = Victoria)
   - A city name visible on a sign
   - A landmark unique to one state
   - A route number format unique to one state

   Examples of evidence that is NOT unique enough for high confidence:
   - Road numbers like A440, B123, A1 etc that exist in multiple states
   - General vegetation like 'tall trees' or 'eucalyptus forest' that
     appears across many states
   - General terrain like 'flat' or 'hilly' that applies to multiple states
   - General road infrastructure that is the same nationally

2. If the best evidence you have applies to 2 or more states equally,
   cap state_confidence at 0.50 maximum.

3. If the best evidence applies to 3 or more states equally,
   cap state_confidence at 0.35.

4. If you genuinely cannot distinguish between states, set state_confidence
   to 0.30 and set final_state to null.
   Do not guess a state just to fill the field.

5. Always list in eliminated_states any states you are ruling out and WHY
   specifically. If you cannot rule out a state, keep it in candidate_states.

6. The broad_region confidence can be higher than state_confidence — it is
   easier to identify a region than a specific state.

IMPORTANT: A wrong confident answer is much worse than an honest uncertain
answer. If you are not sure, say so. A state_confidence of 0.35 with 2-3
candidate states is a perfectly valid and useful output.

All valid states/areas for {country}: {', '.join(all_states)}

Respond in strict JSON only:
{{
  "broad_region": "region name from the knowledge base above",
  "region_confidence": 0.0,
  "candidate_states": ["state1", "state2"],
  "final_state": "specific state name or null if uncertain",
  "state_confidence": 0.0,
  "state_evidence": ["specific visual clue 1", "specific visual clue 2"],
  "eliminated_states": ["State: reason for elimination"],
  "reasoning": "brief overall reasoning"
}}
""".strip()


def _parse_sub_location_response(response_text: str) -> dict[str, Any]:
    """Parse Gemini JSON, recovering from markdown fences."""
    text = (response_text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").replace("json\n", "", 1).replace("JSON\n", "", 1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object in sub-location response")
    return json.loads(text[start:end + 1])


def _validate_response(parsed: dict[str, Any], kb: dict[str, Any]) -> dict[str, Any]:
    """Basic sanity checks — states must belong to this country."""
    all_states_lower: set[str] = set()
    for region in kb.get("regions", []):
        for s in region.get("states", []):
            all_states_lower.add(s.lower())

    # Ensure types are correct
    parsed.setdefault("broad_region", None)
    parsed.setdefault("region_confidence", 0.0)
    parsed.setdefault("candidate_states", [])
    parsed.setdefault("final_state", None)
    parsed.setdefault("state_confidence", 0.0)
    parsed.setdefault("state_evidence", [])
    parsed.setdefault("eliminated_states", [])
    parsed.setdefault("reasoning", "")

    try:
        parsed["region_confidence"] = max(0.0, min(1.0, float(parsed["region_confidence"])))
        parsed["state_confidence"] = max(0.0, min(1.0, float(parsed["state_confidence"])))
    except (TypeError, ValueError):
        parsed["region_confidence"] = 0.0
        parsed["state_confidence"] = 0.0

    if not isinstance(parsed["candidate_states"], list):
        parsed["candidate_states"] = []

    # Validate final_state is in candidate_states
    final_state = parsed.get("final_state")
    if final_state and isinstance(parsed["candidate_states"], list):
        if final_state not in parsed["candidate_states"]:
            parsed["candidate_states"].insert(0, final_state)

    return parsed


_GENERIC_EVIDENCE_TERMS = {
    "tall trees", "eucalyptus", "forest", "flat", "hilly", "dry", "wet",
    "green", "rural", "road", "asphalt", "vegetation", "landscape",
    "terrain", "trees", "bush", "scrub", "highway", "straight",
}


def _sanity_check_confidence(result: dict[str, Any]) -> dict[str, Any]:
    """Post-process Gemini's response to cap overconfident state guesses."""
    state = result.get("final_state")
    state_conf = float(result.get("state_confidence", 0.0) or 0.0)
    candidates = result.get("candidate_states", [])
    evidence = result.get("state_evidence", [])
    original_conf = state_conf
    adjusted = False
    adjustment_reason: str | None = None

    # Rule 1: 3+ candidates but high confidence
    if len(candidates) >= 3 and state_conf > 0.60:
        state_conf = 0.50
        adjusted = True
        adjustment_reason = f"3+ candidates ({len(candidates)}) but confidence was {original_conf:.2f}"

    # Rule 2: Very few evidence items but high confidence
    if len(evidence) <= 2 and state_conf > 0.65:
        new_cap = 0.55
        if state_conf > new_cap:
            state_conf = new_cap
            adjusted = True
            adjustment_reason = (
                adjustment_reason
                or f"only {len(evidence)} evidence item(s) but confidence was {original_conf:.2f}"
            )

    # Rule 3: All evidence is generic — no specific clues
    evidence_text = " ".join(evidence).lower()
    generic_hits = [t for t in _GENERIC_EVIDENCE_TERMS if t in evidence_text]
    specific_count = len(evidence) - len(
        [e for e in evidence if any(t in e.lower() for t in _GENERIC_EVIDENCE_TERMS)]
    )
    if len(evidence) > 0 and specific_count == 0 and state_conf > 0.50:
        state_conf = 0.40
        result["final_state"] = None
        adjusted = True
        adjustment_reason = (
            f"all {len(evidence)} evidence item(s) are generic "
            f"({', '.join(generic_hits[:4])}); original confidence {original_conf:.2f}"
        )

    if adjusted:
        logger.info(
            "Confidence adjusted: %.2f -> %.2f | reason: %s",
            original_conf,
            state_conf,
            adjustment_reason,
        )
        result["state_confidence"] = round(state_conf, 2)
        result["confidence_adjusted"] = True
        result["adjustment_reason"] = adjustment_reason
    else:
        result["confidence_adjusted"] = False
        result["adjustment_reason"] = None

    return result


def _minimal_result(
    country: str,
    reason: str,
    sub_location_attempted: bool = False,
    knowledge_base_used: bool = False,
) -> dict[str, Any]:
    return {
        "country": country,
        "broad_region": None,
        "region_confidence": 0.0,
        "candidate_states": [],
        "final_state": None,
        "state_confidence": 0.0,
        "state_evidence": [],
        "eliminated_states": [],
        "reasoning": reason,
        "knowledge_base_used": knowledge_base_used,
        "sub_location_attempted": sub_location_attempted,
        "skip_reason": reason,
    }


# ── Main entry point ──────────────────────────────────────────────────────────

def run_sub_location_reasoning(analysis_dir: Path = ANALYSIS_DIR) -> dict[str, Any]:
    """Attempt to identify the state/region within the confirmed country.

    Returns a sub_location_result dict and saves to analysis/sub_location_result.json.
    Always returns something safe — never crashes the pipeline.
    """
    # STEP 1 — Load final decision
    final_decision = _load_json(analysis_dir / "final_decision.json", {})
    country = str(final_decision.get("final_country") or "")
    confidence = float(final_decision.get("confidence", 0.0) or 0.0)

    if not country or country == "Unknown":
        result = _minimal_result("Unknown", "No confirmed country available")
        _save(result, analysis_dir)
        return result

    # Check if knowledge base exists for this country
    kb_path = _country_to_kb_path(country)
    if not kb_path.exists():
        logger.info(
            "No regional knowledge base for %s. Skipping sub-location reasoning.", country
        )
        result = _minimal_result(
            country,
            f"No regional knowledge base for {country}.",
            sub_location_attempted=False,
            knowledge_base_used=False,
        )
        _save(result, analysis_dir)
        return result

    # Check confidence threshold
    if confidence < COUNTRY_CONFIDENCE_THRESHOLD:
        logger.info(
            "Country confidence too low for sub-location reasoning "
            "(need >%.2f, got %.2f).",
            COUNTRY_CONFIDENCE_THRESHOLD,
            confidence,
        )
        result = _minimal_result(
            country,
            f"Country confidence {confidence:.0%} below threshold "
            f"{COUNTRY_CONFIDENCE_THRESHOLD:.0%}.",
            sub_location_attempted=False,
            knowledge_base_used=True,
        )
        _save(result, analysis_dir)
        return result

    # STEP 2 — Load knowledge base and screenshots
    kb = _load_json(kb_path, {})
    if not kb or not kb.get("regions"):
        logger.warning("Knowledge base for %s is empty or malformed.", country)
        result = _minimal_result(
            country,
            f"Knowledge base for {country} is empty.",
            sub_location_attempted=False,
            knowledge_base_used=False,
        )
        _save(result, analysis_dir)
        return result

    screenshots = _load_selected_screenshots(analysis_dir)
    if not screenshots:
        logger.warning("No selected screenshots available for sub-location reasoning.")
        result = _minimal_result(
            country,
            "No screenshots available.",
            sub_location_attempted=False,
            knowledge_base_used=True,
        )
        _save(result, analysis_dir)
        return result

    logger.info(
        "Sending sub-location request for %s (%d screenshots, %d regions in KB)",
        country,
        len(screenshots),
        len(kb.get("regions", [])),
    )

    # STEP 3 — Build prompt and call Gemini
    screenshot_names = [p.name for p in screenshots]
    prompt = _build_sub_location_prompt(country, kb, screenshot_names)

    try:
        client = genai.Client(api_key=_load_api_key())
        contents: list[Any] = [prompt]
        for i, path in enumerate(screenshots, start=1):
            contents.append(f"Screenshot {i}: {path.name}")
            contents.append(_image_to_part(path))

        response, model_used = generate_content_with_fallback(
            client=client,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.0,
                system_instruction=(
                    "You identify the specific state or region within a country "
                    "whose identity is already confirmed. Use the regional knowledge "
                    "base provided. Return strict JSON only."
                ),
            ),
            primary_model=MODEL_NAME,
            attempts_per_model=MAX_RETRIES,
            base_delay_seconds=RETRY_DELAY_SECONDS,
            logger=logger,
            operation_name="Sub-location reasoning",
        )
        logger.info("Sub-location reasoning succeeded with model=%s", model_used)

        # STEP 4 — Parse, validate, and sanity-check confidence
        parsed = _parse_sub_location_response(response.text or "{}")
        validated = _validate_response(parsed, kb)
        validated = _sanity_check_confidence(validated)

        result: dict[str, Any] = {
            "country": country,
            "broad_region": validated.get("broad_region"),
            "region_confidence": validated.get("region_confidence", 0.0),
            "candidate_states": validated.get("candidate_states", []),
            "final_state": validated.get("final_state"),
            "state_confidence": validated.get("state_confidence", 0.0),
            "state_evidence": validated.get("state_evidence", []),
            "eliminated_states": validated.get("eliminated_states", []),
            "reasoning": validated.get("reasoning", ""),
            "knowledge_base_used": True,
            "sub_location_attempted": True,
            "model_used": model_used,
            "confidence_adjusted": validated.get("confidence_adjusted", False),
            "adjustment_reason": validated.get("adjustment_reason"),
        }

        if result["final_state"]:
            logger.info(
                "Sub-location result: %s / %s (%.0f%%) in %s region (%.0f%%)",
                country,
                result["final_state"],
                result["state_confidence"] * 100,
                result["broad_region"],
                result["region_confidence"] * 100,
            )
        elif result["broad_region"]:
            logger.info(
                "Sub-location result: %s / region=%s (%.0f%%) — state uncertain",
                country,
                result["broad_region"],
                result["region_confidence"] * 100,
            )

    except Exception as error:
        logger.warning("Sub-location reasoning failed: %s", error)
        result = _minimal_result(
            country,
            f"Reasoning failed: {error}",
            sub_location_attempted=True,
            knowledge_base_used=True,
        )

    # STEP 5 — Save
    _save(result, analysis_dir)
    return result


def _save(result: dict[str, Any], analysis_dir: Path) -> None:
    """Save sub_location_result.json."""
    out_path = analysis_dir / "sub_location_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Saved sub_location_result.json")


def get_available_countries(kb_dir: Path = KNOWLEDGE_BASE_DIR) -> list[str]:
    """Return display names of countries that have a regional knowledge base."""
    if not kb_dir.exists():
        return []
    countries: list[str] = []
    for f in sorted(kb_dir.glob("*.json")):
        try:
            kb = json.loads(f.read_text(encoding="utf-8"))
            name = kb.get("country") or f.stem.replace("_", " ").title()
            countries.append(name)
        except Exception:
            countries.append(f.stem.replace("_", " ").title())
    return countries


def main() -> None:
    result = run_sub_location_reasoning()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
