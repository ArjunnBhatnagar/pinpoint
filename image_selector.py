import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

from gemini_retry import generate_content_with_fallback

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None


# This module performs screenshot selection only. It does not guess countries.
# It filters quality, removes redundant screenshots, detects geographic metas,
# and chooses information-dense, visually diverse images for later reasoning.
SCREENSHOTS_DIR = Path("screenshots")
ANALYSIS_DIR = Path("analysis")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_SELECTED_IMAGES = 3
MAX_GEMINI_RETRIES = 2
GEMINI_RETRY_DELAY_SECONDS = 1.0

BLUR_THRESHOLD = 70.0
MIN_ENTROPY = 3.0
MIN_EDGE_DENSITY = 0.015
MAX_SKY_PERCENTAGE = 0.65
MAX_VEGETATION_PERCENTAGE_WITHOUT_INFRA = 0.75
PHASH_DUPLICATE_DISTANCE = 8

# Configurable scoring weights. These are deliberately transparent so the
# selector remains interpretable and retrieval-ready for future embeddings,
# active exploration, or reinforcement learning work.
SCORE_WEIGHTS = {
    "entropy": 0.15,
    "edge_density": 0.12,
    "gemini_usefulness": 0.28,
    "meta_density": 0.25,
    "ocr_relevance": 0.10,
    "diversity": 0.10,
    "blur_penalty": -0.18,
    "sky_penalty": -0.10,
    "vegetation_penalty": -0.08,
    "duplicate_penalty": -0.35,
    "ui_penalty": -0.12,
}

INFRASTRUCTURE_METAS = {
    "utility_poles",
    "bollards",
    "road_signs",
    "lane_markings",
    "license_plates",
    "road_paint",
    "architecture",
    "infrastructure_style",
}

ENVIRONMENTAL_METAS = {"terrain", "vegetation", "flora"}
SIGNAGE_METAS = {"road_signs", "license_plates", "lane_markings", "road_paint"}
VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

logger = logging.getLogger("image_selector")
logging.basicConfig(level=logging.INFO, format="[ImageSelector] %(message)s")


@dataclass
class LowLevelScores:
    blur_variance: float = 0.0
    entropy: float = 0.0
    edge_density: float = 0.0
    sky_percentage: float = 0.0
    vegetation_percentage: float = 0.0
    ui_blockage_score: float = 0.0
    perceptual_hash: str = ""


@dataclass
class GeminiMetaResult:
    detected_metas: list[str] = field(default_factory=list)
    usefulness_score: float = 0.0
    reasoning: str = ""
    ocr_relevance_estimate: float = 0.0
    dominant_type: str = "unknown"


@dataclass
class ScreenshotAnalysis:
    path: str
    low_level: LowLevelScores
    gemini: GeminiMetaResult = field(default_factory=GeminiMetaResult)
    final_score: float = 0.0
    rejected: bool = False
    rejection_reasons: list[str] = field(default_factory=list)
    duplicate_of: str | None = None
    ranking_explanation: str = ""


def load_screenshots(screenshots_dir: Path = SCREENSHOTS_DIR) -> list[Path]:
    """Load screenshot image paths from the screenshots folder."""
    if not screenshots_dir.exists():
        return []

    return sorted(
        [
            path
            for path in screenshots_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
        ],
        key=lambda path: path.stat().st_mtime,
    )


def ensure_cv2_available() -> None:
    """Fail early with a useful message if OpenCV/NumPy is not installed."""
    if cv2 is None or np is None:
        raise RuntimeError(
            "OpenCV and NumPy are required. Install with: "
            "python -m pip install opencv-python numpy"
        )


def read_cv_image(image_path: Path):
    """Read an image with OpenCV."""
    ensure_cv2_available()
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    return image


def detect_blur(image) -> float:
    """Detect blur using Laplacian variance; higher is sharper."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_entropy(image) -> float:
    """Compute image entropy as a visual complexity estimate."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    probabilities = histogram / max(histogram.sum(), 1)
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def detect_edge_density(image) -> float:
    """Estimate useful structural detail through Canny edge density."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    return float(np.count_nonzero(edges) / edges.size)


def estimate_sky_percentage(image) -> float:
    """Estimate how much of the image is dominated by sky-like pixels."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    blue_sky = (hue >= 85) & (hue <= 130) & (saturation > 35) & (value > 100)
    white_sky = (saturation < 35) & (value > 175)
    upper_half = np.zeros_like(blue_sky, dtype=bool)
    upper_half[: image.shape[0] // 2, :] = True

    sky_mask = (blue_sky | white_sky) & upper_half
    return float(np.count_nonzero(sky_mask) / sky_mask.size)


def estimate_vegetation_percentage(image) -> float:
    """Estimate vegetation dominance using HSV green ranges."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    green_mask = (hue >= 35) & (hue <= 90) & (saturation > 35) & (value > 45)
    return float(np.count_nonzero(green_mask) / green_mask.size)


def estimate_ui_blockage(image) -> float:
    """Estimate UI blockage by measuring high-detail border bands.

    UI overlays, HUDs, browser bars, and ads often occupy edges of the frame.
    This cheap heuristic flags screenshots where border bands contain a lot of
    sharp rectangular detail that may contaminate OCR and visual reasoning.
    """
    height, width = image.shape[:2]
    band_h = max(1, int(height * 0.12))
    band_w = max(1, int(width * 0.12))

    bands = [
        image[:band_h, :],
        image[-band_h:, :],
        image[:, :band_w],
        image[:, -band_w:],
    ]

    densities = [detect_edge_density(band) for band in bands]
    return float(sum(densities) / len(densities))


def compute_phash(image_path: Path) -> str:
    """Compute a simple perceptual hash without adding extra dependencies."""
    with Image.open(image_path) as pil_image:
        gray = pil_image.convert("L").resize((32, 32), Image.LANCZOS)
        pixels = np.array(gray, dtype=np.float32)

    dct = cv2.dct(pixels)
    low_frequency = dct[:8, :8]
    median = np.median(low_frequency[1:, 1:])
    bits = low_frequency > median

    return "".join("1" if bit else "0" for bit in bits.flatten())


def hamming_distance(left_hash: str, right_hash: str) -> int:
    """Return Hamming distance between two equal-length hashes."""
    if len(left_hash) != len(right_hash):
        return max(len(left_hash), len(right_hash))

    return sum(left != right for left, right in zip(left_hash, right_hash))


def run_low_level_analysis(image_paths: list[Path]) -> list[ScreenshotAnalysis]:
    """Run fast local OpenCV/PIL analysis for every screenshot."""
    analyses = []

    for image_path in image_paths:
        image = read_cv_image(image_path)
        low_level = LowLevelScores(
            blur_variance=detect_blur(image),
            entropy=compute_entropy(image),
            edge_density=detect_edge_density(image),
            sky_percentage=estimate_sky_percentage(image),
            vegetation_percentage=estimate_vegetation_percentage(image),
            ui_blockage_score=estimate_ui_blockage(image),
            perceptual_hash=compute_phash(image_path),
        )

        analysis = ScreenshotAnalysis(path=str(image_path), low_level=low_level)
        apply_quality_rejections(analysis)
        analyses.append(analysis)

    detect_duplicates(analyses)
    return analyses


def apply_quality_rejections(analysis: ScreenshotAnalysis) -> None:
    """Reject screenshots with obvious quality problems."""
    scores = analysis.low_level

    if scores.blur_variance < BLUR_THRESHOLD:
        analysis.rejection_reasons.append("blurry screenshot")

    if scores.entropy < MIN_ENTROPY:
        analysis.rejection_reasons.append("low visual complexity")

    if scores.edge_density < MIN_EDGE_DENSITY:
        analysis.rejection_reasons.append("little structural detail")

    if scores.sky_percentage > MAX_SKY_PERCENTAGE:
        analysis.rejection_reasons.append("dominated by sky")

    if scores.ui_blockage_score > 0.20:
        analysis.rejection_reasons.append("possibly blocked by UI")

    analysis.rejected = bool(analysis.rejection_reasons)


def detect_duplicates(analyses: list[ScreenshotAnalysis]) -> None:
    """Mark perceptually redundant screenshots using pHash distance."""
    accepted_hashes: list[tuple[str, ScreenshotAnalysis]] = []

    for analysis in analyses:
        for known_hash, known_analysis in accepted_hashes:
            distance = hamming_distance(analysis.low_level.perceptual_hash, known_hash)

            if distance <= PHASH_DUPLICATE_DISTANCE:
                analysis.rejected = True
                analysis.duplicate_of = known_analysis.path
                analysis.rejection_reasons.append(
                    f"duplicate screenshot, pHash distance {distance}"
                )
                break

        if analysis.duplicate_of is None:
            accepted_hashes.append((analysis.low_level.perceptual_hash, analysis))


def load_api_key() -> str:
    """Load Gemini API key from .env."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY was not found in your .env file.")

    return api_key


def image_to_gemini_part(image_path: Path) -> types.Part:
    """Convert a local image to a Gemini input part."""
    image_bytes = image_path.read_bytes()
    mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"

    return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse Gemini JSON even if it wraps output in markdown fences."""
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1).replace("JSON\n", "", 1)

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in Gemini response.")

    return json.loads(cleaned[start : end + 1])


def normalize_score(value: Any) -> float:
    """Normalize model scores into 0..1."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0

    if number > 1:
        number = number / 100

    return max(0.0, min(1.0, number))


def analyze_with_gemini(image_paths: list[Path]) -> dict[str, GeminiMetaResult]:
    """Use Gemini Flash for meta detection only, never country guessing."""
    if not image_paths:
        return {}

    prompt = """
You are a screenshot quality and geographic-meta detector.

Do NOT guess the country, region, city, or location.

For each screenshot, identify only visible geographic/environmental metas:
- utility_poles
- bollards
- road_signs
- lane_markings
- license_plates
- architecture
- infrastructure_style
- terrain
- vegetation
- flora
- road_paint
- driving_side_clues

Return strict JSON:
{
  "images": [
    {
      "file": "filename.png",
      "detected_metas": ["utility_poles", "road_signs"],
      "usefulness_score": 0.0,
      "ocr_relevance_estimate": 0.0,
      "dominant_type": "infrastructure|signage|environment|mixed|low_value",
      "reasoning": "short reason without country guessing"
    }
  ]
}
""".strip()

    try:
        client = genai.Client(api_key=load_api_key())
    except Exception as error:
        logger.warning("Gemini meta detection disabled: %s", error)
        return {}

    contents: list[Any] = [prompt]

    for image_path in image_paths:
        contents.append(f"Screenshot file: {image_path.name}")
        contents.append(image_to_gemini_part(image_path))

    try:
        response, model_used = generate_content_with_fallback(
            client=client,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.0,
                system_instruction=(
                    "You detect screenshot quality and visible geographic "
                    "metas only. Never guess countries or locations."
                ),
            ),
            primary_model=GEMINI_MODEL,
            attempts_per_model=MAX_GEMINI_RETRIES,
            base_delay_seconds=GEMINI_RETRY_DELAY_SECONDS,
            logger=logger,
            operation_name="Gemini meta detection",
        )
        logger.info("Gemini meta detection succeeded with model=%s", model_used)
        parsed = parse_json_object(response.text or "{}")
        results = {}

        for item in parsed.get("images", []):
            filename = str(item.get("file", ""))
            results[filename] = GeminiMetaResult(
                detected_metas=list(item.get("detected_metas", [])),
                usefulness_score=normalize_score(item.get("usefulness_score")),
                reasoning=str(item.get("reasoning", "")),
                ocr_relevance_estimate=normalize_score(
                    item.get("ocr_relevance_estimate")
                ),
                dominant_type=str(item.get("dominant_type", "unknown")),
            )

        return results
    except Exception as error:
        logger.warning("Gemini meta detection unavailable: %s", error)
        return {}


def attach_gemini_results(
    analyses: list[ScreenshotAnalysis],
    gemini_results: dict[str, GeminiMetaResult],
) -> None:
    """Attach Gemini meta detections to analyses by filename."""
    for analysis in analyses:
        filename = Path(analysis.path).name
        analysis.gemini = gemini_results.get(filename, GeminiMetaResult())
        apply_meta_rejections(analysis)


def apply_meta_rejections(analysis: ScreenshotAnalysis) -> None:
    """Reject vegetation-only images after Gemini checks visible infrastructure."""
    metas = set(analysis.gemini.detected_metas)

    if (
        analysis.low_level.vegetation_percentage
        > MAX_VEGETATION_PERCENTAGE_WITHOUT_INFRA
        and not (metas & INFRASTRUCTURE_METAS)
    ):
        analysis.rejection_reasons.append(
            "dominated by vegetation with no detected infrastructure"
        )

    analysis.rejected = bool(analysis.rejection_reasons)


def scale(value: float, low: float, high: float) -> float:
    """Scale a value into 0..1."""
    if high <= low:
        return 0.0

    return max(0.0, min(1.0, (value - low) / (high - low)))


def compute_final_score(analysis: ScreenshotAnalysis) -> float:
    """Combine local scores and Gemini meta detections into one score."""
    metas = set(analysis.gemini.detected_metas)
    meta_density = min(len(metas) / 8, 1.0)
    blur_penalty = 1.0 - scale(analysis.low_level.blur_variance, 0, 250)
    duplicate_penalty = 1.0 if analysis.duplicate_of else 0.0
    vegetation_penalty = (
        analysis.low_level.vegetation_percentage
        if not (metas & INFRASTRUCTURE_METAS)
        else 0.0
    )

    score = (
        SCORE_WEIGHTS["entropy"] * scale(analysis.low_level.entropy, 2, 7)
        + SCORE_WEIGHTS["edge_density"] * scale(analysis.low_level.edge_density, 0.01, 0.12)
        + SCORE_WEIGHTS["gemini_usefulness"] * analysis.gemini.usefulness_score
        + SCORE_WEIGHTS["meta_density"] * meta_density
        + SCORE_WEIGHTS["ocr_relevance"] * analysis.gemini.ocr_relevance_estimate
        + SCORE_WEIGHTS["blur_penalty"] * blur_penalty
        + SCORE_WEIGHTS["sky_penalty"] * analysis.low_level.sky_percentage
        + SCORE_WEIGHTS["vegetation_penalty"] * vegetation_penalty
        + SCORE_WEIGHTS["duplicate_penalty"] * duplicate_penalty
        + SCORE_WEIGHTS["ui_penalty"] * analysis.low_level.ui_blockage_score
    )

    if analysis.rejected:
        score -= 0.35

    analysis.final_score = round(max(0.0, min(1.0, score)), 4)
    analysis.ranking_explanation = build_ranking_explanation(analysis)
    return analysis.final_score


def build_ranking_explanation(analysis: ScreenshotAnalysis) -> str:
    """Explain why a screenshot ranked where it did."""
    parts = [
        f"score={analysis.final_score:.4f}",
        f"metas={', '.join(analysis.gemini.detected_metas) or 'none'}",
        f"entropy={analysis.low_level.entropy:.2f}",
        f"edges={analysis.low_level.edge_density:.3f}",
        f"blur={analysis.low_level.blur_variance:.1f}",
        f"sky={analysis.low_level.sky_percentage:.2f}",
        f"vegetation={analysis.low_level.vegetation_percentage:.2f}",
    ]

    if analysis.rejection_reasons:
        parts.append(f"rejections={'; '.join(analysis.rejection_reasons)}")

    if analysis.gemini.reasoning:
        parts.append(f"gemini={analysis.gemini.reasoning}")

    return " | ".join(parts)


def classify_analysis_type(analysis: ScreenshotAnalysis) -> str:
    """Classify screenshot role for diversity enforcement."""
    metas = set(analysis.gemini.detected_metas)

    if metas & SIGNAGE_METAS:
        return "signage"

    if metas & INFRASTRUCTURE_METAS:
        return "infrastructure"

    if metas & ENVIRONMENTAL_METAS:
        return "environment"

    return analysis.gemini.dominant_type or "unknown"


def visual_distance(left: ScreenshotAnalysis, right: ScreenshotAnalysis) -> int:
    """Return perceptual hash distance for visual diversity checks."""
    return hamming_distance(left.low_level.perceptual_hash, right.low_level.perceptual_hash)


def enforce_diversity(
    analyses: list[ScreenshotAnalysis],
    max_selected: int = MAX_SELECTED_IMAGES,
) -> list[ScreenshotAnalysis]:
    """Select a diverse final set instead of near-identical high scorers."""
    candidates = sorted(
        [analysis for analysis in analyses if not analysis.rejected],
        key=lambda item: item.final_score,
        reverse=True,
    )

    if not candidates:
        candidates = sorted(analyses, key=lambda item: item.final_score, reverse=True)

    selected: list[ScreenshotAnalysis] = []
    preferred_roles = ["infrastructure", "signage", "environment"]

    for role in preferred_roles:
        for candidate in candidates:
            if candidate in selected:
                continue

            if classify_analysis_type(candidate) != role:
                continue

            if is_diverse_enough(candidate, selected):
                selected.append(candidate)
                break

        if len(selected) >= max_selected:
            return selected[:max_selected]

    for candidate in candidates:
        if candidate in selected:
            continue

        if is_diverse_enough(candidate, selected):
            selected.append(candidate)

        if len(selected) >= max_selected:
            break

    if len(selected) < max_selected:
        for candidate in candidates:
            if candidate not in selected:
                selected.append(candidate)

            if len(selected) >= max_selected:
                break

    apply_diversity_bonus(selected)
    return selected[:max_selected]


def is_diverse_enough(
    candidate: ScreenshotAnalysis,
    selected: list[ScreenshotAnalysis],
) -> bool:
    """Reject near-duplicates during final selection."""
    return all(visual_distance(candidate, existing) > PHASH_DUPLICATE_DISTANCE for existing in selected)


def apply_diversity_bonus(selected: list[ScreenshotAnalysis]) -> None:
    """Add a small diversity bonus to selected screenshots."""
    for analysis in selected:
        analysis.final_score = round(
            min(1.0, analysis.final_score + SCORE_WEIGHTS["diversity"]),
            4,
        )
        analysis.ranking_explanation = build_ranking_explanation(analysis)


def analysis_to_dict(analysis: ScreenshotAnalysis) -> dict[str, Any]:
    """Convert dataclass output to JSON-safe dictionaries."""
    return asdict(analysis)


def save_results(
    analyses: list[ScreenshotAnalysis],
    selected: list[ScreenshotAnalysis],
    analysis_dir: Path = ANALYSIS_DIR,
) -> None:
    """Save full ranking and selected-image summaries."""
    analysis_dir.mkdir(parents=True, exist_ok=True)

    ranking_payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "max_selected_images": MAX_SELECTED_IMAGES,
            "blur_threshold": BLUR_THRESHOLD,
            "min_entropy": MIN_ENTROPY,
            "min_edge_density": MIN_EDGE_DENSITY,
            "max_sky_percentage": MAX_SKY_PERCENTAGE,
            "max_vegetation_percentage_without_infra": (
                MAX_VEGETATION_PERCENTAGE_WITHOUT_INFRA
            ),
            "phash_duplicate_distance": PHASH_DUPLICATE_DISTANCE,
            "score_weights": SCORE_WEIGHTS,
        },
        "all_screenshots": [analysis_to_dict(item) for item in analyses],
    }

    combined_metas = sorted(
        {
            meta
            for analysis in selected
            for meta in analysis.gemini.detected_metas
        }
    )

    selected_payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "combined_visible_metas": combined_metas,
        "selected_screenshots": [analysis_to_dict(item) for item in selected],
    }

    (analysis_dir / "image_ranking.json").write_text(
        json.dumps(ranking_payload, indent=2),
        encoding="utf-8",
    )
    (analysis_dir / "selected_images.json").write_text(
        json.dumps(selected_payload, indent=2),
        encoding="utf-8",
    )


def fallback_all_screenshots(
    screenshots_dir: Path = SCREENSHOTS_DIR,
    analysis_dir: Path = ANALYSIS_DIR,
    error: Exception | None = None,
    image_paths: list[Path] | None = None,
) -> list[Path]:
    """Fail safe: return all screenshots and write fallback diagnostics."""
    if image_paths is None:
        image_paths = load_screenshots(screenshots_dir)

    analysis_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "fallback": True,
        "error": str(error) if error else None,
        "selected_screenshots": [str(path) for path in image_paths],
    }

    (analysis_dir / "selected_images.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (analysis_dir / "image_ranking.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    return image_paths


def run_image_selector(
    image_paths: list[Path] | None = None,
    analysis_dir: Path = ANALYSIS_DIR,
    max_selected: int = MAX_SELECTED_IMAGES,
) -> list[Path]:
    """Rank and select screenshots, always falling back safely on failure."""
    try:
        if image_paths is None:
            image_paths = load_screenshots()

        image_paths = [Path(path) for path in image_paths]

        if not image_paths:
            logger.info("No screenshots provided to selector.")
            return []

        logger.info("Analyzing %s screenshots", len(image_paths))
        analyses = run_low_level_analysis(image_paths)

        gemini_results = analyze_with_gemini(image_paths)
        attach_gemini_results(analyses, gemini_results)

        for analysis in analyses:
            compute_final_score(analysis)

        selected = enforce_diversity(analyses, max_selected)

        if not selected:
            logger.warning("No images passed selection. Falling back to all screenshots.")
            return fallback_all_screenshots(
                analysis_dir=analysis_dir,
                image_paths=image_paths,
            )

        save_results(analyses, selected, analysis_dir)
        logger.info("Selected %s screenshots", len(selected))

        return [Path(analysis.path) for analysis in selected]
    except Exception as error:
        logger.exception("Image selection failed. Falling back to all screenshots.")
        return fallback_all_screenshots(
            analysis_dir=analysis_dir,
            error=error,
            image_paths=image_paths or [],
        )


def select_best_screenshots(
    screenshots_dir: Path = SCREENSHOTS_DIR,
    analysis_dir: Path = ANALYSIS_DIR,
    max_selected: int = MAX_SELECTED_IMAGES,
) -> list[Path]:
    """Select the most useful screenshots, falling back on failure."""
    return run_image_selector(
        image_paths=load_screenshots(screenshots_dir),
        analysis_dir=analysis_dir,
        max_selected=max_selected,
    )


def main() -> None:
    """Run the selector from the terminal."""
    selected = select_best_screenshots()
    print("Selected screenshots:")

    for path in selected:
        print(f"- {path}")


if __name__ == "__main__":
    main()
