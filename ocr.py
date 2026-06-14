import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

try:
    from terminal_output import debug_print as _ocr_debug_print
except ImportError:
    _ocr_debug_print = print


# GPU acceleration dramatically improves OCR speed because EasyOCR runs neural
# detection/recognition models. On CPU, OCR can become the bottleneck in a
# multimodal pipeline: screenshots are captured quickly, but text extraction can
# hold up the Gemini request for minutes.
#
# CUDA setup note: to use an RTX 3050, install a CUDA-enabled PyTorch build that
# matches your NVIDIA driver. If PyTorch is CPU-only, torch.cuda.is_available()
# will be false and this app will fall back to CPU automatically.
OCR_GPU_ENABLED = True
OCR_MAX_IMAGE_WIDTH = 1280
OCR_CONFIDENCE_THRESHOLD = 0.45

# Region Of Interest filtering keeps OCR focused on the actual gameplay
# viewport. Without this preprocessing, multimodal systems can over-weight
# browser tabs, ads, minimaps, menus, and HUD text that are not part of the
# real-world environment.
TOP_CROP_PERCENT = 0.12
BOTTOM_CROP_PERCENT = 0.16
LEFT_CROP_PERCENT = 0.08
RIGHT_CROP_PERCENT = 0.18

OCR_LANGUAGES = ["en"]

UI_TEXT_BLOCKLIST = [
    "singleplayer",
    "guess",
    "adobe",
    "creative cloud",
    "world",
    "return",
    "geoguessr",
    "settings",
    "menu",
    "play",
    "score",
    "round",
    "map",
    "minimap",
    "challenge",
    "shop",
    "sign in",
    "subscribe",
    "cookie",
    "google",
    "whatsapp",
    "youtube",
    "chrome",
    "search",
    "street view",
]

ROAD_CODE_PATTERN = re.compile(
    r"^(?:route\s*)?[a-z]{1,3}\s*-?\s*\d{1,4}$|^route\s+\d{1,4}$",
    re.IGNORECASE,
)

ENVIRONMENT_KEYWORDS = [
    "road",
    "street",
    "st",
    "avenue",
    "ave",
    "route",
    "highway",
    "hwy",
    "km",
    "city",
    "town",
    "village",
    "school",
    "hotel",
    "market",
    "pharmacy",
    "hospital",
    "airport",
    "station",
    "center",
    "centre",
    "north",
    "south",
    "east",
    "west",
]

_reader = None
_reader_uses_gpu = False
_reader_device_name = "CPU"


@dataclass
class OCRDetection:
    image_name: str
    text: str
    confidence: float
    relevance: str = "unknown"
    removal_reason: str | None = None


@dataclass
class OCRImageResult:
    image_name: str
    detections: list[OCRDetection]
    removed_detections: list[OCRDetection]
    error: str | None = None
    processing_time: float = 0.0
    original_size: tuple[int, int] | None = None
    cropped_size: tuple[int, int] | None = None
    ocr_size: tuple[int, int] | None = None
    cropped_image_path: Path | None = None
    skipped_duplicate: bool = False
    duplicate_of: str | None = None


def detect_cuda_device() -> tuple[bool, str]:
    """Return whether CUDA OCR should be used and the detected device name."""
    if not OCR_GPU_ENABLED:
        return False, "CPU"

    try:
        import torch
    except Exception:
        return False, "CPU"

    if not torch.cuda.is_available():
        return False, "CPU"

    try:
        return True, torch.cuda.get_device_name(0)
    except Exception:
        return True, "CUDA GPU"


def get_reader():
    """Create the EasyOCR reader once and reuse it across screenshots."""
    global _reader, _reader_device_name, _reader_uses_gpu

    if _reader is None:
        import easyocr

        _reader_uses_gpu, _reader_device_name = detect_cuda_device()

        if _reader_uses_gpu:
            _ocr_debug_print(f"[OCR] Using GPU acceleration ({_reader_device_name})")
        else:
            _ocr_debug_print("[OCR] Using CPU OCR (CUDA GPU not available)")

        _reader = easyocr.Reader(OCR_LANGUAGES, gpu=_reader_uses_gpu)

    return _reader


def get_image_fingerprint(image_path: Path) -> str:
    """Create a small visual fingerprint so duplicate screenshots can be skipped."""
    with Image.open(image_path) as image:
        small_image = image.convert("L").resize((32, 18))
        return hashlib.sha256(small_image.tobytes()).hexdigest()


def crop_for_environmental_ocr(image: Image.Image) -> Image.Image:
    """Crop a screenshot to the gameplay area most likely to contain signs.

    This removes likely browser UI, minimaps, bottom HUD text, side panels,
    buttons, and ads before OCR. It is a simple ROI filter, not object detection.
    """
    width, height = image.size

    left = int(width * LEFT_CROP_PERCENT)
    top = int(height * TOP_CROP_PERCENT)
    right = int(width * (1 - RIGHT_CROP_PERCENT))
    bottom = int(height * (1 - BOTTOM_CROP_PERCENT))

    if right <= left or bottom <= top:
        return image

    return image.crop((left, top, right, bottom))


def save_cropped_debug_image(
    cropped_image: Image.Image,
    image_path: Path,
    debug_dir: Path | None,
) -> Path | None:
    """Save the OCR crop so ROI tuning can be inspected later."""
    if debug_dir is None:
        return None

    crop_dir = debug_dir / "ocr_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    cropped_path = crop_dir / f"{image_path.stem}_ocr_crop.png"
    cropped_image.save(cropped_path)

    return cropped_path


def prepare_image_for_ocr(image_path: Path, debug_dir: Path | None = None):
    """Crop and resize screenshots before OCR to reduce UI contamination."""
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    cropped_image = crop_for_environmental_ocr(image)
    cropped_size = cropped_image.size
    cropped_image_path = save_cropped_debug_image(cropped_image, image_path, debug_dir)

    if cropped_image.width > OCR_MAX_IMAGE_WIDTH:
        scale = OCR_MAX_IMAGE_WIDTH / cropped_image.width
        new_height = int(cropped_image.height * scale)
        cropped_image = cropped_image.resize(
            (OCR_MAX_IMAGE_WIDTH, new_height),
            Image.LANCZOS,
        )

    return cropped_image, original_size, cropped_size, cropped_image.size, cropped_image_path


def normalize_ocr_text(text: str) -> str:
    """Normalize OCR text for simple relevance filtering."""
    return re.sub(r"\s+", " ", text.strip()).lower()


def get_filter_reason(text: str) -> str | None:
    """Return why OCR text should be removed, or None if it should be kept."""
    normalized = normalize_ocr_text(text)

    if normalized.isdigit() and not ROAD_CODE_PATTERN.fullmatch(normalized):
        _ocr_debug_print(f"[OCR] Rejected isolated digit: {text.strip()}")
        return "isolated digit without road/route context"

    if len(normalized) <= 2 and not ROAD_CODE_PATTERN.fullmatch(normalized):
        return "too short to be useful"

    if any(term in normalized for term in UI_TEXT_BLOCKLIST):
        return "common UI/ad/HUD text"

    if re.fullmatch(r"[\W_]+", normalized):
        return "symbols only"

    if len(normalized) <= 4 and normalized.isalpha():
        return "short low-context word"

    return None


def get_relevance_label(text: str) -> str:
    """Label kept OCR text by how likely it is to describe the environment."""
    normalized = normalize_ocr_text(text)

    if any(term in normalized for term in UI_TEXT_BLOCKLIST):
        return "low"

    if ROAD_CODE_PATTERN.fullmatch(normalized):
        return "high"

    if any(keyword in normalized for keyword in ENVIRONMENT_KEYWORDS):
        return "high"

    if re.search(r"\d", normalized) and len(normalized) >= 3:
        return "medium"

    if len(normalized) >= 5:
        return "medium"

    return "low"


def extract_text_from_image(
    image_path: Path,
    confidence_threshold: float = OCR_CONFIDENCE_THRESHOLD,
    reader=None,
    debug_dir: Path | None = None,
) -> OCRImageResult:
    """Extract high-confidence OCR text from one screenshot."""
    start_time = time.monotonic()
    original_size = None
    cropped_size = None
    ocr_size = None
    cropped_image_path = None

    try:
        if reader is None:
            reader = get_reader()

        image, original_size, cropped_size, ocr_size, cropped_image_path = (
            prepare_image_for_ocr(image_path, debug_dir)
        )

        _ocr_debug_print(
            f"[OCR] Processing {image_path.name}: "
            f"{original_size[0]}x{original_size[1]} -> "
            f"crop {cropped_size[0]}x{cropped_size[1]} -> "
            f"ocr {ocr_size[0]}x{ocr_size[1]}"
        )

        import numpy as np

        raw_results = reader.readtext(np.array(image))
    except Exception as error:
        elapsed = time.monotonic() - start_time
        return OCRImageResult(
            image_name=image_path.name,
            detections=[],
            removed_detections=[],
            error=str(error),
            processing_time=elapsed,
            original_size=original_size,
            cropped_size=cropped_size,
            ocr_size=ocr_size,
            cropped_image_path=cropped_image_path,
        )

    detections = []
    removed_detections = []
    for _box, text, confidence in raw_results:
        clean_text = text.strip()

        if not clean_text:
            continue

        if confidence < confidence_threshold:
            removed_detections.append(
                OCRDetection(
                    image_name=image_path.name,
                    text=clean_text,
                    confidence=confidence,
                    removal_reason="low confidence",
                )
            )
            continue

        filter_reason = get_filter_reason(clean_text)
        if filter_reason:
            removed_detections.append(
                OCRDetection(
                    image_name=image_path.name,
                    text=clean_text,
                    confidence=confidence,
                    removal_reason=filter_reason,
                )
            )
            continue

        relevance = get_relevance_label(clean_text)

        detections.append(
            OCRDetection(
                image_name=image_path.name,
                text=clean_text,
                confidence=confidence,
                relevance=relevance,
            )
        )

    elapsed = time.monotonic() - start_time
    _ocr_debug_print(
        f"[OCR] Completed {image_path.name} in {elapsed:.2f}s "
        f"({len(detections)} kept)"
    )

    return OCRImageResult(
        image_name=image_path.name,
        detections=detections,
        removed_detections=removed_detections,
        processing_time=elapsed,
        original_size=original_size,
        cropped_size=cropped_size,
        ocr_size=ocr_size,
        cropped_image_path=cropped_image_path,
    )


def extract_text_from_images(
    image_paths: list[Path],
    confidence_threshold: float = OCR_CONFIDENCE_THRESHOLD,
    debug_dir: Path | None = None,
) -> list[OCRImageResult]:
    """Extract OCR text from all screenshots in a run."""
    start_time = time.monotonic()

    try:
        reader = get_reader()
    except Exception as error:
        message = (
            "EasyOCR could not initialize. This often happens when the first-use "
            f"model download is blocked or incomplete. Details: {error}"
        )

        return [
            OCRImageResult(
                image_name=image_path.name,
                detections=[],
                removed_detections=[],
                error=message,
            )
            for image_path in image_paths
        ]

    results = []
    seen_fingerprints = {}

    for image_path in image_paths:
        try:
            fingerprint = get_image_fingerprint(image_path)
        except Exception:
            fingerprint = ""

        if fingerprint and fingerprint in seen_fingerprints:
            duplicate_of = seen_fingerprints[fingerprint]
            _ocr_debug_print(f"[OCR] Skipping duplicate screenshot: {image_path.name}")
            results.append(
                OCRImageResult(
                    image_name=image_path.name,
                    detections=[],
                    removed_detections=[],
                    skipped_duplicate=True,
                    duplicate_of=duplicate_of,
                )
            )
            continue

        if fingerprint:
            seen_fingerprints[fingerprint] = image_path.name

        results.append(
            extract_text_from_image(
                image_path,
                confidence_threshold,
                reader,
                debug_dir,
            )
        )

    elapsed = time.monotonic() - start_time
    _ocr_debug_print(f"[OCR] Processing completed in {elapsed:.2f}s")

    return results


def format_ocr_results(results: list[OCRImageResult]) -> str:
    """Format OCR output for terminal display, saving, and Gemini context."""
    kept_count = sum(len(result.detections) for result in results)
    removed_count = sum(len(result.removed_detections) for result in results)
    duplicate_count = sum(1 for result in results if result.skipped_duplicate)

    lines = [
        "OCR Results",
        f"GPU enabled setting: {OCR_GPU_ENABLED}",
        f"OCR device used: {_reader_device_name}",
        f"Max OCR image width: {OCR_MAX_IMAGE_WIDTH}px",
        f"Confidence threshold: {OCR_CONFIDENCE_THRESHOLD:.2f}",
        "ROI crop settings: "
        f"top={TOP_CROP_PERCENT:.2f}, bottom={BOTTOM_CROP_PERCENT:.2f}, "
        f"left={LEFT_CROP_PERCENT:.2f}, right={RIGHT_CROP_PERCENT:.2f}",
        "OCR relevance statistics: "
        f"kept={kept_count}, removed={removed_count}, "
        f"duplicates_skipped={duplicate_count}",
        "",
    ]

    for result in results:
        lines.append(f"[{result.image_name}]")

        if result.original_size and result.cropped_size and result.ocr_size:
            lines.append(
                "  Image size: "
                f"{result.original_size[0]}x{result.original_size[1]} -> "
                f"crop {result.cropped_size[0]}x{result.cropped_size[1]} -> "
                f"{result.ocr_size[0]}x{result.ocr_size[1]}"
            )

        if result.cropped_image_path:
            lines.append(f"  OCR crop: {result.cropped_image_path}")

        if result.processing_time:
            lines.append(f"  OCR time: {result.processing_time:.2f}s")

        if result.skipped_duplicate:
            lines.append(f"  Skipped duplicate of {result.duplicate_of}.")
        elif result.error:
            lines.append(f"  OCR error: {result.error}")
        elif not result.detections:
            lines.append("  Filtered OCR text: none")
        else:
            lines.append("  Filtered OCR text:")
            for detection in result.detections:
                lines.append(
                    f"    - {detection.text} "
                    f"(confidence {detection.confidence:.2f}, "
                    f"relevance {detection.relevance})"
                )

        if result.removed_detections:
            lines.append("  Removed OCR text:")
            for detection in result.removed_detections:
                lines.append(
                    f"    - {detection.text} "
                    f"(confidence {detection.confidence:.2f}, "
                    f"reason: {detection.removal_reason})"
                )

        lines.append("")

    return "\n".join(lines).strip()


def format_ocr_for_gemini(results: list[OCRImageResult]) -> str:
    """Format only filtered environmental OCR text for Gemini context."""
    lines = [
        "Filtered environmental OCR text only.",
        "Removed UI/ad/HUD detections are intentionally excluded.",
        "",
    ]

    has_text = False

    for result in results:
        if not result.detections:
            continue

        has_text = True
        lines.append(f"[{result.image_name}]")

        for detection in result.detections:
            lines.append(
                f"- {detection.text} "
                f"(confidence {detection.confidence:.2f}, "
                f"relevance {detection.relevance})"
            )

        lines.append("")

    if not has_text:
        lines.append("No high-confidence environmentally relevant OCR text found.")

    return "\n".join(lines).strip()


def print_ocr_results(formatted_ocr: str) -> None:
    """Print OCR results with simple terminal formatting."""
    print()
    print("=" * 60)
    print("OCR Text Evidence")
    print("=" * 60)
    print(formatted_ocr)
    print("=" * 60)
    print()
