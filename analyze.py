import os
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

from gemini_retry import generate_content_with_fallback
from prompts import ANALYSIS_PROMPT, GEOGUESSR_SYSTEM_PROMPT
from reference_loader import ReferenceImage


MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_MAX_RETRIES = 2
GEMINI_RETRY_DELAY_SECONDS = 1.0
GEMINI_MAX_IMAGE_WIDTH = 960
GEMINI_JPEG_QUALITY = 85


def get_api_key() -> str:
    """Load GEMINI_API_KEY from the .env file."""
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY was not found in your .env file.")

    return api_key


def read_image_part(image_path: Path) -> types.Part:
    """Resize/compress an image before sending it to Gemini.

    The original screenshot stays untouched on disk. This only optimizes the
    upload payload so poles, signs, road markings, and text remain visible while
    duplicate bandwidth and token-like image cost are reduced.
    """
    with Image.open(image_path) as image:
        image = image.convert("RGB")

        if image.width > GEMINI_MAX_IMAGE_WIDTH:
            scale = GEMINI_MAX_IMAGE_WIDTH / image.width
            new_height = int(image.height * scale)
            image = image.resize((GEMINI_MAX_IMAGE_WIDTH, new_height), Image.LANCZOS)

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=GEMINI_JPEG_QUALITY, optimize=True)
        image_bytes = buffer.getvalue()

    return types.Part.from_bytes(
        data=image_bytes,
        mime_type="image/jpeg",
    )


def analyze_image(image_path: Path) -> str:
    """Send a screenshot to Gemini and return the location analysis."""
    return analyze_images([image_path])


def analyze_images(
    image_paths: list[Path],
    ocr_text: str | None = None,
    reference_images: list[ReferenceImage] | None = None,
) -> str:
    """Send observations, OCR, and references to Gemini in one request."""
    client = genai.Client(api_key=get_api_key())

    contents = [ANALYSIS_PROMPT]

    if ocr_text:
        contents.append("OCR text evidence extracted from the screenshots:")
        contents.append(ocr_text)

    for index, image_path in enumerate(image_paths, start=1):
        contents.append(f"Observed environment view {index}: {image_path.stem}")
        contents.append(read_image_part(image_path))

    if reference_images:
        contents.append(
            "Reference images for visual grounding. Compare observed "
            "infrastructure and environmental patterns against these examples."
        )

        for index, reference in enumerate(reference_images, start=1):
            contents.append(
                "Reference "
                f"{index}: category={reference.category}, region={reference.region}, "
                f"file={reference.path.name}"
            )
            contents.append(read_image_part(reference.path))

    response, _model_used = generate_content_with_fallback(
        client=client,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=GEOGUESSR_SYSTEM_PROMPT,
        ),
        primary_model=MODEL_NAME,
        attempts_per_model=GEMINI_MAX_RETRIES,
        base_delay_seconds=GEMINI_RETRY_DELAY_SECONDS,
        operation_name="Gemini environmental analysis",
    )

    return response.text or "Gemini did not return a text response."


def print_analysis(analysis: str) -> None:
    """Print Gemini's analysis with simple terminal formatting."""
    print()
    print("=" * 60)
    print("Gemini Environmental Comparison Analysis")
    print("=" * 60)
    print(analysis)
    print("=" * 60)
    print()


def main() -> None:
    image_path = Path(input("Screenshot path: ").strip())
    analysis = analyze_image(image_path)
    print_analysis(analysis)


if __name__ == "__main__":
    main()
