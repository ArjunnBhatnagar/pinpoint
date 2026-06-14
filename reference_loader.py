from dataclasses import dataclass
from pathlib import Path

from PIL import Image


# Visual grounding adds explicit comparison examples to the model's context.
# This is an early retrieval-augmented architecture: references are selected
# from folders and sent directly to Gemini, without embeddings or vector search.
COUNTRY_DATA_DIR = Path("Country_Data")
REFERENCE_DATA_DIR = COUNTRY_DATA_DIR / "metas"
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

MAX_REFERENCE_IMAGES = 8

# Infrastructure cues are geographically informative because countries and
# regions often standardize poles, road paint, bollards, guardrails, signs, and
# construction materials in locally recognizable ways.
CATEGORY_PRIORITIES = [
    "poles",
    "street_markings",
    "bollards",
    "street_sign",
    "signposts",
    "license_plate",
    "language",
    "flora",
]

COUNTRY_PRIORITIES: list[str] = []


@dataclass(frozen=True)
class ReferenceImage:
    path: Path
    category: str
    region: str


def is_valid_image_file(path: Path) -> bool:
    """Return True for supported image files only."""
    if not path.is_file() or path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
        return False

    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def load_reference_images(
    reference_root: Path = REFERENCE_DATA_DIR,
) -> dict[str, dict[str, list[ReferenceImage]]]:
    """Recursively scan Country_Data/metas reference images.

    Expected layout:
    Country_Data/metas/<Country_Name>/<meta_category>/<image>

    The dataset uses capitalized underscore country folders in metas, for
    example South_Korea and United_States_of_America_USA. General_info uses a
    different lowercase hyphen layout and is handled by retrieval_engine.py.
    """
    references: dict[str, dict[str, list[ReferenceImage]]] = {}

    if not reference_root.exists():
        return references

    for image_path in reference_root.rglob("*"):
        if not is_valid_image_file(image_path):
            continue

        try:
            relative_parts = image_path.relative_to(reference_root).parts
        except ValueError:
            continue

        if len(relative_parts) < 3:
            continue

        region = relative_parts[0]
        category = relative_parts[1]

        references.setdefault(category, {}).setdefault(region, []).append(
            ReferenceImage(path=image_path, category=category, region=region)
        )

    for regions in references.values():
        for image_list in regions.values():
            image_list.sort(key=lambda item: str(item.path).lower())

    return references


def get_available_reference_categories(
    references: dict[str, dict[str, list[ReferenceImage]]] | None = None,
) -> list[str]:
    """Return available reference categories with at least one image."""
    if references is None:
        references = load_reference_images()

    return sorted(references.keys())


def get_available_regions(
    references: dict[str, dict[str, list[ReferenceImage]]] | None = None,
) -> list[str]:
    """Return available countries or regions with at least one image."""
    if references is None:
        references = load_reference_images()

    regions = {
        region
        for category_regions in references.values()
        for region in category_regions.keys()
    }

    return sorted(regions)


def get_category_rank(category: str, category_priorities: list[str]) -> int:
    """Rank a category, keeping unknown future categories after known ones."""
    try:
        return category_priorities.index(category)
    except ValueError:
        return len(category_priorities)


def get_region_rank(region: str, country_priorities: list[str]) -> int:
    """Rank a region, keeping unprioritized regions after priority regions."""
    try:
        return country_priorities.index(region)
    except ValueError:
        return len(country_priorities)


def select_reference_subset(
    references: dict[str, dict[str, list[ReferenceImage]]] | None = None,
    max_images: int = MAX_REFERENCE_IMAGES,
    category_priorities: list[str] | None = None,
    country_priorities: list[str] | None = None,
) -> list[ReferenceImage]:
    """Select a small, interpretable set of reference images for Gemini.

    This is intentionally lightweight and retrieval-ready: it uses simple
    priority ordering now, leaving room for future similarity search.
    """
    if references is None:
        references = load_reference_images()

    if category_priorities is None:
        category_priorities = CATEGORY_PRIORITIES

    if country_priorities is None:
        country_priorities = COUNTRY_PRIORITIES

    candidates = [
        reference
        for category_regions in references.values()
        for image_list in category_regions.values()
        for reference in image_list
    ]

    candidates.sort(
        key=lambda reference: (
            get_category_rank(reference.category, category_priorities),
            get_region_rank(reference.region, country_priorities),
            reference.category,
            reference.region,
            str(reference.path).lower(),
        )
    )

    return candidates[:max_images]


def format_reference_summary(references: list[ReferenceImage]) -> str:
    """Create a readable summary for logs and run artifacts."""
    if not references:
        return (
            "No reference images selected.\n"
            "Add images under Country_Data/metas/<Country>/<meta_category>/ to enable "
            "visual grounding."
        )

    lines = ["Selected reference images:"]

    for index, reference in enumerate(references, start=1):
        category_rank = get_category_rank(reference.category, CATEGORY_PRIORITIES)
        region_rank = get_region_rank(reference.region, COUNTRY_PRIORITIES)
        lines.append(
            f"{index}. category={reference.category}, "
            f"region={reference.region}, path={reference.path}, "
            "selection_reason="
            f"category_priority_rank_{category_rank}, "
            f"region_priority_rank_{region_rank}"
        )

    return "\n".join(lines)
