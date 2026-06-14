# ================================================
# GEOHINTS SCRAPER — SETUP INSTRUCTIONS
# ================================================
#
# REQUIRED:
#   pip install requests beautifulsoup4 Pillow numpy
#
# NO API KEY NEEDED
# Downloads geohints.com's own curated CDN images
# directly — no Google Maps API, no Street View API,
# no coordinate resolution required.
#
# USAGE:
#   python geohints_scraper.py
#       runs everything (takes ~15-30 mins)
#   python geohints_scraper.py --priority-only
#       runs top 22 countries only (~5 mins)
#   python geohints_scraper.py --country Japan
#       runs one country only
#   python geohints_scraper.py --page utilityPoles
#       runs one meta page only
#   python geohints_scraper.py --country Japan --page utilityPoles
#       only Japan utility poles
#   python geohints_scraper.py --stats
#       shows current database coverage without downloading
#   python geohints_scraper.py --dry-run
#       shows what would be scraped without downloading
#
# OUTPUT:
#   Images saved to Country_Data/metas/
#   Log saved to geohints_scraper_log.txt
# ================================================

from __future__ import annotations

import argparse
import io
import logging
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

# ── PART A: Dependency check ──────────────────────────────────────────────────

_MISSING_DEPS: list[str] = []
try:
    import requests
except ImportError:
    _MISSING_DEPS.append("requests")

try:
    from bs4 import BeautifulSoup
except ImportError:
    _MISSING_DEPS.append("beautifulsoup4")

try:
    from PIL import Image
    import numpy as np
except ImportError:
    _MISSING_DEPS.append("Pillow")
    _MISSING_DEPS.append("numpy")

if _MISSING_DEPS:
    print("ERROR: Missing required packages:")
    for pkg in _MISSING_DEPS:
        print(f"  {pkg}")
    print("\nInstall with:")
    print(f"  pip install {' '.join(_MISSING_DEPS)}")
    sys.exit(1)

# ── PART B: Configuration ─────────────────────────────────────────────────────

# geohints CDN host — all reference images are served from here
GEOHINTS_CDN_HOST = "ocsc00skc0wokcs8kw8g8k84.geohints.com"

# Output folder — must match existing structure
DATABASE_ROOT = Path("Country_Data/metas")

# How many images per country per meta to collect
MAX_IMAGES_PER_COUNTRY_META = 6

# Delay between HTTP requests (seconds) — be polite to the server
REQUEST_DELAY = 0.8

# Log file
LOG_FILE = Path("geohints_scraper_log.txt")

# Source site base URL
GEOHINTS_BASE = "https://geohints.com"

# Which meta pages to scrape → (geohints page slug, local folder name)
# Local folder names match actual Country_Data/metas/{Country}/{folder}/ structure.
GEOHINTS_PAGES: list[tuple[str, str]] = [
    ("utilityPoles",        "poles"),
    ("bollards",            "bollards"),
    ("lines",               "street_markings"),
    ("signs/roadNumbering", "street_sign"),
    ("signs/signPosts",     "signposts"),
    ("signs/streetNames",   "street_name"),
    ("signs/speed",         "street_sign"),
    ("signs/stop",          "street_sign"),
    ("signs/yield",         "street_sign"),
    ("signs/chevrons",      "street_sign"),
    ("trafficLights",       "traffic_lights"),
    ("sidewalks",           "road_surface"),
    ("architecture",        "architecture"),
    ("nature",              "flora"),
    ("sceneries",           "terrain"),
    ("licencePlates",       "license_plate"),
]

# GeoGuessr frequency priority — processed first
HIGH_PRIORITY_COUNTRIES: list[str] = [
    "Brazil", "India", "Japan", "Indonesia", "Nigeria",
    "South Africa", "Mexico", "Thailand", "Turkey",
    "Poland", "Romania", "Russia", "Colombia", "Peru",
    "Chile", "United States", "Canada", "Australia",
    "Argentina", "Ukraine", "Kenya", "Ghana",
]

# HTTP headers — mimic a real browser so geohints serves the full page
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Referer": "https://geohints.com/",
}

# ── Logging setup ─────────────────────────────────────────────────────────────

def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("geohints_scraper")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(fh)

    return logger


log = _setup_logging()

# ── Country name normalization ────────────────────────────────────────────────

_COUNTRY_FOLDER_OVERRIDES: dict[str, str] = {
    "united states":                    "United_States_of_America_USA",
    "usa":                              "United_States_of_America_USA",
    "united states of america":         "United_States_of_America_USA",
    "united kingdom":                   "United_Kingdom_UK",
    "uk":                               "United_Kingdom_UK",
    "uae":                              "United_Arab_Emirates_UAE",
    "united arab emirates":             "United_Arab_Emirates_UAE",
    "south korea":                      "South_Korea",
    "republic of korea":                "South_Korea",
    "north korea":                      "North_Korea",
    "czech republic":                   "Czech_Republic",
    "czechia":                          "Czech_Republic",
    "north macedonia":                  "North_Macedonia",
    "eswatini":                         "Eswatini",
    "swaziland":                        "Eswatini",
    "trinidad and tobago":              "Trinidad_and_Tobago",
    "hong kong":                        "Hong_Kong",
    "new zealand":                      "New_Zealand",
    "south africa":                     "South_Africa",
    "sri lanka":                        "Sri_Lanka",
    "american samoa":                   "American_Samoa",
    "northern mariana islands":         "Northern_Mariana_Islands",
    "kyrgyzstan":                       "Kyrgyzstan",
    "isle of man":                      "Isle_of_Man",
    "puerto rico":                      "Puerto_Rico",
    "palestine":                        "Palestine",
    "dominican republic":               "Dominican_Republic",
    "costa rica":                       "Costa_Rica",
    "el salvador":                      "El_Salvador",
    "burkina faso":                     "Burkina_Faso",
    "cape verde":                       "Cape_Verde",
    "ivory coast":                      "Ivory_Coast",
    "cote d'ivoire":                    "Ivory_Coast",
    "sierra leone":                     "Sierra_Leone",
    "guinea bissau":                    "Guinea_Bissau",
    "equatorial guinea":                "Equatorial_Guinea",
    "central african republic":         "Central_African_Republic",
    "dr congo":                         "DR_Congo",
    "democratic republic of the congo": "DR_Congo",
    "new caledonia":                    "New_Caledonia",
    "french guiana":                    "French_Guiana",
    "reunion":                          "Reunion",
    "christmas island":                 "Christmas_Island",
    "cocos island":                     "Cocos_Island",
    "faroe islands":                    "Faroe_Islands",
    "guam":                             "Guam",
    "laos":                             "Laos",
    "sao tome and principe":            "Sao_Tome_And_Principe",
    "saint pierre and miquelon":        "Saint_Pierre_And_Miquelon",
    "san marino":                       "San_Marino",
    "pitcairn islands":                 "Pitcairn_Islands",
}


def _ascii_lower(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(text))
    return nfkd.encode("ascii", "ignore").decode("ascii").lower().strip()


def _title_underscore(text: str) -> str:
    return "_".join(word.capitalize() for word in text.split())


def normalize_country_to_folder(country_name: str, database_root: Path = DATABASE_ROOT) -> str | None:
    """Return the local metas folder name for a geohints country name."""
    clean = _ascii_lower(country_name).strip()

    if clean in _COUNTRY_FOLDER_OVERRIDES:
        return _COUNTRY_FOLDER_OVERRIDES[clean]

    title_form = _title_underscore(clean)

    if database_root.exists():
        existing = {d.name: d for d in database_root.iterdir() if d.is_dir()}

        if title_form in existing:
            return title_form

        clean_underscored = clean.replace(" ", "_")
        for folder_name in existing:
            if _ascii_lower(folder_name) == clean_underscored:
                return folder_name

        compact_target = clean.replace(" ", "").replace("_", "")
        for folder_name in existing:
            if _ascii_lower(folder_name).replace("_", "") == compact_target:
                return folder_name

    return title_form


# ── PART C: Scrape geohints.com pages (CDN image approach) ───────────────────

def scrape_geohints_page(
    page_slug: str,
    session: "requests.Session",
) -> dict[str, list[str]]:
    """Fetch a geohints.com/meta/{page_slug} page and extract CDN image URLs.

    Page structure (confirmed by inspection):
      <div class="text-center text-3xl font-bold">Africa</div>
      <div class="flex flex-wrap ...">
        <div class="text-white text-md p-2">
          <span class="font-bold"> Botswana</span>
          <img src="https://ocsc00skc0wokcs8kw8g8k84.geohints.com/storage/utilityPoles/pole_1.jpg"/>
          <a href="https://goo.gl/maps/...">Open in Google Maps</a>
        </div>
        ...
      </div>

    Returns dict: country_name → [cdn_image_url, ...]
    """
    url = f"{GEOHINTS_BASE}/meta/{page_slug}"
    log.info(f"[Scraper] Fetching: {url}")

    try:
        resp = session.get(url, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        log.warning(f"[Scraper] Failed to fetch {url}: {exc}")
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")
    country_images: dict[str, list[str]] = {}

    # Each image card is a div that contains a bold span (country name) and a CDN img.
    # The confirmed class pattern is: class="text-white text-md p-2"
    # We search broadly for any div that directly holds both a span.font-bold and a CDN img.
    for card in soup.find_all("div"):
        # Look for a direct child span with font-bold class
        span = card.find("span", class_=lambda c: c and "font-bold" in c, recursive=False)
        # Look for a direct child img from the CDN
        img_tag = card.find(
            "img",
            src=lambda s: s and GEOHINTS_CDN_HOST in s,
            recursive=False,
        )

        if not span or not img_tag:
            continue

        country = span.get_text(strip=True)
        cdn_url = img_tag.get("src", "").strip()

        if not country or not cdn_url:
            continue

        country_images.setdefault(country, []).append(cdn_url)

    # Deduplicate while preserving order
    for country in country_images:
        seen: set[str] = set()
        deduped: list[str] = []
        for url_item in country_images[country]:
            if url_item not in seen:
                seen.add(url_item)
                deduped.append(url_item)
        country_images[country] = deduped

    total_images = sum(len(v) for v in country_images.values())
    log.info(
        f"[Scraper] Page {page_slug}: "
        f"{len(country_images)} countries, {total_images} CDN images found"
    )
    for country, imgs in sorted(country_images.items()):
        log.debug(f"[Scraper]   {country}: {len(imgs)} images")

    return country_images


# ── PART E: Download CDN images directly ─────────────────────────────────────

def is_no_imagery(pil_image: "Image.Image") -> bool:
    """Return True if the image looks like a placeholder (uniform grey/black)."""
    try:
        arr = np.array(pil_image.convert("RGB"), dtype=np.float32)
        std = float(np.std(arr))
        if std < 10.0:
            return True
        mean = float(np.mean(arr))
        if 108 <= mean <= 148 and std < 25:
            return True
        return False
    except Exception:
        return True


def image_entropy(pil_image: "Image.Image") -> float:
    """Shannon entropy of an image — higher means more visual content."""
    try:
        arr = np.array(pil_image.convert("L"), dtype=np.uint8)
        hist, _ = np.histogram(arr, bins=256, range=(0, 256))
        hist = hist[hist > 0].astype(np.float64)
        hist /= hist.sum()
        return float(-np.sum(hist * np.log2(hist)))
    except Exception:
        return 0.0


def download_cdn_image(
    cdn_url: str,
    session: "requests.Session",
) -> "Image.Image | None":
    """Download a geohints CDN image and return it as a PIL Image.

    Returns None if the download fails or the image is a placeholder.
    """
    try:
        time.sleep(REQUEST_DELAY)
        resp = session.get(cdn_url, headers=_HEADERS, timeout=30)
        if resp.status_code != 200:
            log.debug(f"[CDN] HTTP {resp.status_code} for {cdn_url}")
            return None

        content_type = resp.headers.get("Content-Type", "")
        if "image" not in content_type and not cdn_url.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            log.debug(f"[CDN] Non-image content-type '{content_type}' for {cdn_url}")
            return None

        img = Image.open(io.BytesIO(resp.content))
        img.load()

        if is_no_imagery(img):
            log.debug(f"[CDN] Placeholder image detected at {cdn_url}")
            return None

        return img

    except Exception as exc:
        log.debug(f"[CDN] Error downloading {cdn_url}: {exc}")
        return None


# ── PART F: Save images to correct folder ─────────────────────────────────────

_TXT_TEMPLATE = (
    "Country:  {country}\n"
    "Category: {category}\n"
    "\n"
    "Reference image from geohints.com for {country} {category}.\n"
    "CDN source: {cdn_url}\n"
    "Geohints page: {source_page}\n"
)


def get_existing_count(country_folder: Path, category_folder_name: str) -> int:
    cat_path = country_folder / category_folder_name
    if not cat_path.exists():
        return 0
    return sum(
        1 for f in cat_path.iterdir()
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )


def get_next_index(country_folder: Path, category_folder_name: str) -> int:
    cat_path = country_folder / category_folder_name
    if not cat_path.exists():
        return 1
    existing = [
        f for f in cat_path.iterdir()
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]
    if not existing:
        return 1
    indices: list[int] = []
    for f in existing:
        parts = f.stem.rsplit("_", 1)
        if len(parts) == 2:
            try:
                indices.append(int(parts[1]))
            except ValueError:
                pass
    return max(indices, default=0) + 1


def save_reference_image(
    pil_image: "Image.Image",
    country_display: str,
    folder_name: str,
    category_folder_name: str,
    index: int,
    cdn_url: str,
    source_page: str,
    database_root: Path = DATABASE_ROOT,
) -> "Path | None":
    """Save image + companion .txt. Returns saved path, or None if skipped."""
    country_folder = database_root / folder_name
    cat_folder = country_folder / category_folder_name

    if not cat_folder.exists():
        cat_folder.mkdir(parents=True, exist_ok=True)
        log.info(f"[Saver] Created folder: {cat_folder}")

    filename = f"{folder_name}_{category_folder_name}_{index}.jpg"
    img_path = cat_folder / filename

    if img_path.exists():
        log.debug(f"[Saver] Already exists: {filename} — skipping")
        return None

    try:
        pil_image.convert("RGB").save(img_path, format="JPEG", quality=90, optimize=True)
        log.info(f"[Saver] Saved {folder_name}/{category_folder_name}/{filename}")
    except Exception as exc:
        log.warning(f"[Saver] Save failed for {img_path}: {exc}")
        return None

    txt_path = cat_folder / f"{folder_name}_{category_folder_name}_{index}.txt"
    try:
        txt_path.write_text(
            _TXT_TEMPLATE.format(
                country=country_display,
                category=category_folder_name,
                cdn_url=cdn_url,
                source_page=source_page,
            ),
            encoding="utf-8",
        )
    except Exception as exc:
        log.debug(f"[Saver] txt write failed: {exc}")

    return img_path


# ── PART G: Main orchestration ────────────────────────────────────────────────

def _prioritize_countries(
    country_images: dict[str, list[str]],
    priority_only: bool = False,
) -> list[str]:
    priority_lower = {c.lower(): i for i, c in enumerate(HIGH_PRIORITY_COUNTRIES)}
    high: list[str] = []
    rest: list[str] = []

    for country in country_images:
        key = _ascii_lower(country)
        if key in priority_lower:
            high.append(country)
        else:
            rest.append(country)

    high.sort(key=lambda c: priority_lower.get(_ascii_lower(c), 999))

    if priority_only:
        return high
    return high + sorted(rest)


def run_scraper(
    pages: "list[tuple[str, str]] | None" = None,
    countries_filter: "list[str] | None" = None,
    max_per_meta: int = MAX_IMAGES_PER_COUNTRY_META,
    dry_run: bool = False,
    priority_only: bool = False,
    database_root: Path = DATABASE_ROOT,
) -> dict[str, int]:
    """Scrape geohints CDN images and save to the reference database."""
    if pages is None:
        pages = GEOHINTS_PAGES

    counters = {
        "pages_scraped": 0,
        "countries_processed": 0,
        "images_downloaded": 0,
        "images_already_existed": 0,
        "images_skipped_placeholder": 0,
        "images_failed": 0,
    }

    filter_lower = {_ascii_lower(c) for c in countries_filter} if countries_filter else None
    session = requests.Session()
    start_time = time.monotonic()

    for page_slug, category_folder_name in pages:
        log.info(f"\n[Scraper] -- {page_slug} -> {category_folder_name} --")

        country_images = scrape_geohints_page(page_slug, session)
        if not country_images:
            log.info(f"[Scraper] No images found for {page_slug} — skipping")
            continue

        counters["pages_scraped"] += 1
        log.info(f"[Scraper] Countries found on page: {len(country_images)}")

        ordered = _prioritize_countries(country_images, priority_only)

        for country_name in ordered:
            if filter_lower and _ascii_lower(country_name) not in filter_lower:
                continue

            cdn_urls = country_images[country_name]
            if not cdn_urls:
                continue

            folder_name = normalize_country_to_folder(country_name, database_root)
            if not folder_name:
                log.debug(f"[Scraper] Cannot map '{country_name}' to a folder — skipping")
                continue

            country_folder = database_root / folder_name
            existing = get_existing_count(country_folder, category_folder_name)
            needed = max_per_meta - existing

            if needed <= 0:
                counters["images_already_existed"] += existing
                log.debug(f"[Scraper] {folder_name}/{category_folder_name}: at max ({existing})")
                continue

            log.info(
                f"[Scraper] Processing {folder_name}/{category_folder_name} "
                f"(have {existing}, need {needed}, available {len(cdn_urls)})"
            )

            if dry_run:
                log.info(
                    f"  [DRY RUN] Would download up to {min(needed, len(cdn_urls))} "
                    f"of {len(cdn_urls)} images for {country_name}"
                )
                counters["countries_processed"] += 1
                continue

            counters["countries_processed"] += 1
            saved = 0
            next_idx = get_next_index(country_folder, category_folder_name)
            source_page = f"{GEOHINTS_BASE}/meta/{page_slug}"

            for cdn_url in cdn_urls:
                if saved >= needed:
                    break

                log.info(f"  [{saved + 1}/{needed}] {cdn_url.split('/')[-1]}")

                img = download_cdn_image(cdn_url, session)
                if img is None:
                    counters["images_skipped_placeholder"] += 1
                    continue

                saved_path = save_reference_image(
                    img,
                    country_display=country_name,
                    folder_name=folder_name,
                    category_folder_name=category_folder_name,
                    index=next_idx,
                    cdn_url=cdn_url,
                    source_page=source_page,
                    database_root=database_root,
                )

                if saved_path:
                    saved += 1
                    next_idx += 1
                    counters["images_downloaded"] += 1
                else:
                    counters["images_already_existed"] += 1

            if saved > 0:
                log.info(
                    f"[Scraper] {folder_name}/{category_folder_name}: "
                    f"{saved}/{needed} images saved"
                )

    elapsed = time.monotonic() - start_time
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)

    label = "DRY RUN COMPLETE" if dry_run else "COMPLETE"
    log.info(f"\n{'='*42}")
    log.info(f"GEOHINTS SCRAPER {label}")
    log.info(f"{'='*42}")
    log.info(f"Pages scraped:              {counters['pages_scraped']}")
    log.info(f"Countries processed:        {counters['countries_processed']}")
    log.info(f"Images downloaded:          {counters['images_downloaded']}")
    log.info(f"Images already existed:     {counters['images_already_existed']}")
    log.info(f"Images skipped (bad):       {counters['images_skipped_placeholder']}")
    log.info(f"Time taken:                 {mins}m {secs:02d}s")
    log.info(f"{'='*42}")

    return counters


# ── Stats display ─────────────────────────────────────────────────────────────

def show_stats(database_root: Path = DATABASE_ROOT) -> None:
    if not database_root.exists():
        print(f"[Stats] Database root not found: {database_root}")
        return

    META_CATS = [
        "poles", "bollards", "street_markings", "street_sign", "signposts",
        "street_name", "flora", "traffic_lights", "road_surface",
        "architecture", "terrain", "license_plate", "language",
    ]

    countries = sorted(d for d in database_root.iterdir() if d.is_dir())
    print(f"\n[Stats] Database: {database_root}  ({len(countries)} country folders)")
    print()

    category_totals: dict[str, int] = {}
    country_rows: list[tuple[str, dict[str, int]]] = []

    for country_dir in countries:
        row: dict[str, int] = {}
        for cat in META_CATS:
            cat_dir = country_dir / cat
            if cat_dir.exists():
                n = sum(
                    1 for f in cat_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                )
                row[cat] = n
                category_totals[cat] = category_totals.get(cat, 0) + n
            else:
                row[cat] = 0
        country_rows.append((country_dir.name, row))

    header_cats = ["poles", "bollards", "street_markings", "street_sign", "flora", "terrain", "architecture"]
    print(f"{'Country':<35}" + "".join(f"{c[:10]:<11}" for c in header_cats))
    print("-" * (35 + 11 * len(header_cats)))

    for name, row in country_rows:
        if not any(row.values()):
            continue
        line = f"{name:<35}" + "".join(
            f"{str(row[c]) if row[c] else '-':<11}" for c in header_cats
        )
        print(line)

    print()
    print("[Stats] Category totals:")
    for cat, total in sorted(category_totals.items(), key=lambda x: -x[1]):
        print(f"  {cat:<25} {total:>4} images")
    print(f"\n[Stats] TOTAL: {sum(category_totals.values())} images")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scrape geohints.com CDN images into Country_Data/metas/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python geohints_scraper.py                        # all pages, all countries
  python geohints_scraper.py --priority-only        # top 22 GeoGuessr countries
  python geohints_scraper.py --country Brazil       # one country, all pages
  python geohints_scraper.py --page utilityPoles    # one page, all countries
  python geohints_scraper.py --country Japan --page utilityPoles
  python geohints_scraper.py --stats                # coverage report
  python geohints_scraper.py --dry-run              # preview without downloading
        """,
    )
    parser.add_argument("--page", metavar="SLUG",
                        help="Only this page slug (e.g. utilityPoles)")
    parser.add_argument("--country", metavar="NAME",
                        help="Only this country (e.g. Japan)")
    parser.add_argument("--stats", action="store_true",
                        help="Show database coverage and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be scraped, no downloading")
    parser.add_argument("--priority-only", action="store_true",
                        help="Only HIGH_PRIORITY_COUNTRIES")
    parser.add_argument("--max", type=int, default=MAX_IMAGES_PER_COUNTRY_META,
                        metavar="N", help=f"Max images per country/meta (default: {MAX_IMAGES_PER_COUNTRY_META})")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.stats:
        show_stats()
        return

    if args.page:
        matched = [(slug, cat) for slug, cat in GEOHINTS_PAGES if slug == args.page]
        if not matched:
            print(f"ERROR: Unknown page slug '{args.page}'")
            print("Valid slugs:", ", ".join(s for s, _ in GEOHINTS_PAGES))
            sys.exit(1)
        pages = matched
    else:
        pages = GEOHINTS_PAGES

    countries_filter = [args.country] if args.country else None

    if args.dry_run:
        log.info("[Scraper] DRY RUN — no images will be downloaded")

    run_scraper(
        pages=pages,
        countries_filter=countries_filter,
        max_per_meta=args.max,
        dry_run=args.dry_run,
        priority_only=args.priority_only,
    )


if __name__ == "__main__":
    main()
