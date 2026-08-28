# Pinpoint

An AI-powered GeoGuessr assistant that identifies your country, region, and state from Street View screenshots. Pinpoint uses Gemini 2.5 Flash to run a 13-stage reasoning pipeline - capturing multiple views, running OCR, retrieving country-specific reference images, and cross-checking environmental clues against a meta-knowledge base of visual identifiers (road signs, utility poles, bollards, lane markings, flora, terrain, and more) - to make a grounded, confidence-verified prediction.

## Features

- **Smart capture** - 360° scan + reactive exploration to collect the highest-evidence screenshots before analysis
- **13-category visual meta analysis** - bollards, road paint, utility poles, lane markings, terrain, flora, architecture, license plates, road signs, language/script, street names, signposts, driving side
- **1,840+ reference image database** - 125 countries scraped from [geohints.com](https://geohints.com), used for grounded image-to-image comparison
- **Regional and state-level reasoning** - sub-location knowledge bases for 15 countries (Australia, Russia, USA, Canada, Brazil, Mexico, Argentina, Chile, Japan, South Africa, India, Indonesia, Nigeria, Kenya, Colombia)
- **Accuracy tracking** - logs ground-truth answers after each round; `--stats` flag shows country/region/state accuracy, failure stage breakdown, and most-confused country pairs

## Setup

1. **Clone the repo**
   ```
   git clone https://github.com/yourname/pinpoint.git
   cd pinpoint
   ```

2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Add your Gemini API key**
   ```
   cp .env.example .env
   # Edit .env and replace your_api_key_here with your real key
   ```

4. **Build the reference database**

   The `Country_Data/` folder is gitignored and must be built locally. Run the scraper once to download reference images from geohints.com:
   ```
   python geohints_scraper.py
   ```
   This creates `Country_Data/` with ~1,840 categorised reference images across 125 countries. It takes a few minutes and only needs to be run once (or to refresh the database).

5. **Run Pinpoint**
   ```
   python main.py
   ```
   Press **G** to start a capture and analysis run. Press **ESC** to quit.

## Usage

### Basic run
Press **G** while GeoGuessr is the active window. Pinpoint captures the scene, runs the full 13-stage pipeline, and prints the final country prediction with confidence score and region/state.

After each run you are prompted:
```
Press Enter to skip, or type the correct answer:
```
Type the correct country name to log the result for accuracy tracking.

### Accuracy stats
```
python main.py --stats
```
Shows a summary of all logged rounds including country/region/state accuracy, failure stage breakdown, average confidence when correct vs. wrong, and the most-confused country pairs (useful for identifying which entries to add to `knowledge_base/country_metas.json`).

### Debug mode
```
set PINPOINT_DEBUG=1
python main.py
```
Enables full verbose output including raw Gemini responses, OCR processing details, and per-step pipeline logs.

## Architecture

The pipeline runs these stages in order:

| # | Stage | Description |
|---|-------|-------------|
| 1 | **Smart Capture** | 360° scan + reactive movement to collect high-evidence screenshots |
| 2 | **Image Selection** | Scores and filters screenshots by visual information density |
| 3 | **OCR** | Extracts text (road signs, place names) from selected screenshots |
| 4 | **Environmental Analysis** | Gemini 2.5 Flash analyses the live scene: terrain, flora, infrastructure |
| 5 | **First-Pass Reasoning** | Structured country ranking with injected meta-knowledge for all 125 countries |
| 6 | **Reference Retrieval** | Fetches reference images and text for the top candidate countries |
| 7 | **Grounded Comparison** | Gemini compares live screenshots directly against reference images |
| 8 | **Regional Consistency** | Validates the predicted country against region-cluster logic |
| 9 | **Confidence Verification** | Penalty/floor system converts raw scores to calibrated confidence |
| 10 | **Adaptive Re-Exploration** | Optionally captures more screenshots if confidence is low |
| 11 | **Final Decision** | Selects the authoritative answer from the pipeline chain |
| 12 | **Sub-Location Reasoning** | For supported countries, identifies state/province using regional KB |
| 13 | **Accuracy Tracking** | Logs ground truth and computes failure stage for post-round review |

## Tech Stack

- **Python 3.11+**
- **Gemini 2.5 Flash** (via Google Gen AI SDK) — environmental analysis, grounded comparison, sub-location reasoning
- **OpenCV / Pillow** — image pre-processing and quality scoring
- **EasyOCR** — text extraction from screenshots (optional; disabled by default)
- **keyboard** — global hotkey listener for in-game triggering
- **python-dotenv** — `.env` file loading

## Reference Images

Reference images are scraped at runtime from [geohints.com](https://geohints.com) by `geohints_scraper.py`. They are **not included in this repository** due to copyright and terms-of-service considerations. The scraper is included so you can build the database locally. Full credit for the reference image data goes to [geohints.com](https://geohints.com).

## License

MIT License — see [LICENSE](LICENSE)
