# Gemini Grocery Flyer Annotator

Automatically detects and annotates promotional offers in grocery flyer PDFs using a multimodal LLM (Gemini or GPT-4). Each detected promotion gets a colored bounding box, a label showing the price and product name, and a numbered circle — all drawn directly onto the flyer image.

## How It Works

```
PDF flyer
   │
   ▼
pdf2image → per-page JPEGs
   │
   ▼
Resize to ≤ 1024px → send to Gemini / GPT-4 with detection prompt
   │
   ▼
LLM returns JSON: [{label, bbox, color, outline}, ...]
   │
   ├── Save raw JSON to annotation dir (for review / reuse)
   │
   ▼
Draw annotations on original full-res image
   │
   ▼
Save annotated JPEG to output dir
```

### Detection prompt

The LLM is given a grocery-flyer-expert persona and asked to return **one bounding box per unique price offer**, covering both the price text and associated product image/name. Coordinates are returned as decimal fractions (0.0–1.0) of image width/height, then scaled back to the original image resolution before drawing.

### Annotation style

| Visual element | What it shows |
|---|---|
| Semi-transparent colored fill | Promotion region |
| Thick colored border (14px) | Promotion boundary |
| Colored label panel (top-left) | Price + product name |
| Numbered circle (top-right) | Promotion index |

Up to 12 distinct color presets are cycled; any extra colors from the LLM response are normalized and used directly.

## Prerequisites

**Python 3.11+** and the [uv](https://github.com/astral-sh/uv) package manager (or pip).

**Poppler** is required by `pdf2image` for PDF rendering. Install it for your OS:

- **Windows**: Download from [poppler releases](https://github.com/oschwartz10612/poppler-windows/releases), unzip, and add the `bin/` folder to your `PATH`.
- **macOS**: `brew install poppler`
- **Linux**: `sudo apt install poppler-utils`

## Installation

```bash
# Clone and enter the project
cd ComputerVision/data_annotation

# Install Python dependencies with uv
uv sync

# Or with pip
pip install langchain langchain-google-genai langchain-openai pillow pdf2image python-dotenv
```

## Configuration

Create a `.env` file in the project directory:

```env
# Required for Gemini (default model)
GOOGLE_API_KEY=your_google_api_key_here

# Required if using an OpenAI model
OPENAI_API_KEY=your_openai_api_key_here
```

The default model is `gemini-3.1-pro-preview` (set via `MODEL_NAME` at the top of [gemini_annotation.py](gemini_annotation.py)). To switch to an OpenAI model, change `MODEL_NAME` to e.g. `gpt-4.1-mini`.

## Usage

```bash
python gemini_annotation.py \
  --pdf path/to/flyer.pdf \
  --output annotated_images/ \
  --annotation annotation_json/ \
  [--images_dir cropped_images/] \
  [--start_page 1] \
  [--end_page 3]
```

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--pdf` | Yes | — | Path to the input PDF flyer |
| `--output` | Yes | — | Directory where annotated images are saved |
| `--annotation` | Yes | — | Directory where raw LLM JSON responses are saved |
| `--images_dir` | No | `cropped_images` | Directory for intermediate per-page JPEGs |
| `--start_page` | No | `1` | First page to process (1-indexed) |
| `--end_page` | No | all pages | Last page to process (inclusive) |

### Example

```bash
python gemini_annotation.py \
  --pdf weekly_flyer.pdf \
  --output output/annotated \
  --annotation output/annotation \
  --start_page 1 \
  --end_page 4
```

**Output structure:**

```
output/
├── annotated/
│   ├── weekly_flyer_1.jpg   # annotated page 1
│   ├── weekly_flyer_2.jpg
│   └── ...
├── annotation/
│   ├── weekly_flyer_1.json  # raw LLM JSON for page 1
│   └── ...
cropped_images/
│   ├── weekly_flyer_1.jpg   # intermediate page renders
│   └── ...
```

Already-processed pages are skipped on re-runs (both intermediate images and annotated outputs).

## Project Structure

```
data_annotation/
├── gemini_annotation.py   # main script
├── pyproject.toml         # project metadata and dependencies
├── .env                   # API keys (not committed)
└── README.md
```
