# Synthetic Speech Dataset Generator

This project contains scripts and tools to generate a synthetic speech dataset using Wikipedia articles. The pipeline extracts AI-related articles, converts them to speech using Kokoro TTS, and enhances the audio with background noise for more realistic training data.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Components](#components)
  - [1. Dockerfile](#1-dockerfile)
  - [2. api.py](#2-apipy)
  - [3. data_preparation.ipynb](#3-data_preparationipynb)
  - [4. data_filteration.ipynb](#4-data_filterationipynb)
  - [5. create_metadata.py](#5-create_metadatapy)
  - [6. merge_audio_with_noise.py](#6-merge_audio_with_noisepy)
  - [7. Package Management (uv)](#7-package-management-uv)
- [Usage Workflow](#usage-workflow)
- [Configuration](#configuration)

## Overview

This project implements a complete pipeline for creating synthetic speech datasets:

1. **Data Extraction**: Extract Wikipedia articles related to artificial intelligence
2. **Article Filtering**: Use Google Gemini to identify the most relevant AI articles
3. **Speech Synthesis**: Convert text to speech using Kokoro TTS
4. **Audio Enhancement**: Add background noise to create more realistic training data
5. **Metadata Generation**: Create metadata files with audio durations and information

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker (for running Kokoro TTS API)
- Hugging Face account (for dataset access)
- Google Gemini API key (for article filtering)

## Project Structure

```
kokoro-tts/
├── Dockerfile                 # Docker configuration for Kokoro TTS API
├── api.py                     # FastAPI server with Gradio UI
├── data_preparation.ipynb     # Notebook for extracting AI articles
├── data_filteration.ipynb     # Notebook for filtering 100 articles
├── create_metadata.py         # Script to generate metadata CSV
├── merge_audio_with_noise.py  # Script to add background noise
├── pyproject.toml            # Project dependencies
├── uv.lock                    # Locked dependency versions
├── noise1/                    # Background noise audio files
│   ├── noise1.wav
│   ├── noise2.wav
│   └── ... (noise1-10.wav)
└── temp/                      # Output directory for processed audio
```

## Components

### 1. Dockerfile

**Purpose**: Docker configuration for hosting Kokoro TTS API server.

**Description**: 
- Sets up a Python 3.10 slim container
- Installs system dependencies (espeak-ng, libsndfile1)
- Installs CPU-only PyTorch and Kokoro TTS
- Configures FastAPI server on port 8080

**Usage**:
```bash
docker build -t kokoro-tts .
docker run -p 8080:8080 kokoro-tts
```

**Key Features**:
- Lightweight CPU-only setup
- Pre-configured with all required dependencies
- Exposes API on port 8080

---

### 2. api.py

**Purpose**: FastAPI server with Gradio UI for Kokoro TTS text-to-speech generation.

**Description**:
- Provides REST API endpoint (`/v1/audio/speech`) for TTS generation
- Includes interactive Gradio web UI at root path (`/`)
- Supports multiple voices and adjustable speech speed
- Returns audio in WAV format

**API Endpoints**:
- `POST /v1/audio/speech`: Generate speech from text
  - Request body: `{"text": "string", "voice": "string", "speed": float}`
  - Response: WAV audio file
- `GET /health`: Health check endpoint

**Available Voices**:
- Female voices: `af_heart`, `af_alloy`, `af_aoede`, `af_bella`, `af_jessica`, `af_kore`, `af_nicole`, `af_nova`, `af_river`, `af_sarah`, `af_sky`
- Male voices: `am_adam`, `am_echo`, `am_eric`, `am_fenrir`, `am_liam`, `am_michael`, `am_onyx`, `am_puck`, `am_santa`
- British voices: `bf_alice`, `bf_emma`, `bf_isabella`, `bf_lily`, `bm_daniel`, `bm_fable`, `bm_george`, `bm_lewis`

**Usage**:
```bash
# Start the API server (via Docker)
docker run -p 8080:8080 kokoro-tts

# Or run directly
uvicorn api:app --host 0.0.0.0 --port 8080
```

**Access**:
- Web UI: `http://localhost:8080/`
- API: `http://localhost:8080/v1/audio/speech`

---

### 3. data_preparation.ipynb

**Purpose**: Jupyter notebook for extracting Wikipedia articles related to machine learning and AI.

**Description**:
- Loads English Wikipedia dataset from Hugging Face
- Filters articles containing "machine learning" at least 3 times
- Uploads filtered dataset to Hugging Face Hub

**Workflow**:
1. Load Wikipedia dataset (`wikimedia/wikipedia`, version `20231101.en`)
2. Filter articles with `has_machine_learning_3times()` function
3. Authenticate with Hugging Face Hub
4. Push filtered dataset to Hub (e.g., `username/wiki-ai-filtered`)

**Requirements**:
- Hugging Face account and authentication
- `datasets` library
- Sufficient disk space for Wikipedia dataset

**Output**: 
- Filtered dataset uploaded to Hugging Face Hub containing articles with machine learning content

---

### 4. data_filteration.ipynb

**Purpose**: Jupyter notebook for identifying 100 most relevant AI articles using Google Gemini model.

**Description**:
- Loads the filtered dataset from `data_preparation.ipynb`
- Uses Google Gemini 2.5 Flash to analyze article titles
- Identifies 100 most relevant AI articles (excluding lists/tool collections)
- Creates a refined dataset with selected articles

**Workflow**:
1. Load filtered dataset from Hugging Face
2. Extract article IDs, URLs, and titles
3. Create prompt for Gemini model to identify relevant articles
4. Use LangChain with Google Gemini to get 100 article IDs
5. Filter dataset to include only selected articles
6. Push refined dataset to Hugging Face Hub

**Requirements**:
- Google Gemini API key (set in environment variables)
- `langchain-google-genai` library
- `python-dotenv` for environment variable management

**Configuration**:
```python
# Set GOOGLE_API_KEY in .env file or environment
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
```

**Output**:
- Refined dataset with 100 AI-related articles uploaded to Hugging Face Hub

---

### 5. create_metadata.py

**Purpose**: Python script to extract metadata (duration, ID, title) from audio dataset and save to CSV.

**Description**:
- Loads audio dataset from Hugging Face (`srinathmkce/wiki-ai-audio`)
- Calculates duration for each audio file
- Formats duration as `HH:MM:SS`
- Creates CSV file with columns: `id`, `title`, `duration`

**Usage**:
```bash
python create_metadata.py
```

**Output**:
- `metadata.csv` file containing:
  - `id`: Article/sample ID
  - `title`: Article title
  - `duration`: Audio duration in `HH:MM:SS` format

**Features**:
- Progress tracking for large datasets
- Error handling for missing or invalid audio
- Summary statistics (total samples, average duration, etc.)

**Requirements**:
- `datasets` library
- `pandas` library
- `librosa` and `soundfile` for audio processing

**Example Output**:
```csv
id,title,duration
1,Artificial Intelligence,00:05:23
2,Machine Learning,00:03:45
...
```

---

### 6. merge_audio_with_noise.py

**Purpose**: Python script to merge background noise with audio files from the dataset.

**Description**:
- Processes all audio samples from the dataset
- Randomly selects background noise from `noise1/noise1.wav` to `noise1/noise10.wav`
- Merges noise with audio at configurable volume levels
- Saves both original and merged audio files for comparison

**Usage**:

**Process all audios (default)**:
```bash
python merge_audio_with_noise.py
```

**Process all with custom settings**:
```bash
python merge_audio_with_noise.py --output-dir temp/my_audios --noise-db 10
```

**Process a single audio**:
```bash
python merge_audio_with_noise.py --sample-index 5
```

**Command-line Arguments**:
- `--sample-index`: Process single sample by index (default: None, processes all)
- `--output-dir`: Output directory for audio files (default: `temp/merged_audios`)
- `--noise-db`: Noise volume reduction in dB (default: 10.0, lower = louder noise)

**Features**:
- **Random noise selection**: Each audio gets a randomly selected noise file
- **Progress tracking**: Real-time progress bar with `tqdm`
- **Batch processing**: Efficiently processes all samples
- **Comparison files**: Saves both original and merged versions
- **Error handling**: Continues processing even if some samples fail

**Output Structure**:
```
output_dir/
├── {sample_id}_original.wav
├── {sample_id}_merged.wav
├── ...
```

**Noise Volume Settings**:
- `--noise-db 10`: Audible background noise (default)
- `--noise-db 15`: Moderate background noise
- `--noise-db 5`: Very prominent background noise
- `--noise-db 0`: Noise at same level as audio (may overpower)

**Requirements**:
- `datasets` library
- `pydub` for audio manipulation
- `numpy` for audio array processing
- `tqdm` for progress tracking
- `librosa` and `soundfile` for audio decoding

---

### 7. Package Management (uv)

**Purpose**: Modern Python package manager for dependency management.

**Description**:
- `uv` is a fast Python package installer and resolver
- `pyproject.toml` defines project dependencies
- `uv.lock` locks dependency versions for reproducibility

**Installation**:
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or on Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Usage**:

**Install dependencies**:
```bash
uv sync
```

**Add a new dependency**:
```bash
uv add package-name
```

**Run scripts**:
```bash
uv run python create_metadata.py
uv run python merge_audio_with_noise.py
```

**Key Dependencies** (from `pyproject.toml`):
- `kokoro>=0.9.4`: Text-to-speech engine
- `datasets>=4.4.1`: Hugging Face datasets
- `pydub>=0.25.1`: Audio manipulation
- `librosa>=0.11.0`: Audio processing
- `pandas>=2.3.3`: Data manipulation
- `soundfile>=0.13.1`: Audio file I/O
- And more...

**Benefits**:
- Fast dependency resolution
- Reproducible builds with lock file
- Virtual environment management
- Cross-platform support

---

## Usage Workflow

### Complete Pipeline

1. **Setup Environment**:
   ```bash
   # Install uv
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Install dependencies
   uv sync
   ```

2. **Start Kokoro TTS API**:
   ```bash
   # Build and run Docker container
   docker build -t kokoro-tts .
   docker run -p 8080:8080 kokoro-tts
   ```

3. **Extract Articles** (in Jupyter):
   - Run `data_preparation.ipynb` to filter Wikipedia articles
   - Authenticate with Hugging Face Hub
   - Push filtered dataset

4. **Filter Articles** (in Jupyter):
   - Run `data_filteration.ipynb` to select 100 AI articles
   - Requires Google Gemini API key
   - Push refined dataset

5. **Generate Speech** (external step):
   - Use Kokoro TTS API or `colab_inference.ipynb`
   - Convert article text to speech
   - Upload audio dataset to Hugging Face

6. **Create Metadata**:
   ```bash
   python create_metadata.py
   ```

7. **Add Background Noise**:
   ```bash
   python merge_audio_with_noise.py --output-dir temp/merged_audios --noise-db 10
   ```

## Configuration

### Environment Variables

Create a `.env` file for sensitive configuration:

```env
# Google Gemini API Key (for data_filteration.ipynb)
GOOGLE_API_KEY=your_api_key_here

# Hugging Face Token (for dataset uploads)
HF_TOKEN=your_hf_token_here
```

### Noise Files

Place background noise files in `noise1/` directory:
- `noise1.wav` through `noise10.wav`
- Supported formats: WAV
- Any sample rate/channels (will be converted automatically)

## Notes

- The project uses CPU-only PyTorch for Kokoro TTS to keep Docker image size small
- Background noise files should be in WAV format for best compatibility
- All audio processing maintains original sample rates and channel configurations
