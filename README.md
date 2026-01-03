# Local PDF Invoice Parser

FastAPI web app that converts PDF invoices to plain text using a local LMStudio server (OpenAI-compatible). PDFs are rendered to images with PyMuPDF, OCRed via a vision model, and saved as `.txt` files (with optional per-page text). Outputs persist on disk and can be downloaded as a ZIP.

## Features
- Drag & drop multiple PDFs or select an entire folder (browser folder picker).
- Configure LMStudio base URL and model in the UI.
- Local-only processing: PDFs -> PNGs -> LMStudio vision model -> `.txt`.
- Progress tracking per file and per page.
- Download all outputs as a ZIP; files saved under `data/output/`.

## Requirements
- Python 3.10+
- LMStudio running locally with an OpenAI-compatible endpoint (e.g., `http://127.0.0.1:1234`) and a vision-capable model loaded.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want to force wheel-only installs (to avoid slow/fragile source builds), use:

```bash
pip install --only-binary=:all: -r requirements.txt
```

> Note: Older PyMuPDF pins may not ship wheels for newer Python versions (e.g., Python 3.14 on Apple Silicon), causing `pip` to attempt a source build that fails during MuPDF/zlib compilation. The current dependency pinning uses a recent PyMuPDF wheel to prevent this.

## Run the server
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000/ in your browser.

## Usage
1. Enter your LMStudio base URL (e.g., `http://127.0.0.1:1234`) and model name (e.g., `gpt-4o-mini-vision` or your loaded vision model).
2. Optionally enable “Also save per-page .txt files”.
3. Drag & drop PDFs, select multiple files, or pick a folder containing PDFs.
4. Click **Start Processing**.
5. Monitor progress per file and per page. When finished, download the ZIP or access saved outputs under `data/output/<job_id>/`.

## Storage layout
- Uploads: `data/uploads/<job_id>/`
- Rendered images: `data/images/<job_id>/<pdf_stem>/page_XXX.png`
- Outputs: `data/output/<job_id>/<pdf_stem>.txt` (and optional per-page files in a subfolder)

## Notes
- OCR calls use the OpenAI SDK configured with `base_url=<LMStudio>/v1` and `api_key="lm-studio"`. The app accepts either `<LMStudio>` or `<LMStudio>/v1` and normalizes the URL automatically.
- Vision requests send `image_url` as a local `file://...` URI (rather than base64 data URLs) to align with LMStudio/DeepSeek OCR expectations. Run LMStudio on the same host so it can read the temporary image files.
- Rendering uses PyMuPDF at 2x scaling for better OCR fidelity.
- Concurrency is limited (PDFs: 2 at a time; pages: 2 at a time) to avoid overwhelming the local model.
