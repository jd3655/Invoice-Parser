import asyncio
import base64
import io
import os
import shutil
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import fitz  # type: ignore
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field


DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
IMAGE_DIR = DATA_DIR / "images"
OUTPUT_DIR = DATA_DIR / "output"
STATIC_DIR = Path(__file__).parent / "static"
INDEX_FILE = STATIC_DIR / "index.html"

PDF_CONCURRENCY = 2
PAGE_CONCURRENCY = 2
RENDER_SCALE = 3  # higher DPI for clearer OCR


class JobConfig(BaseModel):
    base_url: str = Field(default="http://127.0.0.1:1234")
    model: str = Field(default="gpt-4o-mini-vision")
    save_per_page: bool = Field(default=False)


@dataclass
class PageProgress:
    current: int = 0
    total: int = 0


@dataclass
class FileProgress:
    filename: str
    path: Path
    status: str = "queued"
    pages: PageProgress = field(default_factory=PageProgress)
    output_path: Optional[Path] = None
    error: Optional[str] = None


@dataclass
class JobState:
    job_id: str
    config: JobConfig
    status: str = "created"
    files: Dict[str, FileProgress] = field(default_factory=dict)
    error: Optional[str] = None
    task: Optional[asyncio.Task] = None


jobs: Dict[str, JobState] = {}
pdf_semaphore = asyncio.Semaphore(PDF_CONCURRENCY)
page_semaphore = asyncio.Semaphore(PAGE_CONCURRENCY)


app = FastAPI(title="Local PDF Invoice Parser")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def ensure_directories() -> None:
    """Create required data directories."""
    for path in [UPLOAD_DIR, IMAGE_DIR, OUTPUT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def get_job(job_id: str) -> JobState:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    if not INDEX_FILE.exists():
        raise HTTPException(status_code=500, detail="UI not found")
    return HTMLResponse(INDEX_FILE.read_text(encoding="utf-8"))


@app.post("/api/jobs")
async def create_job(config: JobConfig) -> Dict[str, str]:
    ensure_directories()
    job_id = str(uuid.uuid4())
    jobs[job_id] = JobState(job_id=job_id, config=config)
    return {"job_id": job_id}


@app.post("/api/jobs/{job_id}/upload")
async def upload_pdfs(job_id: str, files: List[UploadFile] = File(...)) -> Dict[str, int]:
    job = get_job(job_id)
    saved = 0
    upload_base = UPLOAD_DIR / job_id
    upload_base.mkdir(parents=True, exist_ok=True)

    for upload in files:
        if not upload.filename:
            continue
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in {".pdf", ".png"}:
            continue
        dest_path = upload_base / Path(upload.filename).name
        with dest_path.open("wb") as f:
            shutil.copyfileobj(upload.file, f)
        job.files[dest_path.name] = FileProgress(filename=dest_path.name, path=dest_path)
        saved += 1

    if saved == 0:
        raise HTTPException(status_code=400, detail="No PDF or PNG files uploaded")
    return {"saved": saved}


@app.post("/api/jobs/{job_id}/start")
async def start_job(job_id: str) -> Dict[str, str]:
    job = get_job(job_id)
    if job.status == "processing":
        raise HTTPException(status_code=400, detail="Job already processing")
    if not job.files:
        raise HTTPException(status_code=400, detail="No files uploaded for this job")

    job.status = "processing"
    job.error = None
    task = asyncio.create_task(process_job(job_id))
    job.task = task
    task.add_done_callback(lambda t: task_callback(job_id, t))
    return {"status": "started"}


def task_callback(job_id: str, task: asyncio.Task) -> None:
    if task.cancelled():
        return
    if task.exception():
        job = jobs.get(job_id)
        if job:
            job.status = "error"
            job.error = str(task.exception())


@app.get("/api/jobs/{job_id}/status")
async def job_status(job_id: str) -> Dict[str, object]:
    job = get_job(job_id)
    files_status = {}
    for name, progress in job.files.items():
        files_status[name] = {
            "status": progress.status,
            "pages": {"current": progress.pages.current, "total": progress.pages.total},
            "output_path": str(progress.output_path) if progress.output_path else None,
            "error": progress.error,
        }

    return {
        "job_id": job.job_id,
        "status": job.status,
        "error": job.error,
        "files": files_status,
    }


@app.get("/api/jobs/{job_id}/download")
async def download_results(job_id: str) -> StreamingResponse:
    job = get_job(job_id)
    output_folder = OUTPUT_DIR / job_id
    if not output_folder.exists():
        raise HTTPException(status_code=404, detail="No outputs for this job")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(output_folder):
            for filename in files:
                path = Path(root) / filename
                arcname = path.relative_to(output_folder)
                zipf.write(path, arcname.as_posix())
    buffer.seek(0)
    headers = {
        "Content-Disposition": f'attachment; filename="invoice_outputs_{job_id}.zip"'
    }
    return StreamingResponse(buffer, media_type="application/zip", headers=headers)


async def process_job(job_id: str) -> None:
    job = get_job(job_id)
    tasks = []
    for file_progress in job.files.values():
        tasks.append(asyncio.create_task(process_single_file(job, file_progress)))
    await asyncio.gather(*tasks, return_exceptions=True)

    if any(fp.status == "error" for fp in job.files.values()):
        job.status = "error"
    else:
        job.status = "completed"


async def process_single_file(job: JobState, file_progress: FileProgress) -> None:
    async with pdf_semaphore:
        try:
            file_progress.status = "rendering"
            images = await prepare_images(job.job_id, file_progress)
            file_progress.pages.total = len(images)
            file_progress.status = "ocr"
            ocr_texts: List[str] = []
            for idx, image_path in enumerate(images):
                file_progress.pages.current = idx + 1
                text = await ocr_image(
                    image_path=image_path,
                    model=job.config.model,
                    base_url=job.config.base_url,
                )
                ocr_texts.append(text)
                if job.config.save_per_page:
                    await save_page_text(job.job_id, file_progress, idx + 1, text)
            combined_text = build_combined_text(ocr_texts)
            output_path = await save_combined_text(job.job_id, file_progress, combined_text)
            file_progress.output_path = output_path
            file_progress.status = "done"
        except Exception as exc:
            file_progress.status = "error"
            file_progress.error = str(exc)


async def prepare_images(job_id: str, file_progress: FileProgress) -> List[Path]:
    suffix = file_progress.path.suffix.lower()
    if suffix == ".pdf":
        return await render_pdf_to_images(job_id, file_progress)
    if suffix == ".png":
        images_dir = IMAGE_DIR / job_id / Path(file_progress.filename).stem
        images_dir.mkdir(parents=True, exist_ok=True)
        destination = images_dir / Path(file_progress.filename).name
        if destination != file_progress.path:
            shutil.copy(file_progress.path, destination)
        return [destination]
    raise ValueError("Unsupported file type. Only PDF and PNG files are allowed.")


async def render_pdf_to_images(job_id: str, file_progress: FileProgress) -> List[Path]:
    images_dir = IMAGE_DIR / job_id / Path(file_progress.filename).stem
    images_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    pdf_path = file_progress.path

    def _render() -> List[Path]:
        output_paths: List[Path] = []
        doc = fitz.open(pdf_path)
        try:
            for page_number in range(doc.page_count):
                page = doc.load_page(page_number)
                pix = page.get_pixmap(matrix=fitz.Matrix(RENDER_SCALE, RENDER_SCALE))
                filename = f"page_{page_number + 1:03}.png"
                image_path = images_dir / filename
                pix.save(image_path.as_posix(), output="png")
                output_paths.append(image_path)
        finally:
            doc.close()
        return output_paths

    loop = asyncio.get_running_loop()
    paths = await loop.run_in_executor(None, _render)
    return paths


async def ocr_image(image_path: Path, model: str, base_url: str) -> str:
    async with page_semaphore:
        image_bytes = image_path.read_bytes()
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        image_data_url = f"data:image/png;base64,{encoded_image}"
        client = OpenAI(base_url=f"{base_url.rstrip('/')}/v1", api_key="lm-studio")

        prompt = (
            "Perform OCR on the provided invoice page image. "
            "Preserve all numbers, punctuation, currency symbols, and line breaks. "
            "Keep table alignment as text where possible. "
            "Return only the transcribed text. If the page is blank, return an empty string. "
            "If the content is faint or low-contrast, do your best to transcribe it accurately."
        )

        def _call() -> str:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=2048,
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "OCR this invoice page and return raw text only.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data_url, "detail": "high"},
                            },
                        ],
                    },
                ],
            )
            return response.choices[0].message.content or ""

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _call)


def build_combined_text(pages: List[str]) -> str:
    output_lines: List[str] = []
    total = len(pages)
    for idx, text in enumerate(pages):
        output_lines.append(f"----- PAGE {idx + 1} / {total} -----")
        output_lines.append(text.strip())
        output_lines.append("")
    return "\n".join(output_lines).strip() + "\n"


async def save_combined_text(job_id: str, file_progress: FileProgress, content: str) -> Path:
    output_dir = OUTPUT_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(file_progress.filename).stem}.txt"

    def _write() -> None:
        output_path.write_text(content, encoding="utf-8")

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _write)
    return output_path


async def save_page_text(job_id: str, file_progress: FileProgress, page_number: int, content: str) -> None:
    output_dir = OUTPUT_DIR / job_id / Path(file_progress.filename).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    page_path = output_dir / f"page_{page_number:03}.txt"

    def _write() -> None:
        page_path.write_text(content.strip() + "\n", encoding="utf-8")

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _write)


@app.on_event("startup")
async def startup_event() -> None:
    ensure_directories()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
