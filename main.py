import json
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import ValidationError

from imaging import decode_image, encode_mask_png, encode_png, overlay_mask, write_dicom
from models import SegmentationJob, SegmentationRequest
from segmentation import predict_mask

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(os.getenv("SAM_RESULTS_DIR", "results"))
ALLOWED_ORIGINS = os.getenv(
    "SAM_ALLOWED_ORIGINS", "http://localhost,http://localhost:3000"
).split(",")
JOB_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
SOURCE_NAME = "source.png"
MASK_NAME = "mask.png"
OVERLAY_NAME = "overlay.png"
DICOM_NAME = "segmentation.dcm"
REQUEST_NAME = "request.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("results stored under %s", RESULTS_DIR.resolve())
    yield


app = FastAPI(title="SAM Segmentation API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def artifact_path(job_id: str, filename: str) -> Path:
    """Resolve a stored artifact, rejecting ids that could escape the results directory."""
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise HTTPException(status_code=400, detail="malformed job id")

    path = RESULTS_DIR / job_id / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"no {filename} stored for job {job_id}")

    return path


@app.get("/health_check/")
async def health_check():
    return {"status": "ok"}


@app.post("/segment_input_image/", response_model=SegmentationJob)
async def segment_input_image(file: UploadFile = File(...), sam_input: str = Form(...)):
    try:
        request = SegmentationRequest.model_validate_json(sam_input)
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=json.loads(error.json())) from error

    try:
        image = decode_image(await file.read())
    except ValueError as error:
        raise HTTPException(status_code=415, detail=str(error)) from error

    try:
        mask = predict_mask(image, request)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    overlay = overlay_mask(image, mask)

    job_id = uuid.uuid4().hex
    job_dir = RESULTS_DIR / job_id
    job_dir.mkdir(parents=True)

    job = SegmentationJob(
        job_id=job_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        model_type=request.model_type,
        image_width=image.shape[1],
        image_height=image.shape[0],
        prompt_counts={
            "positive_points": len(request.positive_points),
            "negative_points": len(request.negative_points),
            "boxes": len(request.boxes),
        },
    )

    # Everything that produced the result is kept so a segmentation can be audited or replayed
    (job_dir / REQUEST_NAME).write_text(
        json.dumps({"job": job.model_dump(), "prompt": request.model_dump()}, indent=2)
    )
    (job_dir / SOURCE_NAME).write_bytes(encode_png(image))
    (job_dir / MASK_NAME).write_bytes(encode_mask_png(mask))
    (job_dir / OVERLAY_NAME).write_bytes(encode_png(overlay))
    write_dicom(overlay, job_id, job_dir / DICOM_NAME)

    logger.info("job %s stored", job_id)
    return job


@app.get("/get_segmented_image/{job_id}")
async def get_segmented_image(job_id: str):
    return FileResponse(
        artifact_path(job_id, OVERLAY_NAME),
        media_type="image/png",
        filename=f"overlay_{job_id}.png",
    )


@app.get("/get_mask_image/{job_id}")
async def get_mask_image(job_id: str):
    return FileResponse(
        artifact_path(job_id, MASK_NAME),
        media_type="image/png",
        filename=f"mask_{job_id}.png",
    )


@app.get("/get_dicom_image/{job_id}")
async def get_dicom_image(job_id: str):
    return FileResponse(
        artifact_path(job_id, DICOM_NAME),
        media_type="application/dicom",
        filename=f"segmentation_{job_id}.dcm",
    )
