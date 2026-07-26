# SAM Segmentation API

REST API that segments medical images with Meta's **Segment Anything Model (SAM)**. It is the
inference backend for the [SAM-OHIF](https://github.com/storesm/SAM-OHIF) viewer extension: the
viewer sends a captured image plus the marks a clinician drew on it, and the API returns the mask
as a PNG overlay and as a DICOM secondary capture.

The service is meant to run inside the hospital network. Images never leave the institution, which
is what keeps the workflow compatible with medical data protection rules.

## Design

The viewer posts an image and the marks drawn on it to `/segment_input_image/`, FastAPI runs SAM
against a checkpoint held in memory, and the viewer then fetches the overlay, the mask or the DICOM
file for that job.

Requests are identified by a **job id**, a random UUID minted per segmentation. Each job gets a
directory under `results/` holding everything needed to audit or replay it:

```
results/<job_id>/
  request.json      the prompts and job metadata, including the creation timestamp
  source.png        the image as received
  mask.png          the binary mask
  overlay.png       the mask blended over the image
  segmentation.dcm  DICOM secondary capture of the overlay
```

A UUID keeps identity separate from timing: two requests in the same instant cannot collide, the id
survives a URL path unambiguously, and it is not guessable. The creation time is metadata, recorded
in `request.json`.

## Endpoints

### `POST /segment_input_image/`

Multipart form with two fields:

| Field | Type | Description |
| --- | --- | --- |
| `file` | file | The image to segment (PNG, JPEG, anything OpenCV decodes) |
| `sam_input` | string | JSON prompt, see below |

```json
{
  "model_type": "vit_h",
  "positive_points": [{ "x": 250, "y": 300 }],
  "negative_points": [{ "x": 750, "y": 300 }],
  "boxes": [{ "start_x": 100, "start_y": 60, "width": 400, "height": 240 }]
}
```

Coordinates are pixels of the uploaded image. Boxes accept a negative `width` or `height`, so a
drag in any direction works. At least one point or one box is required.

Returns the job descriptor:

```json
{
  "job_id": "ffb57262d6df493a961499358bef3834",
  "created_at": "2026-07-26T15:06:29.971867+00:00",
  "model_type": "vit_h",
  "image_width": 1800,
  "image_height": 1200,
  "prompt_counts": { "positive_points": 1, "negative_points": 0, "boxes": 1 }
}
```

| Status | Meaning |
| --- | --- |
| 415 | The uploaded file could not be decoded as an image |
| 422 | The prompt is malformed, or nothing usable survived clipping to the image |

### `GET /get_segmented_image/{job_id}`

The mask blended over the image and outlined, as PNG.

### `GET /get_mask_image/{job_id}`

The raw binary mask as PNG, for callers that want to composite it themselves.

### `GET /get_dicom_image/{job_id}`

DICOM secondary capture of the overlay. No patient identity is attached: the source is an uploaded
rendering that carries none, so `PatientName` is `ANONYMOUS` and `PatientIdentityRemoved` is `YES`.

### `GET /health_check/`

`{"status": "ok"}` once the service is up.

Malformed job ids are rejected with 400 before anything touches the filesystem; unknown ones give
404.

## Installation

```bash
git clone https://github.com/storesm/SAM-API.git
cd SAM-API
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download a SAM checkpoint from the
[official repository](https://github.com/facebookresearch/segment-anything#model-checkpoints) and
put it where `SAM_CHECKPOINT` points:

```bash
mkdir -p sam_models
curl -L -o sam_models/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Running

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Interactive documentation is generated at `/docs`.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `SAM_CHECKPOINT` | `sam_models/sam_vit_h_4b8939.pth` | Path to the model weights |
| `SAM_MODEL_TYPE` | `vit_h` | Architecture matching the checkpoint |
| `SAM_RESULTS_DIR` | `results` | Where job directories are written |
| `SAM_ALLOWED_ORIGINS` | `http://localhost,http://localhost:3000` | Comma separated CORS origins |

The checkpoint is loaded once and kept resident. CUDA is used when available; on CPU a single
segmentation of a 1800×1200 image takes roughly 45 seconds, almost all of it computing the image
embedding.

## Tests

```bash
python test_segmentation.py
```

Covers the prompt geometry, the overlay blending, the PNG round trip and the DICOM output. It needs
neither the checkpoint nor a GPU. For a manual end to end check, post `images/truck.jpg` with a
point on a wheel and confirm the returned overlay tints the tyre.

## License

MIT. See [LICENSE](LICENSE).
