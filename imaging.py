from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset

MASK_COLOR_RGB = (255, 64, 64)
MASK_OPACITY = 0.45
CONTOUR_THICKNESS = 2


def decode_image(data: bytes) -> np.ndarray:
    """Decode an uploaded image to RGB, the channel order SAM was trained on."""
    buffer = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("the uploaded file is not a decodable image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def encode_png(image: np.ndarray) -> bytes:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise ValueError("PNG encoding failed")
    return buffer.tobytes()


def encode_mask_png(mask: np.ndarray) -> bytes:
    """Single channel mask where the segmented region is opaque white."""
    return encode_png(mask.astype(np.uint8) * 255)


def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend the mask over the image and outline it, so the anatomy stays readable underneath."""
    overlay = image.copy()
    tint = np.zeros_like(image)
    tint[:] = MASK_COLOR_RGB

    blended = cv2.addWeighted(image, 1 - MASK_OPACITY, tint, MASK_OPACITY, 0)
    overlay[mask] = blended[mask]

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, MASK_COLOR_RGB, CONTOUR_THICKNESS)

    return overlay


def write_dicom(image: np.ndarray, job_id: str, output_path: Path) -> None:
    """Secondary capture of the overlaid result, with no patient identity attached.

    A real DICOM SEG object would carry the mask as a proper segmentation rather
    than burnt-in pixels, switch to highdicom if a PACS has to ingest this.
    """
    now = datetime.now(timezone.utc)

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    dataset = FileDataset(str(output_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = pydicom.uid.generate_uid()
    dataset.SeriesInstanceUID = pydicom.uid.generate_uid()

    # The source is an uploaded rendering, so there is no identity to carry over
    dataset.PatientName = "ANONYMOUS"
    dataset.PatientID = job_id
    dataset.PatientIdentityRemoved = "YES"
    dataset.Modality = "OT"
    dataset.SeriesDescription = "SAM segmentation overlay"
    dataset.ConversionType = "WSD"
    dataset.StudyDate = now.strftime("%Y%m%d")
    dataset.StudyTime = now.strftime("%H%M%S")
    dataset.ContentDate = dataset.StudyDate
    dataset.ContentTime = dataset.StudyTime
    dataset.SeriesNumber = 1
    dataset.InstanceNumber = 1

    dataset.SamplesPerPixel = 3
    dataset.PhotometricInterpretation = "RGB"
    dataset.PlanarConfiguration = 0
    dataset.Rows, dataset.Columns = image.shape[:2]
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.PixelData = np.ascontiguousarray(image, dtype=np.uint8).tobytes()

    dataset.save_as(output_path, enforce_file_format=True)
