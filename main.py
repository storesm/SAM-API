import base64
from http.client import HTTPException
from io import BytesIO
import io
import json
from pathlib import Path
import re
import time
from typing import Union
import uuid

import cv2
from fastapi import FastAPI, Form, Response, UploadFile, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, StreamingResponse
import numpy as np
from models.api_models import Item, SAMInput
from fastapi.middleware.cors import CORSMiddleware

import matplotlib.pyplot as plt
from preprocessing import decode_image, decode_inputs, format_inputs_to_sam
from sam.sam import sam_execution
from sam.utils.plotting import show_image, show_raw_image

import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian

app = FastAPI()

# Configuración de los orígenes permitidos para CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.post("/segment_input_image/")
async def segment_input_image(
    file: UploadFile = File(...),
    sam_input: str = Form(...),
):
    sam_input = json.loads(sam_input)

    image = await decode_image(file)
    print("imagen shape: ", image.shape)

    # Variable with the actual date and time
    now = time.time()

    # Decode the inputs
    positive_points, negative_points, input_box = decode_inputs(sam_input)

    # Format the inputs to be used in the SAM model
    input_points, input_labels = format_inputs_to_sam(positive_points, negative_points)

    # Execute the SAM model
    masks = sam_execution(image, input_box, input_points, input_labels, verbose=False)

    overlaid_image = overlay_mask(image, masks)

    # Codificar la imagen resultante en formato PNG
    img_bytes = encode_image(overlaid_image)

    # Save the inputs in the server
    inputs_path = f"log/inputs_{now}.json"
    with open(inputs_path, "w") as f:
        json.dump(sam_input, f)

    # Save the image in the server
    image_path = f"log/image_{now}.png"
    cv2.imwrite(image_path, image)

    # Save the masked image in the server
    overlaid_image_path = f"log/overlaid_image_{now}.png"
    cv2.imwrite(overlaid_image_path, overlaid_image)

    # Save the ndarray of masks in the server
    masks_path = f"log/masks_matrix{now}.npy"
    np.save(masks_path, masks)

    # Utiliza la función para guardar el archivo DICOM
    output_dicom_path = f"log/output_{now}.dcm"
    save_dicom_from_image(overlaid_image, output_dicom_path)

    # Guardar la imagen PNG en un archivo temporal
    segmented_image_path = f"log/segmented_image_{now}.png"
    with open(segmented_image_path, "wb") as f:
        f.write(img_bytes)

    # Devolver solo el timestamp
    return {"timestamp": now}

@app.get("/get_segmented_image/{timestamp}")
async def get_segmented_image(timestamp: float):
    segmented_image_path = f"log/segmented_image_{timestamp}.png"
    if Path(segmented_image_path).exists():
        return FileResponse(segmented_image_path, media_type="image/png", filename=f"segmented_image_{timestamp}.png")
    else:
        raise HTTPException(status_code=404, detail="Segmented image not found")

@app.get("/get_dicom_image/{timestamp}")
async def get_dicom_image(timestamp: float):
    output_dicom_path = f"log/output_{timestamp}.dcm"
    if Path(output_dicom_path).exists():
        return FileResponse(output_dicom_path, media_type="application/dicom", filename=f"output_{timestamp}.dcm")
    else:
        raise HTTPException(status_code=404, detail="DICOM file not found")

def save_dicom_from_image(image_array, output_path):
    # Verificar si la imagen es a color
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        # Convertir la imagen a RGB
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    
    # Escalar los píxeles para que estén en el rango adecuado para DICOM
    image_array = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Crear un nuevo conjunto de datos DICOM
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset(output_path, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # Información del paciente y del estudio
    ds.PatientName = "John Doe"
    ds.PatientID = "123456"
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.Modality = "OT"  # Other
    ds.StudyDate = time.strftime("%Y%m%d")
    ds.StudyTime = time.strftime("%H%M%S")
    ds.ContentDate = time.strftime("%Y%m%d")
    ds.ContentTime = time.strftime("%H%M%S")

    # Establecer dimensiones de la imagen
    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "RGB"
    ds.Rows, ds.Columns, _ = image_array.shape
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PlanarConfiguration = 0
    ds.PixelData = image_array.tobytes()

    # Guardar el archivo DICOM
    ds.save_as(output_path)

def overlay_mask(image, masks):
    overlaid_image = np.copy(image)
    if masks is not None:
        # Crear una imagen binaria de la máscara
        mask_image = np.uint8(masks[0] * 255)
        # Crear una imagen de color rojo para la máscara
        red_image = np.zeros_like(image)
        red_image[:, :] = [0, 0, 255]
        # Superponer la imagen roja sobre la imagen original donde la máscara no es cero
        overlaid_image = np.where(mask_image[:, :, None] != 0, red_image, overlaid_image)
    return overlaid_image

def encode_image(image):
    is_success, buffer = cv2.imencode(".png", image)
    return buffer.tobytes()
