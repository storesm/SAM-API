import base64
from http.client import HTTPException
from io import BytesIO
import io
import json
from pathlib import Path
import re
import time
from tkinter import Image
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

# Configuración de los origenes permitidos para CORS
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

    # # Return the mask to the client
    # return {"masks": masks.tolist()}

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
    output_dicom_path = "output.dcm"
    save_dicom_from_image(overlaid_image, output_dicom_path)

    # Devolver la imagen al cliente
    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")


def save_dicom_from_image(image_array, output_path):
    # Convertir la imagen a escala de grises si es a color
    if len(image_array.shape) == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    # Escalar los píxeles para que estén en el rango adecuado para DICOM
    image_array = cv2.normalize(image_array, None, 0, 65535, cv2.NORM_MINMAX)

    # Crear un nuevo conjunto de datos DICOM
    ds = FileDataset(output_path, {}, file_meta=Dataset())

    # Información del paciente y del estudio
    ds.PatientName = "John Doe"
    ds.PatientID = "123456"
    ds.StudyInstanceUID = str(uuid.uuid4())
    ds.SeriesInstanceUID = str(uuid.uuid4())

    # Establecer el tipo de archivo DICOM
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    # Crear píxeles de la imagen
    ds.Rows, ds.Columns = image_array.shape
    ds.PixelData = image_array.tobytes()

    # Guardar el archivo DICOM
    ds.save_as(output_path)


def overlay_mask(image, masks):
    overlaid_image = np.copy(image)
    if masks is not None:
        # Convertir la matriz de máscara a una imagen binaria
        mask_image = np.uint8(masks[0] * 255)
        # Aplicar la máscara a la imagen original
        overlaid_image[mask_image != 0] = [
            0,
            0,
            255,
        ]  # Color rojo para las áreas de la máscara
    return overlaid_image


def encode_image(image):
    buffer = io.BytesIO()
    plt.imsave(buffer, image, format="png")
    img_bytes = buffer.getvalue()
    return img_bytes
