import base64
from http.client import HTTPException
from io import BytesIO
import json
from pathlib import Path
import re
import time
from typing import Union
import uuid

from fastapi import FastAPI, Form, Response, UploadFile, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse
import numpy as np
from api_models import Item, SAMInput
from fastapi.middleware.cors import CORSMiddleware
import os
import cv2
import matplotlib.pyplot as plt
from sam.utils.plotting import show_image, show_raw_image

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


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}


# Base de datos simulada de sesiones de usuario
sessions_db = {}


def save_image(file, sessionID):
    # Guardar la imagen
    with open(f"images/{sessionID}.png", "wb") as buffer:
        buffer.write(file)


@app.post("/get_image_and_parameters/{sessionID}")
async def get_image_and_parameters(
    sessionID: str,
    file: UploadFile = File(...),
    sam_input: str = Form(...),
):
    sam_input = json.loads(sam_input)

    # Preprocesar la imagen sin guardarla en disco
    image = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Get all the positive_points in sam_input
    print("Printing each value in SAMInput")
    for key, value in sam_input.items():
        print(key, value)

    positive_points = np.array(
        [[point["x"], point["y"]] for point in sam_input["positive_points"]]
    )

    input_point = positive_points
    input_label = np.array([0, 0])

    show_image((10, 8), image, None, None, input_point, input_label)

    print(f"images/{sessionID}.png")
