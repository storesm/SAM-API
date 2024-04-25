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
from models.api_models import Item, SAMInput
from fastapi.middleware.cors import CORSMiddleware

import matplotlib.pyplot as plt
from preprocessing import decode_image, decode_inputs, format_inputs_to_sam
from sam.sam import sam_execution
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


@app.post("/get_image_and_parameters/")
async def get_image_and_parameters(
    file: UploadFile = File(...),
    sam_input: str = Form(...),
):
    sam_input = json.loads(sam_input)

    image = await decode_image(file)
    print("imagen shape: ", image.shape)

    positive_points, negative_points, input_box = decode_inputs(sam_input)

    input_points, input_labels = format_inputs_to_sam(positive_points, negative_points)

    masks = sam_execution(image, input_box, input_points, input_labels, verbose=True)

    # Return the mask to the client
    return {"masks": masks.tolist()}
