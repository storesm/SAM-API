import base64
from http.client import HTTPException
from io import BytesIO
import re
from typing import Union

from fastapi import FastAPI, Response, UploadFile, File
from fastapi.encoders import jsonable_encoder
from api_models import Item, SAMInput
from fastapi.middleware.cors import CORSMiddleware
import os

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


def save_image(file):
    # Guardar la imagen
    with open("images/captured_image.png", "wb") as buffer:
        buffer.write(file)


@app.post("/get_image/")
async def get_image(file: UploadFile = File(...)):
    save_image(await file.read())
    # plot the image with matplotlib:
    import cv2
    import matplotlib.pyplot as plt

    image = cv2.imread("images/captured_image.png")
    if image is None:
        raise HTTPException(
            status_code=400, detail="El archivo no es una imagen válida"
        )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()

    print("captured_image.png")


@app.post("/get_parameters/")
async def get_parameters(sam_input: SAMInput):
    print(sam_input)
    for key, value in sam_input.model_dump().items():
        print(key, value)
