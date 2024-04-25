import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
from .utils.plotting import show_image, show_raw_image

MOCKED_IMAGE_PATH = "images/truck.jpg"
MOCKED_SAM_CHECKPOINT = "sam/sam_models/sam_vit_h_4b8939.pth"
MOCKED_MODEL_TYPE = "vit_h"
FIGSIZE = (10, 8)


def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_sam_predictor(checkpoint, model_type):
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    return SamPredictor(sam)


def predict_masks(image, predictor, input_box, input_point, input_label):
    predictor.set_image(image)

    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box,
        multimask_output=False,
    )

    return masks


def sam_execution(
    image,
    input_box,
    input_point,
    input_label,
    model_checkpoint=MOCKED_SAM_CHECKPOINT,
    model_type=MOCKED_MODEL_TYPE,
    verbose=False,
):
    if verbose:
        show_raw_image(FIGSIZE, image)
    predictor = get_sam_predictor(model_checkpoint, model_type)
    if verbose:
        show_image(FIGSIZE, image, None, input_box, input_point, input_label)
    masks = predict_masks(image, predictor, input_box, input_point, input_label)
    if verbose:
        show_image(FIGSIZE, image, masks, input_box, input_point, input_label)
    return masks


if __name__ == "__main__":
    # Defining mocked inputs
    image = load_image(MOCKED_IMAGE_PATH)
    input_box = np.array([425, 600, 700, 875])
    input_point = np.array([[575, 750]])
    input_label = np.array([0])

    masks = sam_execution(image, input_box, input_point, input_label, verbose=True)
