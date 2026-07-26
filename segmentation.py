import logging
import os
from functools import lru_cache

import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

from models import SegmentationRequest
from prompts import build_box_prompt, build_point_prompt

logger = logging.getLogger(__name__)

CHECKPOINT_PATH = os.getenv("SAM_CHECKPOINT", "sam_models/sam_vit_h_4b8939.pth")
MODEL_TYPE = os.getenv("SAM_MODEL_TYPE", "vit_h")


@lru_cache(maxsize=1)
def get_predictor(checkpoint: str = CHECKPOINT_PATH, model_type: str = MODEL_TYPE) -> SamPredictor:
    """Loading the checkpoint costs several seconds and a couple of GB, so it is kept resident."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("loading SAM %s from %s on %s", model_type, checkpoint, device)
    model = sam_model_registry[model_type](checkpoint=checkpoint)
    model.to(device)
    return SamPredictor(model)


def predict_mask(image: np.ndarray, request: SegmentationRequest) -> np.ndarray:
    """Return a boolean mask with the same height and width as the RGB image given."""
    height, width = image.shape[:2]
    point_coords, point_labels = build_point_prompt(request, width, height)
    box = build_box_prompt(request, width, height)

    if point_coords is None and box is None:
        raise ValueError("no usable prompt survived validation against the image bounds")

    predictor = get_predictor()
    predictor.set_image(image)

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=False,
    )

    logger.info("segmented %dx%d image, score %.3f", width, height, float(scores[0]))
    return masks[0].astype(bool)
