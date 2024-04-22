import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
from utils.plotting import show_image, show_raw_image

# Loading and showing example image
image = cv2.imread("images/truck.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
show_raw_image((10, 10), image)

# Selecting objects with SAM (combining points and boxes)
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

input_box = np.array([425, 600, 700, 875])
input_point = np.array([[575, 750]])
input_label = np.array([0])

show_image((10, 10), image, None, input_box, input_point, input_label)

# Predicting masks
predictor.set_image(image)

masks, _, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    box=input_box,
    multimask_output=False,
)

show_image((10, 10), image, masks, input_box, input_point, input_label)
