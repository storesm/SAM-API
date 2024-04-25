import os
import cv2
import numpy as np


async def decode_image(file):
    image = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    return image


def decode_inputs(sam_input):
    positive_points = np.array(
        [[point["x"], point["y"]] for point in sam_input["positive_points"]]
    )

    negative_points = np.array(
        [[point["x"], point["y"]] for point in sam_input["negative_points"]]
    )
    input_box = (
        np.array(
            [
                sam_input["rectangles"][0]["startX"],
                sam_input["rectangles"][0]["startY"],
                sam_input["rectangles"][0]["width"],
                sam_input["rectangles"][0]["height"],
            ]
        )
        if sam_input["rectangles"]
        else None
    )
    return positive_points, negative_points, input_box


def format_inputs_to_sam(positive_points, negative_points):
    input_points = np.concatenate((positive_points, negative_points))

    # 0 para los positivos y 1 para los negativos
    zero_labels = np.ones(len(positive_points))
    one_labels = np.zeros(len(negative_points))
    input_labels = np.concatenate((zero_labels, one_labels))

    return input_points, input_labels
