import numpy as np

from models import SegmentationRequest

POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0
MIN_BOX_SIDE = 1


def build_point_prompt(request: SegmentationRequest, width: int, height: int):
    """Stack positive and negative clicks into the (coords, labels) pair SAM takes."""
    coordinates = []
    labels = []

    for point in request.positive_points:
        coordinates.append([min(point.x, width - 1), min(point.y, height - 1)])
        labels.append(POSITIVE_LABEL)

    for point in request.negative_points:
        coordinates.append([min(point.x, width - 1), min(point.y, height - 1)])
        labels.append(NEGATIVE_LABEL)

    if not coordinates:
        return None, None

    return np.array(coordinates, dtype=np.float32), np.array(labels, dtype=np.int32)


def build_box_prompt(request: SegmentationRequest, width: int, height: int):
    """SAM accepts a single box per prediction, so only the first one is honoured."""
    if not request.boxes:
        return None

    x_min, y_min, x_max, y_max = request.boxes[0].to_corners()
    box = np.array(
        [
            np.clip(x_min, 0, width - 1),
            np.clip(y_min, 0, height - 1),
            np.clip(x_max, 0, width - 1),
            np.clip(y_max, 0, height - 1),
        ],
        dtype=np.float32,
    )

    if box[2] - box[0] < MIN_BOX_SIDE or box[3] - box[1] < MIN_BOX_SIDE:
        return None

    return box
