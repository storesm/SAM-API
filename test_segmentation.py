import tempfile
from pathlib import Path

import numpy as np
import pydicom

from imaging import decode_image, encode_mask_png, encode_png, overlay_mask, write_dicom
from models import Box, Point, SegmentationRequest
from prompts import NEGATIVE_LABEL, POSITIVE_LABEL, build_box_prompt, build_point_prompt

WIDTH = 64
HEIGHT = 48


def sample_image() -> np.ndarray:
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[:, :, 1] = 200
    return image


def sample_mask() -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[10:20, 10:20] = True
    return mask


def test_box_corners_normalise_a_backwards_drag():
    dragged_up_left = Box(start_x=40, start_y=30, width=-15, height=-10)
    assert dragged_up_left.to_corners() == (25, 20, 40, 30)

    dragged_down_right = Box(start_x=25, start_y=20, width=15, height=10)
    assert dragged_down_right.to_corners() == (25, 20, 40, 30)


def test_box_prompt_is_clipped_to_the_image():
    request = SegmentationRequest(boxes=[Box(start_x=10, start_y=10, width=500, height=500)])
    box = build_box_prompt(request, WIDTH, HEIGHT)
    assert box.tolist() == [10, 10, WIDTH - 1, HEIGHT - 1]


def test_degenerate_box_is_dropped_rather_than_sent_to_sam():
    request = SegmentationRequest(boxes=[Box(start_x=10, start_y=10, width=0, height=0)])
    assert build_box_prompt(request, WIDTH, HEIGHT) is None


def test_point_labels_follow_positive_then_negative_order():
    request = SegmentationRequest(
        positive_points=[Point(x=5, y=5), Point(x=6, y=6)],
        negative_points=[Point(x=7, y=7)],
    )
    coords, labels = build_point_prompt(request, WIDTH, HEIGHT)

    assert coords.tolist() == [[5, 5], [6, 6], [7, 7]]
    assert labels.tolist() == [POSITIVE_LABEL, POSITIVE_LABEL, NEGATIVE_LABEL]


def test_points_are_clamped_inside_the_image():
    request = SegmentationRequest(positive_points=[Point(x=9999, y=9999)])
    coords, _ = build_point_prompt(request, WIDTH, HEIGHT)
    assert coords.tolist() == [[WIDTH - 1, HEIGHT - 1]]


def test_a_box_only_request_carries_no_point_prompt():
    request = SegmentationRequest(boxes=[Box(start_x=1, start_y=1, width=20, height=20)])
    coords, labels = build_point_prompt(request, WIDTH, HEIGHT)
    assert coords is None and labels is None


def test_prompt_free_request_is_rejected():
    try:
        SegmentationRequest()
    except ValueError:
        return
    raise AssertionError("a request with no points and no boxes must not validate")


def test_overlay_tints_the_mask_and_leaves_the_rest_untouched():
    image = sample_image()
    overlay = overlay_mask(image, sample_mask())

    assert overlay.shape == image.shape
    assert not np.array_equal(overlay[15, 15], image[15, 15])
    assert np.array_equal(overlay[40, 40], image[40, 40])
    # blending keeps the anatomy readable instead of replacing it with flat colour
    assert overlay[15, 15][1] > 0


def test_png_round_trip_preserves_channel_order():
    image = sample_image()
    assert np.array_equal(decode_image(encode_png(image)), image)


def test_mask_png_is_binary():
    decoded = decode_image(encode_mask_png(sample_mask()))
    assert set(np.unique(decoded).tolist()) == {0, 255}


def test_dicom_keeps_pixels_and_drops_identity():
    overlay = overlay_mask(sample_image(), sample_mask())

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "segmentation.dcm"
        write_dicom(overlay, "0" * 32, output)
        dataset = pydicom.dcmread(output)

    assert dataset.PatientName == "ANONYMOUS"
    assert dataset.PatientIdentityRemoved == "YES"
    assert (dataset.Rows, dataset.Columns) == (HEIGHT, WIDTH)
    assert np.array_equal(dataset.pixel_array, overlay)


if __name__ == "__main__":
    for name, case in sorted(globals().items()):
        if name.startswith("test_"):
            case()
            print("ok", name)
