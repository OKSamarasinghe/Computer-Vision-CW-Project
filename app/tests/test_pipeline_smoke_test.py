"""
Lightweight smoke tests for the preprocessing pipeline. These do NOT require the trained model file,
so they can run before the model is copied into app/models/, or in CI. Run with:

    cd app && pytest tests/
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import preprocessing
from config import TARGET_IMAGE_SIZE


def _make_synthetic_fundus_image():
    """A small synthetic BGR image with a black border and a bright circular centre, standing in
    for a real fundus photograph so these tests don't depend on any sample data being present."""
    synthetic_image = np.zeros((300, 300, 3), dtype=np.uint8)
    center_y, center_x = 150, 150
    y_grid, x_grid = np.ogrid[:300, :300]
    circular_mask = (y_grid - center_y) ** 2 + (x_grid - center_x) ** 2 <= 120 ** 2
    synthetic_image[circular_mask] = [40, 90, 160]
    return synthetic_image


def test_crop_dark_border_reduces_or_preserves_image_size():
    synthetic_image = _make_synthetic_fundus_image()
    cropped_image = preprocessing.crop_dark_border_from_fundus_image(synthetic_image)
    assert cropped_image.shape[0] <= synthetic_image.shape[0]
    assert cropped_image.shape[1] <= synthetic_image.shape[1]


def test_resize_produces_the_configured_target_size():
    synthetic_image = _make_synthetic_fundus_image()
    resized_image = preprocessing.resize_fundus_image_to_target_size(synthetic_image)
    # cv2.resize's target_size argument is (width, height); shape is (height, width, channels).
    assert resized_image.shape[1::-1] == TARGET_IMAGE_SIZE


def test_full_pipeline_runs_end_to_end_without_error():
    synthetic_image = _make_synthetic_fundus_image()
    output_image = preprocessing.preprocess_fundus_image_for_inference(synthetic_image)
    assert output_image.shape[:2] == TARGET_IMAGE_SIZE[::-1]
    assert output_image.dtype == np.uint8


def test_pipeline_with_intermediate_steps_returns_all_five_stages():
    synthetic_image = _make_synthetic_fundus_image()
    _, intermediate_steps = preprocessing.preprocess_fundus_image_for_inference(
        synthetic_image, return_intermediate_steps=True
    )
    assert len(intermediate_steps) == 5
