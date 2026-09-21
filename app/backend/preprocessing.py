"""
Fundus image preprocessing pipeline.

This module is a direct port of the preprocessing pipeline defined and validated in Notebook 2 (Data
Preprocessing) of the Kaggle coursework -- see notebooks/02_data_preprocessing/ for the original,
executed version with printed outputs. Every function below was copied over with the same parameter
values Notebook 2 used, not re-derived from scratch.

Keeping this pipeline identical between training (Kaggle) and inference (this local application)
matters: any mismatch here would mean the model receives images it was never trained to recognise,
silently degrading every prediction the application makes. The steps, in order, are:

    1. crop_dark_border_from_fundus_image          -- remove non-diagnostic black background
    2. resize_fundus_image_to_target_size          -- standardise to 224x224
    3. apply_clahe_contrast_enhancement            -- local micro-contrast boost (LAB, L-channel only)
    4. apply_ben_graham_local_average_subtraction  -- cancel large-scale illumination variation

Note on pixel range: Notebook 2's original pipeline had a 5th step, normalising pixels to [0, 1] for
storage. That step is intentionally NOT reproduced here. Notebook 5 Cell 6 added an explicit runtime
assertion confirming EfficientNetB0 in this project expects raw [0, 255]-range input -- the saved
preprocessed training images were written out as 0-255 PNGs and re-loaded that way for training, not
fed in as normalised [0, 1] floats. Feeding this application's output straight to the model (without
dividing by 255) matches that training-time behaviour exactly. See docs/innovation_design_notes.md.
"""

import cv2
import numpy as np

from config import (
    TARGET_IMAGE_SIZE,
    DARK_BORDER_CROP_THRESHOLD,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    BEN_GRAHAM_BLUR_SIGMA_FRACTION,
)


def crop_dark_border_from_fundus_image(fundus_image_bgr, darkness_threshold=DARK_BORDER_CROP_THRESHOLD):
    """Crop the near-black circular border surrounding a raw fundus photograph.

    The black background outside the circular retinal field of view carries no diagnostic
    information, so removing it (a) counts as a noise-removal step and (b) makes the effective
    resolution of the retina itself more consistent across images of differing raw sizes, before
    resizing. Identical to Notebook 2 Cell 4.
    """
    grayscale_image = cv2.cvtColor(fundus_image_bgr, cv2.COLOR_BGR2GRAY)
    non_black_pixel_mask = grayscale_image > darkness_threshold

    if non_black_pixel_mask.sum() == 0:
        # Safety fallback: a degenerate (e.g. fully black/corrupted) image is returned unchanged
        # rather than crashing the pipeline.
        return fundus_image_bgr

    rows_with_content = np.any(non_black_pixel_mask, axis=1)
    columns_with_content = np.any(non_black_pixel_mask, axis=0)
    top_row, bottom_row = np.where(rows_with_content)[0][[0, -1]]
    left_column, right_column = np.where(columns_with_content)[0][[0, -1]]

    return fundus_image_bgr[top_row:bottom_row + 1, left_column:right_column + 1]


def resize_fundus_image_to_target_size(fundus_image_bgr, target_size=TARGET_IMAGE_SIZE):
    """Standardise every image to the fixed input resolution the CNN backbone expects.

    Identical to Notebook 2 Cell 7 (cv2.INTER_AREA, chosen because it is the recommended
    interpolation method for shrinking images, which is what almost every raw fundus photo needs).
    """
    return cv2.resize(fundus_image_bgr, target_size, interpolation=cv2.INTER_AREA)


def apply_clahe_contrast_enhancement(
    fundus_image_bgr, clip_limit=CLAHE_CLIP_LIMIT, tile_grid_size=CLAHE_TILE_GRID_SIZE
):
    """Boost local micro-contrast on the L (lightness) channel only.

    Operating in LAB colour space and equalising only the L channel avoids distorting colour
    balance -- colour (e.g. haemorrhage redness) carries diagnostic information that a plain
    grayscale equalisation would blur. Identical to Notebook 2 Cell 5.
    """
    lab_image = cv2.cvtColor(fundus_image_bgr, cv2.COLOR_BGR2LAB)
    lightness_channel, a_channel, b_channel = cv2.split(lab_image)

    clahe_operator = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_lightness_channel = clahe_operator.apply(lightness_channel)

    merged_lab_image = cv2.merge((enhanced_lightness_channel, a_channel, b_channel))
    return cv2.cvtColor(merged_lab_image, cv2.COLOR_LAB2BGR)


def apply_ben_graham_local_average_subtraction(
    fundus_image_bgr, blur_sigma_fraction=BEN_GRAHAM_BLUR_SIGMA_FRACTION
):
    """Subtract a heavily Gaussian-blurred local average from the image.

    This is Ben Graham's method from the winning solution of the original 2015 Kaggle diabetic
    retinopathy competition: it cancels large-scale illumination differences between images taken
    under different lighting/camera conditions, while boosting small local structures such as
    microaneurysms, haemorrhages and vessels that carry diagnostic weight. Identical to
    Notebook 2 Cell 6.
    """
    image_height, image_width = fundus_image_bgr.shape[:2]
    gaussian_blur_sigma = max(image_height, image_width) / blur_sigma_fraction

    locally_averaged_image = cv2.GaussianBlur(fundus_image_bgr, (0, 0), sigmaX=gaussian_blur_sigma)

    # addWeighted(src1, alpha, src2, beta, gamma) = alpha*src1 + beta*src2 + gamma.
    # 4x original minus 4x local average, re-centred at mid-grey (128) -- matches Ben Graham's
    # published formulation and keeps output pixel values in a valid displayable range.
    return cv2.addWeighted(fundus_image_bgr, 4, locally_averaged_image, -4, 128)


def preprocess_fundus_image_for_inference(fundus_image_bgr, return_intermediate_steps=False):
    """Run the full inference-time preprocessing pipeline and return a model-ready image.

    Returns a uint8 array in [0, 255] -- NOT normalised to [0, 1]; see the module docstring for why.
    If return_intermediate_steps is True, also returns a dict of every intermediate image, used by
    the Streamlit UI's optional "show preprocessing steps" panel.
    """
    cropped_image = crop_dark_border_from_fundus_image(fundus_image_bgr)
    resized_image = resize_fundus_image_to_target_size(cropped_image)
    contrast_enhanced_image = apply_clahe_contrast_enhancement(resized_image)
    illumination_corrected_image = apply_ben_graham_local_average_subtraction(contrast_enhanced_image)

    if return_intermediate_steps:
        intermediate_steps_dictionary = {
            "1. raw upload": fundus_image_bgr,
            "2. dark border cropped": cropped_image,
            "3. resized 224x224": resized_image,
            "4. CLAHE contrast enhanced": contrast_enhanced_image,
            "5. Ben Graham illumination corrected": illumination_corrected_image,
        }
        return illumination_corrected_image, intermediate_steps_dictionary

    return illumination_corrected_image
