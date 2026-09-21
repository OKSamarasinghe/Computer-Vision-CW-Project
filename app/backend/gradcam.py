"""
Grad-CAM explanation overlay -- ported directly from Notebook 6 (Model Evaluation & Grad-CAM),
Cell 13, with one addition: describe_heatmap_activation_region(), a quantitative heatmap-localisation
summary that turns the visual overlay into a short, human-readable sentence for the application's
explanation panel. No model weights change anywhere in this module; it only reads gradients of the
already-trained model, exactly as Notebook 6 does.

Notebook 6 also found (and reported honestly, not hidden) that Grad-CAM on this model sometimes
anchors on the optic disc rather than lesion tissue -- a known failure mode of gradient-based
saliency methods on fundus images, not a bug. This application surfaces that same caveat to the user
(see the disclaimer text used alongside describe_heatmap_activation_region()'s output in the UI).
"""

import numpy as np
import tensorflow as tf
import matplotlib.cm as matplotlib_color_maps

from config import LAST_CONVOLUTIONAL_LAYER_NAME


def locate_last_convolutional_layer(keras_model, candidate_layer_name=LAST_CONVOLUTIONAL_LAYER_NAME):
    """Return (owning_model, layer_name) for the last convolutional layer.

    Handles both the case where EfficientNetB0 is nested as a sub-model layer and the case where its
    layers were flattened directly into the outer model -- identical logic to Notebook 6 Cell 13.
    """
    for layer in keras_model.layers:
        if hasattr(layer, "layers"):  # this layer is itself a nested (sub-)model
            for nested_layer in layer.layers:
                if nested_layer.name == candidate_layer_name:
                    return layer, candidate_layer_name
    for layer in keras_model.layers:
        if layer.name == candidate_layer_name:
            return keras_model, candidate_layer_name
    raise ValueError(
        f"Could not find a layer named '{candidate_layer_name}' in the loaded model. "
        "Inspect model.summary() and adjust LAST_CONVOLUTIONAL_LAYER_NAME in config.py."
    )


def build_gradcam_gradient_model(trained_model):
    """Build the auxiliary model whose outputs are [last_conv_activations, class_predictions].

    Reconstructs the exact prediction graph without relying on the outer model's `.input`/`.output`
    (which may be undefined for a Sequential-style model that was never built through the Functional
    API with a real Input tensor) -- the same approach validated in Notebook 6 Cell 13. Called once
    per application session and cached; see frontend/streamlit_app.py.
    """
    owning_submodel, last_conv_layer_name = locate_last_convolutional_layer(trained_model)

    backbone_input_tensor = owning_submodel.input
    backbone_last_conv_output_tensor = owning_submodel.get_layer(last_conv_layer_name).output
    backbone_full_output_tensor = owning_submodel.output

    # Manually re-apply the outer model's remaining head layers (pooling, dropout, dense, dropout,
    # dense) on top of the backbone's full output tensor, symbolically, so the reconstructed graph
    # produces identical predictions to the outer model without ever touching its `.input`.
    head_output_tensor = backbone_full_output_tensor
    for outer_model_layer in trained_model.layers:
        if outer_model_layer is owning_submodel:
            continue
        head_output_tensor = outer_model_layer(head_output_tensor)

    return tf.keras.models.Model(
        inputs=backbone_input_tensor,
        outputs=[backbone_last_conv_output_tensor, head_output_tensor],
    )


def generate_gradcam_heatmap(gradcam_gradient_model, model_input_batch, target_class_index):
    """Run one forward+backward pass and return a normalised (0-1) 2D Grad-CAM heatmap."""
    with tf.GradientTape() as gradient_tape:
        last_conv_layer_output, model_predictions = gradcam_gradient_model(model_input_batch)
        target_class_score = model_predictions[:, target_class_index]

    gradients_of_target_wrt_conv_output = gradient_tape.gradient(target_class_score, last_conv_layer_output)
    pooled_gradients = tf.reduce_mean(gradients_of_target_wrt_conv_output, axis=(0, 1, 2))

    last_conv_layer_output_single = last_conv_layer_output[0]
    weighted_feature_map = last_conv_layer_output_single @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(weighted_feature_map)
    heatmap = tf.maximum(heatmap, 0)  # ReLU: only positive contributions to the predicted class
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam_heatmap_on_image(original_image_uint8, heatmap, overlay_strength=0.4):
    """Resize the heatmap to the original image size, colour-map it, and alpha-blend it over the
    original fundus image for a human-readable overlay."""
    resized_heatmap = tf.image.resize(
        heatmap[..., tf.newaxis], (original_image_uint8.shape[0], original_image_uint8.shape[1])
    )
    resized_heatmap = tf.squeeze(resized_heatmap).numpy()

    colour_mapped_heatmap = matplotlib_color_maps.jet(resized_heatmap)[:, :, :3]
    colour_mapped_heatmap = (colour_mapped_heatmap * 255).astype(np.uint8)

    blended_overlay_image = (
        colour_mapped_heatmap * overlay_strength + original_image_uint8 * (1 - overlay_strength)
    ).astype(np.uint8)
    return blended_overlay_image


def describe_heatmap_activation_region(heatmap):
    """Turn a raw heatmap into a one-line, human-readable location description.

    App-level addition beyond Notebook 6: computes the intensity-weighted centroid of the heatmap
    and reports which quadrant of the image it falls in, plus how concentrated vs. diffuse the
    activation is. This is deliberately simple, transparent, geometry-only post-processing of the
    existing heatmap -- not a new trained component -- so the explanation panel can say something
    more specific than "look at the coloured overlay".
    """
    heatmap_height, heatmap_width = heatmap.shape
    y_coordinates, x_coordinates = np.mgrid[0:heatmap_height, 0:heatmap_width]

    total_activation = heatmap.sum() + 1e-8
    centroid_y = float((y_coordinates * heatmap).sum() / total_activation)
    centroid_x = float((x_coordinates * heatmap).sum() / total_activation)

    vertical_position = "upper" if centroid_y < heatmap_height / 2 else "lower"
    horizontal_position = "left" if centroid_x < heatmap_width / 2 else "right"

    # Fraction of pixels carrying at least 50% of peak activation -- a rough proxy for whether
    # attention is tightly localised (a small lesion) or spread across a large region (diffuse
    # changes, or a confound such as the optic disc dominating a large bright area -- the exact
    # failure mode Notebook 6 documented for this model on some Severe/Proliferative_DR examples).
    concentration_ratio = float((heatmap > 0.5).sum() / heatmap.size)
    concentration_description = "tightly localised" if concentration_ratio < 0.15 else "broadly distributed"

    return {
        "centroid_normalised_position": (centroid_y / heatmap_height, centroid_x / heatmap_width),
        "region_description": f"{vertical_position}-{horizontal_position} region of the image",
        "concentration_ratio": concentration_ratio,
        "concentration_description": concentration_description,
    }
