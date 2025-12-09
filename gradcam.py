"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Implementation
for Alzheimer's Detection Model Visualization

This module generates heatmaps showing which regions of the brain MRI
the CNN model focused on when making its prediction.

Reference:
Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization" (2017)
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2


class GradCAM:
    """
    Generates Grad-CAM heatmaps for a given model and target layer.
    """

    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM.

        Args:
            model: The neural network model
            target_layer: The layer to generate activations from (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        """Hook to save forward pass activations"""
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients"""
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_image, target_class=None):
        """
        Generate Grad-CAM heatmap for an input image.

        Args:
            input_image: Input tensor (1, C, H, W)
            target_class: Target class index (if None, uses predicted class)

        Returns:
            heatmap: Numpy array (H, W) with values 0-1
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_image)

        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        # Generate heatmap
        # Global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        # Apply ReLU (only positive influences)
        cam = F.relu(cam)

        # Normalize to 0-1
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, target_class

    def overlay_heatmap(self, heatmap, original_image, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on original image.

        Args:
            heatmap: Grad-CAM heatmap (H, W)
            original_image: PIL Image or numpy array
            alpha: Transparency of heatmap (0-1)
            colormap: OpenCV colormap

        Returns:
            PIL Image with heatmap overlay
        """
        # Convert original image to numpy
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)

        # Resize heatmap to match original image
        h, w = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized),
            colormap
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Convert grayscale to RGB if needed
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        elif original_image.shape[2] == 1:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

        # Overlay
        overlayed = cv2.addWeighted(
            original_image.astype(np.uint8),
            1 - alpha,
            heatmap_colored,
            alpha,
            0
        )

        return Image.fromarray(overlayed)


def generate_gradcam_visualization(model, input_tensor, original_image, class_names):
    """
    High-level function to generate complete Grad-CAM visualization.

    Args:
        model: AlzheimerDetector model
        input_tensor: Preprocessed input tensor (1, 3, 128, 128)
        original_image: Original PIL Image
        class_names: List of class names

    Returns:
        dict with:
            - heatmap_only: PIL Image of just the heatmap
            - overlayed: PIL Image with heatmap overlayed on original
            - predicted_class: Predicted class name
            - confidence: Prediction confidence (0-1)
            - heatmap_array: Raw numpy heatmap array
    """
    # Get the last convolutional layer (conv_block_2)
    # Model structure: Conv2d, ReLU, Conv2d, ReLU, MaxPool2d
    # Index 2 is the last Conv2d layer (index -2 would be ReLU which doesn't work)
    target_layer = model.conv_block_2[2]  # Last Conv2d layer

    # Create Grad-CAM object
    gradcam = GradCAM(model, target_layer)

    # Generate heatmap
    heatmap, predicted_class_idx = gradcam.generate_heatmap(input_tensor)

    # Get prediction confidence
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output[0], dim=0)
        confidence = probabilities[predicted_class_idx].item()

    # Create visualizations
    # 1. Heatmap only (colorized)
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    heatmap_image = Image.fromarray(heatmap_colored)
    heatmap_image = heatmap_image.resize(original_image.size, Image.BILINEAR)

    # 2. Overlay on original
    overlayed_image = gradcam.overlay_heatmap(heatmap, original_image, alpha=0.5)

    return {
        'heatmap_only': heatmap_image,
        'overlayed': overlayed_image,
        'predicted_class': class_names[predicted_class_idx],
        'confidence': confidence,
        'heatmap_array': heatmap,
        'class_index': predicted_class_idx
    }


def create_comparison_image(original, heatmap, overlayed):
    """
    Create a side-by-side comparison of original, heatmap, and overlay.

    Args:
        original: PIL Image
        heatmap: PIL Image
        overlayed: PIL Image

    Returns:
        PIL Image with all three side by side
    """
    # Resize all to same height
    target_height = 300
    aspect_ratio = original.width / original.height
    target_width = int(target_height * aspect_ratio)

    original_resized = original.resize((target_width, target_height), Image.BILINEAR)
    heatmap_resized = heatmap.resize((target_width, target_height), Image.BILINEAR)
    overlayed_resized = overlayed.resize((target_width, target_height), Image.BILINEAR)

    # Create combined image
    total_width = target_width * 3 + 40  # 20px padding between images
    combined = Image.new('RGB', (total_width, target_height), color=(255, 255, 255))

    combined.paste(original_resized, (0, 0))
    combined.paste(heatmap_resized, (target_width + 20, 0))
    combined.paste(overlayed_resized, (target_width * 2 + 40, 0))

    return combined
