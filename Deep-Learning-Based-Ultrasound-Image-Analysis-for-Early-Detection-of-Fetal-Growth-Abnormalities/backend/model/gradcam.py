"""
Grad-CAM Visualization Module
==============================
Generates attention visualizations for U-Net predictions.
Supports Grad-CAM on the encoder and attention-map overlay from the decoder.
"""

import os
# Prevent OpenMP runtime conflict crashes on Windows when torch + cv2 are imported
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from torchvision import transforms  # type: ignore


class GradCAM:
    """Generates Grad-CAM heatmaps for U-Net / CNN predictions."""

    def __init__(self, model, device="cpu"):
        # Unwrap model wrapper if FetalUltrasoundUNet or similar wrapper is passed
        if hasattr(model, "model") and isinstance(model.model, nn.Module):
            self.model = model.model
        else:
            self.model = model

        self.device = torch.device(device) if isinstance(device, str) else device
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            if grad_output and grad_output[0] is not None:
                self.gradients = grad_output[0].detach()

        target_layer = self._find_target_layer()

        if target_layer is not None:
            self.hook_handles.append(target_layer.register_forward_hook(forward_hook))
            self.hook_handles.append(target_layer.register_full_backward_hook(backward_hook))
        else:
            print("[GradCAM] Warning: No suitable target layer found for Grad-CAM hooks.")

    def _find_target_layer(self):
        """Find the most suitable convolutional / bottleneck layer for Grad-CAM."""
        if hasattr(self.model, "encoder4"):
            encoder4 = self.model.encoder4
            # Look for the last Conv2d inside encoder4
            for mod in reversed(list(encoder4.modules())):
                if isinstance(mod, nn.Conv2d):
                    return mod
            return encoder4[-1] if hasattr(encoder4, "__getitem__") else encoder4

        if hasattr(self.model, "layer4"):
            layer4 = self.model.layer4
            for mod in reversed(list(layer4.modules())):
                if isinstance(mod, nn.Conv2d):
                    return mod
            return layer4[-1] if hasattr(layer4, "__getitem__") else layer4

        if hasattr(self.model, "bottleneck"):
            for mod in reversed(list(self.model.bottleneck.modules())):
                if isinstance(mod, nn.Conv2d):
                    return mod

        # Fallback: scan all submodules in reverse for the last Conv2d
        for mod in reversed(list(self.model.modules())):
            if isinstance(mod, nn.Conv2d):
                return mod

        return None

    def remove_hooks(self):
        """Clean up registered PyTorch hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def __del__(self):
        self.remove_hooks()

    def generate(self, input_tensor, class_idx=None, target_size=None):
        """
        Generate Grad-CAM heatmap for the given input tensor.

        Args:
            input_tensor (torch.Tensor): 4D tensor (1, C, H, W) or 3D tensor (C, H, W).
            class_idx (int, optional): Target class index for Grad-CAM.
            target_size (tuple, optional): (width, height) to resize output CAM.

        Returns:
            np.ndarray: 2D normalized heatmap of shape (H, W) in range [0, 1].
        """
        self.model.eval()

        # Ensure execution within autograd-enabled context (works even if caller is in torch.no_grad())
        with torch.enable_grad():
            self.gradients = None
            self.activations = None

            if not isinstance(input_tensor, torch.Tensor):
                raise TypeError(f"input_tensor must be a torch.Tensor, got {type(input_tensor)}")

            if input_tensor.ndim == 3:
                input_tensor = input_tensor.unsqueeze(0)

            input_tensor = input_tensor.to(self.device).requires_grad_(True)
            self.model.zero_grad()

            output = self.model(input_tensor)
            if isinstance(output, tuple):
                output = output[0]

            if class_idx is None:
                class_idx = int(torch.argmax(output, dim=1).item())

            target_score = output[0, class_idx]
            target_score.backward(retain_graph=True)

            h = input_tensor.shape[2] if input_tensor.ndim >= 4 else 224
            w = input_tensor.shape[3] if input_tensor.ndim >= 4 else 224

            if self.gradients is None or self.activations is None:
                cam = np.zeros((h, w), dtype=np.float32)
                if target_size:
                    cam = cv2.resize(cam, target_size, interpolation=cv2.INTER_LINEAR)
                return cam

            gradients = self.gradients[0]      # shape: (C, H_f, W_f)
            activations = self.activations[0]  # shape: (C, H_f, W_f)

            # Global average pooling of gradients
            weights = gradients.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)

            # Weighted sum of activations
            cam_tensor = (weights * activations).sum(dim=0)  # (H_f, W_f)
            cam_tensor = F.relu(cam_tensor)

            cam_min = float(cam_tensor.min().item())
            cam_max = float(cam_tensor.max().item())
            if cam_max > cam_min:
                cam_tensor = (cam_tensor - cam_min) / (cam_max - cam_min)
            else:
                cam_tensor = torch.zeros_like(cam_tensor)

            cam = cam_tensor.detach().cpu().numpy().astype(np.float32)

            if target_size:
                cam = cv2.resize(cam, target_size, interpolation=cv2.INTER_LINEAR)

            return cam

    def visualize(self, image_input, output_path=None, heatmap_intensity=0.4, attention_map=None):
        """
        Blend Grad-CAM heatmap or U-Net attention map onto original image.

        Args:
            image_input (str, Path, PIL.Image, or np.ndarray): Input image or path.
            output_path (str or Path, optional): Path to save overlay image.
            heatmap_intensity (float): Blending weight for heatmap [0.0 - 1.0].
            attention_map (np.ndarray, optional): Precomputed 2D attention map.

        Returns:
            np.ndarray: RGB overlay image (H, W, 3) as uint8.
        """
        # Load and convert image to RGB numpy array
        if isinstance(image_input, (str, Path)):
            image_pil = Image.open(str(image_input)).convert("RGB")
            image_np = np.array(image_pil)
        elif isinstance(image_input, Image.Image):
            image_pil = image_input.convert("RGB")
            image_np = np.array(image_pil)
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                image_np = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.ndim == 3 and image_input.shape[2] == 3:
                image_np = image_input.copy()
            elif image_input.ndim == 3 and image_input.shape[2] == 4:
                image_np = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            else:
                raise ValueError(f"Unsupported image array shape: {image_input.shape}")
            image_pil = Image.fromarray(image_np)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        orig_h, orig_w = image_np.shape[:2]

        if attention_map is not None:
            cam = np.asarray(attention_map, dtype=np.float32)
            cam = np.squeeze(cam)
            if cam.ndim != 2:
                cam = np.zeros((orig_h, orig_w), dtype=np.float32)
            else:
                c_min = float(np.min(cam))
                c_max = float(np.max(cam))
                if c_max > c_min:
                    cam = (cam - c_min) / (c_max - c_min)
                else:
                    cam = np.zeros_like(cam, dtype=np.float32)
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = transform(image_pil).unsqueeze(0).to(self.device)
            cam = self.generate(input_tensor)

        # Resize CAM to match original image dimensions
        cam_resized = cv2.resize(cam, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        cam_resized = np.clip(cam_resized, 0.0, 1.0)

        # Generate JET heatmap and convert OpenCV BGR to RGB
        heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Blend original RGB image with RGB heatmap
        heatmap_intensity = float(np.clip(heatmap_intensity, 0.0, 1.0))
        visualization_rgb = cv2.addWeighted(
            image_np,
            1.0 - heatmap_intensity,
            heatmap_rgb,
            heatmap_intensity,
            0
        )

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # cv2.imwrite expects BGR order
            cv2.imwrite(str(output_path), cv2.cvtColor(visualization_rgb, cv2.COLOR_RGB2BGR))

        return visualization_rgb


class ClinicalVisualization:
    """Create comprehensive visualizations for clinical reporting."""

    @staticmethod
    def _to_rgb_image(image_input):
        """Helper to safely convert various image inputs into an RGB PIL Image or ndarray."""
        if image_input is None:
            return None
        if isinstance(image_input, (str, Path)):
            return Image.open(str(image_input)).convert("RGB")
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        if isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            if image_input.ndim == 3 and image_input.shape[2] == 4:
                return cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            return image_input
        return image_input

    @staticmethod
    def create_diagnostic_figure(image_path, grad_cam_overlay=None, prediction_info=None, save_path=None):
        """
        Create side-by-side diagnostic figure (original vs attention overlay).

        Args:
            image_path: Path or image array of original ultrasound.
            grad_cam_overlay: Path or RGB array of attention map overlay.
            prediction_info (dict, optional): Diagnostic prediction details.
            save_path (str or Path, optional): File path to save figure.

        Returns:
            matplotlib.figure.Figure: Generated figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        original_image = ClinicalVisualization._to_rgb_image(image_path)
        axes[0].imshow(original_image)
        axes[0].set_title("Original Ultrasound Image", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        if grad_cam_overlay is not None:
            overlay = ClinicalVisualization._to_rgb_image(grad_cam_overlay)
            axes[1].imshow(overlay)
            axes[1].set_title("U-Net Attention Map\n(Red = High Model Focus)", fontsize=12, fontweight="bold")
        else:
            axes[1].imshow(original_image)
            axes[1].set_title("No Attention Map Available", fontsize=12, fontweight="bold")
        axes[1].axis("off")

        if prediction_info:
            risk = prediction_info.get("risk_level", prediction_info.get("class_label", "Unknown"))
            conf = float(prediction_info.get("confidence", prediction_info.get("confidence_percentage", 0)))
            fig.text(
                0.5,
                0.02,
                f"Predicted: {risk} | Confidence: {conf:.1f}%",
                ha="center",
                fontsize=11,
                style="italic",
            )

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(save_path), dpi=150, bbox_inches="tight")

        return fig

    @staticmethod
    def create_confidence_plot(all_probabilities, save_path=None):
        """
        Generate bar plot showing confidence breakdown across diagnostic categories.

        Args:
            all_probabilities (dict): Dictionary mapping class names to confidence scores.
            save_path (str or Path, optional): File path to save plot.

        Returns:
            matplotlib.figure.Figure: Generated bar plot figure.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        if not all_probabilities:
            all_probabilities = {"Normal": 0.0, "FGR": 0.0, "Abnormal": 0.0}

        classes = list(all_probabilities.keys())
        raw_probs = [float(all_probabilities[k]) for k in classes]

        # Convert probabilities in [0, 1] range to percentages if needed
        if raw_probs and max(raw_probs) <= 1.0 and sum(raw_probs) <= 1.5:
            probs = [p * 100.0 for p in raw_probs]
        else:
            probs = raw_probs

        max_val = max(probs) if probs else 0.0
        colors = ["#2ecc71" if p == max_val and p > 0 else "#3498db" for p in probs]
        bars = ax.barh(classes, probs, color=colors, edgecolor="black", linewidth=1.5)

        ax.set_xlabel("Confidence (%)", fontsize=11, fontweight="bold")
        ax.set_title("Risk Classification Confidence", fontsize=13, fontweight="bold")
        ax.set_xlim(0, 105)

        for bar, prob in zip(bars, probs):
            ax.text(prob + 2, bar.get_y() + bar.get_height() / 2, f"{prob:.1f}%", va="center", fontweight="bold")

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(save_path), dpi=150, bbox_inches="tight")

        return fig
