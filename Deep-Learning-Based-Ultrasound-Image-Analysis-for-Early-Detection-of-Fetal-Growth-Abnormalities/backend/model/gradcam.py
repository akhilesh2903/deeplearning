"""
Grad-CAM Visualization Module
==============================
Generates attention visualizations for U-Net predictions.
Supports Grad-CAM on the encoder and attention-map overlay from the decoder.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms


class GradCAM:
    """Generates Grad-CAM heatmaps for U-Net / CNN predictions."""

    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer = None
        if hasattr(self.model, "encoder4"):
            target_layer = self.model.encoder4[-1]
        elif hasattr(self.model, "layer4"):
            target_layer = self.model.layer4[-1]
        else:
            for module in reversed(list(self.model.modules())):
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
                    break

        if target_layer is not None:
            target_layer.register_forward_hook(forward_hook)
            target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        target_score = output[0, class_idx]
        target_score.backward()

        if self.gradients is None or self.activations is None:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros_like(activations[0])
        for i, weight in enumerate(weights):
            cam += weight * activations[i]

        cam = F.relu(cam)
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam.cpu().numpy()

    def visualize(self, image_path, output_path=None, heatmap_intensity=0.5, attention_map=None):
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = transform(image).unsqueeze(0).to(self.device)

        if attention_map is not None:
            cam = attention_map
        else:
            cam = self.generate(input_tensor)

        cam = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        visualization = cv2.addWeighted(image_np, 1 - heatmap_intensity, heatmap, heatmap_intensity, 0)

        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

        return visualization


class ClinicalVisualization:
    """Create comprehensive visualizations for clinical reporting."""

    @staticmethod
    def create_diagnostic_figure(image_path, grad_cam_overlay=None, prediction_info=None, save_path=None):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        original_image = Image.open(image_path)

        axes[0].imshow(original_image)
        axes[0].set_title("Original Ultrasound Image", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        if grad_cam_overlay is not None:
            axes[1].imshow(cv2.cvtColor(grad_cam_overlay, cv2.COLOR_BGR2RGB))
            axes[1].set_title("U-Net Attention Map\n(Red = High Model Focus)", fontsize=12, fontweight="bold")
        else:
            axes[1].imshow(original_image)
            axes[1].set_title("No Attention Map Available", fontsize=12, fontweight="bold")
        axes[1].axis("off")

        if prediction_info:
            risk = prediction_info.get("risk_level", prediction_info.get("class_label", "Unknown"))
            conf = prediction_info.get("confidence", 0)
            fig.text(0.5, 0.02, f"Predicted: {risk} | Confidence: {conf:.1f}%", ha="center", fontsize=11, style="italic")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    @staticmethod
    def create_confidence_plot(all_probabilities, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 6))
        classes = list(all_probabilities.keys())
        probs = list(all_probabilities.values())
        colors = ["#2ecc71" if p == max(probs) else "#3498db" for p in probs]
        bars = ax.barh(classes, probs, color=colors, edgecolor="black", linewidth=1.5)
        ax.set_xlabel("Confidence (%)", fontsize=11, fontweight="bold")
        ax.set_title("Risk Classification Confidence", fontsize=13, fontweight="bold")
        ax.set_xlim(0, 100)
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax.text(prob + 2, i, f"{prob:.1f}%", va="center", fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
