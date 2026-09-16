"""
Inference Pipeline Module
==========================
High-level inference wrapper for the U-Net fetal ultrasound classifier.
"""

from .unet_model import FetalUltrasoundUNet


class InferencePipeline:
    """Runs U-Net inference and formats results for downstream modules."""

    def __init__(self, model_name="unet_resnet18", device="cpu"):
        self.model = FetalUltrasoundUNet(device=device)
        self.device = device

    def run_inference(self, image_path):
        prediction = self.model.predict(image_path)

        if not prediction["success"]:
            return {"status": "error", "message": prediction["error"]}

        return {
            "status": "success",
            "image_path": image_path,
            "cnn_prediction": {
                "predicted_class": prediction["predicted_class"],
                "class_label": prediction["class_label"],
                "risk_level": prediction["risk_level"],
                "confidence_percentage": round(prediction["confidence"], 2),
                "all_probabilities": {
                    k: round(v, 2) for k, v in prediction["all_probabilities"].items()
                },
                "risk_probabilities": {
                    k: round(v, 2) for k, v in prediction["risk_probabilities"].items()
                },
                "condition_description": prediction["condition_description"],
            },
            "attention_map": prediction.get("attention_map"),
            "image_embedding": prediction["embedding"].tolist(),
            "model_info": self.model.get_model_info(),
        }

    def batch_inference(self, image_paths):
        return [self.run_inference(path) for path in image_paths]
