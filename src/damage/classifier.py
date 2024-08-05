import os
import sys
import logging
import os.path

import torch
import torch.nn as nn
from PIL import Image
from transformers import BeitForImageClassification, AutoImageProcessor


class DamageDetector(nn.Module):
    """
    Damage detection class that initializes the ViT setup and allows for processing all signs in all inspectionWalkthrough media.
    To use the damage detector, call object() (do not use object.forward()). This will return the labels for each sign.

    Future Improvement: Accomodate batching for more efficient inference

    @authors: Benjamin Sanati
    """
    def __init__(self, model_type="simple", initialise=True):
        """
        @brief: Initializes the damage detector for processing. Sets up the classifier once, reducing the total processing time compared to
        setting up on every inference call.

        @authors: Benjamin Sanati
        """
        super(DamageDetector, self).__init__()

        if model_type == "detailed":
            repo_name = r"src/common/finetuned_models/BEiT-fine-finetuned"
        elif model_type == "simple":
            repo_name = r"src/common/finetuned_models/BEiT-coarse-finetuned"
        else:
            raise ValueError("Invalid model type. Choose either 'detailed' or 'simple'.")

        self.device = torch.device("cuda")
        self.model = BeitForImageClassification.from_pretrained(repo_name).to(self.device)
        if initialise:
            self._initialise_model(repo_name)

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _initialise_model(self, repo_name):
        self.image_processor = AutoImageProcessor.from_pretrained(repo_name)

    def forward(self, data_src):
        """
        @brief: Runs damage detection model on each sign in model. Returns labels for each object in each image.
        Args:
            data_src: source of images in local storage folder

        Returns:
            labels - A list of [# images, # objects] object labels

        @authors: Benjamin Sanati
        """
        labels = []
        image_files = os.listdir(data_src)

        for index, (filename) in enumerate(image_files):
            image_path = os.path.join(data_src, filename)
            try:
                image = Image.open(image_path).convert("RGB")

                # Prepare image for the model
                encoding = self.image_processor(image, return_tensors="pt")

                # Forward pass
                with torch.no_grad():
                    encoding["pixel_values"] = encoding["pixel_values"].to(self.device)
                    outputs = self.model(**encoding)
                    logits = outputs.logits

                # Prediction
                predicted_class_idx = logits.argmax(-1).item()
                labels.append(predicted_class_idx)
            except Exception as e:
                self.logger.info(f"Error processing {image_path}: {e}", file=sys.stderr)
                labels.append("error")

        return labels

    def get_class_label(self, class_idx):
        id2label = lambda idx : self.model.config.id2label[idx].lower()
        if type(class_idx) == list:
            return [id2label(idx) for idx in class_idx]
        else:
            return id2label(class_idx)
