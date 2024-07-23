import os
import sys
import pickle
import os.path
import argparse

import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from transformers import BeitForImageClassification, AutoImageProcessor


class DamageDetector(nn.Module):
    """
    Damage detection class that initializes the ViT setup and allows for processing all signs in all inspectionWalkthrough media.
    To use the damage detector, call object() (do not use object.forward()). This will return the labels for each sign.

    @authors: Benjamin Sanati
    """
    def __init__(self, model_type):
        """
        @brief: Initializes the damage detector for processing. Sets up the classifier once, reducing the total processing time compared to
        setting up on every inference call.

        @authors: Benjamin Sanati
        """
        super(DamageDetector, self).__init__()

        sys.stdout = open(os.devnull, "w")  # Block printing momentarily
        if model_type == "detailed":
            repo_name = r"src/damage/finetuned_models/BEiT-fine-finetuned"
        elif model_type == "simple":
            repo_name = r"src/damage/finetuned_models/BEiT-coarse-finetuned"
        else:
            raise ValueError("Invalid model type. Choose either 'detailed' or 'simple'.")

        self.device = torch.device("cuda")
        self.image_processor = AutoImageProcessor.from_pretrained(repo_name)
        self.model = BeitForImageClassification.from_pretrained(repo_name).to(self.device)

        sys.stdout = sys.__stdout__  # Enable printing

    def forward(self, data_src):
        """
        @brief: Runs damage detection model on each sign in model. Returns labels for each object in each image.
        Args:
            data_src: source of images in local storage folder

        Returns:
            labels - A list of [# images, # objects] object labels

        @authors: Benjamin Sanati
        """
        # TODO: Accomodate batching for more efficient inference
        labels = []
        image_files = os.listdir(data_src)
        total_images = len(image_files)

        loop = tqdm(enumerate(image_files), total=total_images)
        for index, (filename) in loop:
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
                predicted_class = self.model.config.id2label[predicted_class_idx].lower()
                labels.append(predicted_class)
            except Exception as e:
                print(f"Error processing {image_path}: {e}", file=sys.stderr)
                labels.append("error")

            # Update progress bar
            loop.set_description(f"Sign [{index + 1}/{len(os.listdir(data_src))}]")

        return labels
