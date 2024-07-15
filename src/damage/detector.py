import os
import sys
import os.path
import argparse

import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from transformers import BeitForImageClassification, BeitFeatureExtractor

sys.path.insert(0, r"src/detector")
sys.path.insert(0, r"../detector/yolov7")
sys.path.insert(0, r"src/detector/yolov7")

from models.experimental import attempt_load  # noqa
from utils.datasets import LoadStreams, LoadImages  # noqa
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression  # noqa
from utils.general import apply_classifier, scale_coords, xyxy2xywh  # noqa
from utils.general import strip_optimizer, set_logging, increment_path  # noqa
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel  # noqa


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

        self.device = torch.device("cuda")
        self.feature_extractor = BeitFeatureExtractor.from_pretrained(repo_name)
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

        labels = []
        loop = tqdm(enumerate(os.listdir(data_src)), total=len(os.listdir(data_src)))
        for index, (filename) in loop:
            image_path = os.path.join(data_src, filename)
            image = Image.open(image_path)

            # Prepare image for the model
            encoding = self.feature_extractor(image.convert("RGB"), return_tensors="pt")

            # ############## #
            # GET PREDICTION #
            # ############## #

            # Forward pass
            with torch.no_grad():
                encoding["pixel_values"] = encoding["pixel_values"].to(self.device)
                outputs = self.model(**encoding)
                logits = outputs.logits

            # Prediction
            predicted_class_idx = logits.argmax(-1).item()
            predicted_class = self.model.config.id2label[predicted_class_idx]
            predicted_class = predicted_class.lower()
            labels.append(predicted_class)

            # Update progress bar
            loop.set_description(f"\t\tSign [{index + 1}/{len(os.listdir(data_src))}]")

        return labels


if __name__ == "__main__":
    # Setup argparse config
    parser = argparse.ArgumentParser(description="Processing Configuration")
    parser.add_argument(
        "--data", type=str, help="Data Folder Name.", default="gold_std"
    )
    args = parser.parse_args()
    data_folder = args.data

    # Load the configuration
    os.chdir("../..")

    model = DamageDetector(model_type="detailed")

    # run inference
    model(f"src/common/data/{data_folder}/processed_img")
    print("Inference Complete!", flush=True)
