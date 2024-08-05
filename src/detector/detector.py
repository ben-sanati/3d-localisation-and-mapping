import argparse
import glob
import logging
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from numpy import random
from skimage import transform
from tqdm import tqdm
from ultralytics import YOLOv10

sys.path.insert(0, r"../..")

from src.damage.classifier import DamageDetector  # noqa


class ObjectDetector(nn.Module):
    """
    Object detector class using the YOLOv10 model.
    """

    def __init__(
        self,
        conf_thresh,
        iou_thresh,
        img_size,
        batch_size,
        view_img,
        save_img,
        data_root,
        weights=r"src/common/finetuned_models/yolov10/best.pt",
        temp_damage_path=r"src/common/temp_damage",
    ):
        super(ObjectDetector, self).__init__()

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Configuring Models...")

        # Initialize data and hyperparameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = weights
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.view_img = view_img
        self.img_size = img_size
        self.batch_size = batch_size
        self.save_img = save_img
        self.data_root = data_root
        self.temp_damage_path = temp_damage_path
        self.idx = 0

        # Load YOLOv10 model and damage classification model
        self.model = YOLOv10(weights).to(self.device)
        self.damage_classifier = DamageDetector()

        # Define class names and colors for visualization
        self.names = self.model.names
        self.classes = list(range(len(self.names)))
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        self.logger.info("Models Configured.")

    def forward(self):
        """
        Process the images in the data_root using the YOLOv10 model and return predictions in the expected format.

        Returns:
            predictions (dict[list[list]]): This is a dictionary, where the key is the image index, and the value
            is a list of prediction lists for each bbox identified in the frame.
        """
        predictions = {}
        output_dir = "runs/detect/predict/labels"

        # Run inference
        self.logger.info("Performing Inference...")
        self.model(
            source=self.data_root,
            batch=self.batch_size,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            save_txt=True,
            save_conf=True,
            verbose=False,
        )

        # Ensure all images have a corresponding txt file, create empty txt files if necessary
        image_files = sorted(
            os.listdir(self.data_root), key=lambda x: int(Path(x).stem)
        )
        txt_files = sorted(Path(output_dir).glob("*.txt"), key=lambda x: int(x.stem))
        txt_file_names = {txt_file.stem for txt_file in txt_files}

        for image_file in image_files:
            image_stem = Path(image_file).stem
            if image_stem not in txt_file_names:
                (Path(output_dir) / f"{image_stem}.txt").touch()

        loop = tqdm(enumerate(image_files), total=len(image_files))
        for idx, image_file in loop:
            # Load image and predictions
            img_path = os.path.join(self.data_root, image_file)
            img = cv2.imread(img_path)
            img_height, img_width, _ = img.shape
            txt_file = Path(output_dir) / f"{Path(image_file).stem}.txt"

            # Process images
            preds = self._read_predictions(txt_file, img_width, img_height)
            self._processed_image([img], [preds])

            # Get damage classification
            self._parse_damage(img, preds)

            # Integrate damage classifier
            damage_classification = self.damage_classifier(self.temp_damage_path)

            # Delete images in temp damage folder
            self._delete_all_files_in_directory()

            # Add to dictionary
            for bbox, classification in zip(preds, damage_classification):
                bbox.insert(-2, classification)

            predictions[idx] = preds

        # Clean up the output directory
        shutil.rmtree("runs")

        return predictions

    def _read_predictions(self, txt_file, img_width, img_height):
        """
        Reads the predictions from the file and converts them to the expected format.
        """
        with open(txt_file, "r") as file:
            lines = file.readlines()

        predictions = []
        for line in lines:
            data = line.strip().split()
            label = int(data[0])
            x_center, y_center, w, h, conf = map(float, data[1:])

            # Convert YOLO format (center x, center y, width, height) to (x1, y1, x2, y2)
            x1 = (x_center - w / 2) * img_width
            y1 = (y_center - h / 2) * img_height
            x2 = (x_center + w / 2) * img_width
            y2 = (y_center + h / 2) * img_height

            predictions.append([x1, y1, x2, y2, conf, label])

        return predictions

    def _processed_image(self, data, preds):
        """
        Process and optionally visualize or save the processed images with bounding boxes.
        """
        for idx, (img, pred) in enumerate(zip(data, preds)):
            img = img.copy()
            for bbox in pred:
                x1, y1, x2, y2, conf, label = bbox
                color = self.colors[label]
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(
                    img,
                    self.names[label],
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2,
                )

            if self.view_img:
                cv2.imshow("Image", img)
                cv2.waitKey(0)

            if isinstance(self.save_img, str) and self.save_img:
                save_path = Path(self.save_img) / f"image_{self.idx}.png"
                cv2.imwrite(str(save_path), img)
                self.idx += 1

        if self.view_img:
            cv2.destroyAllWindows()

    def _parse_damage(self, img, pred):
        # Perform Homography on each sign in image
        for bbox_idx, (p) in enumerate(pred):
            bbox_coord = p[:4]
            self._perform_homography(bbox_coord, img, bbox_idx)

    def _perform_homography(self, coord, frame_image, bbox_idx):
        x1, y1 = coord[0], coord[1]
        x2, y2 = coord[2], coord[3]

        # Source points: corners of the box
        src = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]], dtype=np.float32)

        # Destination points: corners of the image
        dst = np.array(
            [
                [0, 0],
                [0, frame_image.shape[0] - 1],
                [frame_image.shape[1] - 1, frame_image.shape[0] - 1],
                [frame_image.shape[1] - 1, 0],
            ],
            dtype=np.float32,
        )

        # Compute the homography matrix
        H, status = cv2.findHomography(src, dst)

        # Apply the homography transformation
        tf_img = cv2.warpPerspective(
            frame_image, H, (frame_image.shape[1], frame_image.shape[0])
        )

        # View image
        if self.view_img:
            cv2.imshow("Sign", tf_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        cv2.imwrite(
            f"{self.temp_damage_path}/{bbox_idx}.png", (tf_img * 255).astype(np.uint8)
        )

    def _delete_all_files_in_directory(self):
        files = glob.glob(os.path.join(self.temp_damage_path, "*"))
        for f in files:
            os.remove(f)


if __name__ == "__main__":
    # Setup logging and argparse for configurations
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="YOLOv10 Object Detection")
    parser.add_argument(
        "--data", type=str, help="Data Folder Name.", default="gold_std"
    )
    args = parser.parse_args()

    os.chdir("../..")

    # Initialize and run the object detector
    detector = ObjectDetector(
        conf_thresh=0.9,
        iou_thresh=0.7,
        img_size=640,
        batch_size=16,
        view_img=True,
        save_img=f"src/common/data/{args.data}/processed_img",
        data_root=f"src/common/data/{args.data}/rtabmap_extract/data_rgb",
    )

    predictions = detector()
    logging.info("Inference complete!")
