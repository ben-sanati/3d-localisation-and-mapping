import argparse
import os
import sys
import pickle

import cv2
import numpy as np
import torch
import torch.nn as nn
from numpy import random
from tqdm import tqdm
from skimage import transform

sys.path.insert(0, r"../..")
sys.path.insert(0, r"src/detector")
sys.path.insert(0, r"../detector/yolov7")
sys.path.insert(0, r"src/detector/yolov7")

from src.utils.config import ConfigLoader  # noqa
from yolov7.models.experimental import attempt_load  # noqa
from yolov7.utils.general import non_max_suppression  # noqa


class ObjectDetector(nn.Module):
    """
    Object detector class that initializes the object detector setup and allows for processing all instances of the
    inspectionWalkthrough object. To use the object detector call object() (do not use object.forward()). This will
    return bbox coordinates and the labels for each identified object.

    @authors: Benjamin Sanati
    """

    def __init__(
        self,
        conf_thresh,
        iou_thresh,
        img_size,
        batch_size,
        view_img,
        save_img,
        weights=r"src/common/finetuned_models/best.pt",
    ):
        """
        @brief: Initializes the object detector for processing. Sets up object detector once, reducing the total
        processing time compared to setting up on every inference call.

                NMS breakdown:
                    1) Discard all the boxes having probabilities less than or equal to a pre-defined threshold
                        (say, 0.5)
                    2) For the remaining boxes:
                        a) Pick the box with the highest probability and take that as the output prediction
                        b) Discard any other box which has IoU greater than the threshold with the output box from
                            step 2
                    3) Repeat step 2 until all the boxes are either taken as the output prediction or discarded
        Args:
            image_size: size of input image (1280 for YOLOv7-e6 model)
            conf_thresh: Minimum confidence requirement from YOLOv7 model output (~0.55 is seen to be the best from the
                        object detector training plots)
            iou_thresh: IoU threshold for NMS
            num_classes: Number of classes that can be defined (number of the types of signs)
            view_img: A bool to view the output of a processed image during processing

        @authors: Benjamin Sanati
        """
        super(ObjectDetector, self).__init__()
        print("\tConfiguring Model...", flush=True)

        sys.stdout = open(os.devnull, "w")  # block printing momentarily

        # Initialize data and hyperparameters (to be made into argparse arguments)
        self.device = torch.device("cuda:0")
        self.weights = weights
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.view_img = view_img
        self.img_size = img_size
        self.batch_size = batch_size
        self.save_img = save_img
        self.idx = 0

        # Preprocess images
        self._initialize_model()
        self._initialize_auxiliary_data()
        print("\tModel Configured.", flush=True)

    def _initialize_model(self):
        """
        Load the YOLOv7 model.
        """
        sys.stdout = open(os.devnull, "w")  # Temporarily suppress printing

        self.model = attempt_load(self.weights, map_location=self.device).half()
        self.stride = int(self.model.stride.max())
        self.names = (
            self.model.module.names
            if hasattr(self.model, "module")
            else self.model.names
        )

        # Warm-up the model
        self.model(
            torch.zeros(1, 3, self.img_size, self.img_size)
            .to(self.device)
            .type_as(next(self.model.parameters()))
        )

        sys.stdout = sys.__stdout__  # Restore printing

    def _initialize_auxiliary_data(self):
        """
        Initialize auxiliary data like color codes and classes.
        """
        self.classes = list(range(len(self.names)))
        self.colours = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

    def forward(self, dataloader):
        """
        Process the images using a DataLoader to handle batch processing.

        Args:
            data_loader: PyTorch DataLoader containing batches of image tensors.
        """
        predictions = {}
        self.model.eval()
        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        with torch.no_grad():
            for idx, (_data, _, _) in loop:
                # Make prediction and save processed images
                data, preds = self._inference(_data)
                self._processed_image(data, preds)

                # Add to dictionary
                preds = [tensor.cpu().tolist() for tensor in preds]
                for img_idx, (pred, img) in enumerate(zip(preds, _data)):
                    predictions[(idx * self.batch_size) + img_idx] = pred
                    self._parse_damage(img, pred)

                # Update progress bar
                loop.set_description(f"Image [{idx + 1}/{len(dataloader)}]")

        return predictions

    def _inference(self, data):
        data = data.half().to(self.device)
        preds = self.model(data)[0]
        preds = non_max_suppression(preds, self.conf_thresh, self.iou_thresh)

        return data, preds

    def _processed_image(self, data, preds):
        data = data.cpu().numpy()
        for idx, (img, pred) in enumerate(zip(data, preds)):
            img, pred = self._parse_content(img, pred)
            for num_signs, (info) in enumerate(pred):
                # add bboxes around objects
                box = info[:4]
                label = int(info[-1])

                # rescale bboxes
                box = self._resize_bbox(box, self.img_size, self.img_size)

                # add bboxes to image
                cv2.rectangle(
                    img,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    tuple(self.colours[label]),
                    10,
                )

                # add filled bboxes with object label above bboxes
                c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                line_thickness = 1.1  # line/font thickness
                tf = max(line_thickness - 1, 1)  # font thickness
                t_size = cv2.getTextSize(
                    self.names[label], 0, fontScale=line_thickness / 3, thickness=tf
                )[0]
                c2 = int(box[0]) + t_size[0], int(box[1]) - t_size[1] - 3
                cv2.rectangle(
                    img, c1, c2, self.colours[label], -1, cv2.LINE_AA
                )  # fill the rectangle with the label
                cv2.putText(
                    img,
                    self.names[label],
                    (c1[0], c1[1] - 2),
                    0,
                    line_thickness / 3,
                    [225, 255, 255],
                    thickness=tf,
                    lineType=cv2.LINE_AA,
                )

            # Convert to RGB from BGR
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.view_img:
                cv2.imshow("Image", rgb_img)
                cv2.waitKey(0)

            # Save image
            cv2.imwrite(f"{self.save_img}/image{self.idx}.png", rgb_img)
            self.idx += 1

        if self.view_img:
            cv2.destroyAllWindows()

    @staticmethod
    def _parse_content(img, pred):
        pred = pred.cpu().tolist()
        img = np.transpose(img, (1, 2, 0))

        if img.max() <= 1.0:
            img *= 255.0

        img = img.astype(np.uint8).copy()
        return img, pred

    def _resize_bbox(self, bbox, image_height, image_width):
        x_scale = image_height / self.img_size
        y_scale = image_width / self.img_size

        bbox[0] *= x_scale
        bbox[1] *= y_scale
        bbox[2] *= x_scale
        bbox[3] *= y_scale
        return bbox

    def _parse_damage(self, img, pred):
        # Parse image
        img = img.squeeze().cpu()
        img = img.permute(1, 2, 0).numpy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for bbox_idx, (p) in enumerate(pred):
            bbox_coord = p[:4]
            sign_img = self._perform_homography(bbox_coord, img)

    def _perform_homography(self, coord, frame_image):
        x1, y1 = coord[0], coord[1]
        x2, y2 = coord[2], coord[3]

        # Source points: corners of the box
        src = np.array(
            [
                [x1, y1],
                [x1, y2],
                [x2, y2],
                [x2, y1]
            ]
        ).reshape((4, 2))

        # Destination points: corners of the image
        dst = np.array(
            [
                [0, 0],
                [0, frame_image.shape[1]],
                [frame_image.shape[0], frame_image.shape[1]],
                [frame_image.shape[0], 0],
            ]
        ).reshape((4, 2))

        # Compute the homography matrix
        tform = transform.estimate_transform('projective', src, dst)

        # Apply the homography transformation
        tf_img = transform.warp(frame_image, tform.inverse)

        # View image
        if self.view_img:
            cv2.imshow("Sign", tf_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return tf_img


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
    config_path = r"src/common/configs/variables.cfg"
    cfg = ConfigLoader(config_path, data_folder)

    # Read the variables file
    with open(cfg.pickle_path, "rb") as file:
        variables = pickle.load(file)

    model = ObjectDetector(
        conf_thresh=0.5,
        iou_thresh=0.65,
        img_size=1280,
        batch_size=2,
        view_img=True,
        save_img=f"src/common/data/{data_folder}/processed_img",
        weights=r"src/common/finetuned_models/best.pt",
    )

    # run inference
    model(variables["dataloader"])
    print("Inference Complete!", flush=True)
