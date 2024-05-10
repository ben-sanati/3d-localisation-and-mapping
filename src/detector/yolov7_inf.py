import os
import sys
import ntpath
from tqdm import tqdm

import cv2
import numpy as np
from numpy import random

import torch
import torch.nn as nn
from torchvision.transforms import transforms

from PIL import Image
from skimage import transform

sys.path.insert(0, r'yolov7')

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import non_max_suppression, scale_coords


class ObjectDetector(nn.Module):
    """
    Object detector class that initializes the object detector setup and allows for processing all instances of the inspectionWalkthrough object.
    To use the object detector call object() (do not use object.forward()). This will return bbox coordinates and the labels for each identified object.

    @authors: Benjamin Sanati
    """

    def __init__(self, conf_thresh, iou_thresh, img_size, view_img, image_folder):
        """
        @brief: Initializes the object detector for processing. Sets up object detector once, reducing the total processing time compared to setting up on every inference call.
                NMS breakdown:
                    1) Discard all the boxes having probabilities less than or equal to a pre-defined threshold (say, 0.5)
                    2) For the remaining boxes:
                        a) Pick the box with the highest probability and take that as the output prediction
                        b) Discard any other box which has IoU greater than the threshold with the output box from the above step
                    3) Repeat step 2 until all the boxes are either taken as the output prediction or discarded
        Args:
            image_size: size of input image (1280 for YOLOv7-e6 model)
            conf_thresh: Minimum confidence requirement from YOLOv7 model output (~0.55 is seen to be the best from the object detector training plots)
            iou_thresh: IoU threshold for NMS
            num_classes: Number of classes that can be defined (number of the types of signs)
            view_img: A bool to view the output of a processed image during processing

        @authors: Benjamin Sanati
        """
        super(ObjectDetector, self).__init__()

        sys.stdout = open(os.devnull, 'w')  # block printing momentarily

        # Initialize data and hyperparameters (to be made into argparse arguments)
        self.device = torch.device('cuda:0')
        self.weights = r"yolov7/finetuned_models/best.pt"
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.view_img = view_img
        self.img_size = img_size
        self.image_folder = image_folder

        # Preprocess images
        print("Preprocessing...", flush=True)
        self._initialize_model()
        self._initialize_auxiliary_data()
        self._preprocess_img()

    def _initialize_model(self):
        """Load the YOLOv7 model."""
        sys.stdout = open(os.devnull, 'w')  # Temporarily suppress printing

        self.model = attempt_load(self.weights, map_location=self.device).half()
        self.stride = int(self.model.stride.max())
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        # Warm-up the model
        self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(self.model.parameters())))

        sys.stdout = sys.__stdout__  # Restore printing

    def _initialize_auxiliary_data(self):
        """Initialize auxiliary data like color codes and classes."""
        self.classes = list(range(len(self.names)))
        self.colours = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

    def _preprocess_img(self):
        """
        Preprocess all images in the image folder for processing.
        """
        for filename in os.listdir(self.image_folder):
            file_path = os.path.join(self.image_folder, filename)

            # Open the image
            with Image.open(file_path) as img:
                # Resize the image
                img = img.resize((1280, 1280), Image.Resampling.LANCZOS)
                img.save(file_path)

    def forward(self):
        """
        @brief: Runs object detection model on each image in inspectionWalkthrough. Uploads processed images to firestore storage. Returns bbox coordinate and labels for each object in each image.
        Args:
            data_src: source of images in local storage folder
            processed_destination: destination of processed image to be saved to local storage folder

        @authors: Benjamin Sanati
        """
        # load images
        dataset = LoadImages(self.image_folder, img_size=self.img_size, stride=self.stride)
        print("Making Predictions...", flush=True)
        loop = tqdm(enumerate(dataset), total=len(dataset))

        for idx, (path, img, im0s, vid_cap) in loop:
            # Infer the image and scale it
            img, pred = self._infer_image(img)
            im0, pred = self._rescale_content(path, pred, img, im0s, dataset)

            # Save the processed image
            self._save_processed_image(im0, pred, path)

            # Update progress bar
            loop.set_description(f"Image [{idx + 1}/{len(dataset)}]")

    def _infer_image(self, img):
        """Run inference on a single image."""
        img = torch.from_numpy(img).half().unsqueeze(0).to(self.device)
        img /= 255.0

        # Inference
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=None)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thresh, self.iou_thresh, classes=self.classes, agnostic=False)[0]
        return img, pred

    def _rescale_content(self, path, pred, img, im0s, dataset):
        # save image with bbox predictions overlay
        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

        # Rescale boxes from img_size to im0 size
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0.shape).round()

        # resize image
        resize_image = transforms.Resize([self.img_size, self.img_size])
        im0 = np.array(resize_image(Image.fromarray(im0)))

        return im0, pred

    def _save_processed_image(self, im0, pred, path):
        for num_signs, (info) in enumerate(pred):
            # add bboxes around objects
            box = info[:4]
            label = int(info[-1])

            # rescale bboxes
            box = self._resize_bbox(box, self.img_size, self.img_size)

            # add bboxes to image
            cv2.rectangle(im0, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), tuple(self.colours[label]), 10)

            # add filled bboxes with object label above bboxes
            c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            line_thickness = 5  # line/font thickness
            tf = max(line_thickness - 1, 1)  # font thickness
            t_size = cv2.getTextSize(self.names[label], 0, fontScale=line_thickness / 3, thickness=tf)[0]
            c2 = int(box[0]) + t_size[0], int(box[1]) - t_size[1] - 3
            cv2.rectangle(im0, c1, c2, self.colours[label], -1, cv2.LINE_AA)  # fill the rectangle with the label
            cv2.putText(im0, self.names[label], (c1[0], c1[1] - 2), 0, line_thickness / 3, [225, 255, 255],
                        thickness=tf, lineType=cv2.LINE_AA)

        # save image
        head, tail = ntpath.split(path)
        data_dst = os.path.join("./out/test_img", tail)
        cv2.imwrite(data_dst, im0)

    def _resize_bbox(self, bbox, image_height, image_width):
        x_scale = (image_height / self.img_size)
        y_scale = (image_width / self.img_size)

        bbox[0] *= x_scale
        bbox[1] *= y_scale
        bbox[2] *= x_scale
        bbox[3] *= y_scale

        return bbox


if __name__ == '__main__':
    print(f"Configuring Model...", flush=True)
    model = ObjectDetector(
        conf_thresh=0.5,
        iou_thresh=0.65,
        img_size=1280,
        view_img=True,
        image_folder="data/test_img"
    )
    print(f"Model Configured!", flush=True)

    # run inference
    model()
    print("Inference Complete!", flush=True)
