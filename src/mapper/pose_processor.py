import os
import sys
import cv2
import pickle
import argparse
import configparser
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import open3d as o3d
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
from torchvision.transforms.functional import to_pil_image

sys.path.insert(0, r'../..')

from src.detector.dataset import ImageDataset

from src.utils.config import ConfigLoader
from src.utils.visualisation import Visualiser
from src.utils.transformations import VisualisationTransforms


class ProcessPose:
    def __init__(
        self,
        pose,
        dataset,
        bbox_coordinates,
        img_size,
        depth_width,
        depth_height,
        display_rgbd=False,
        display_3d=False,
        scale_depth=100,
        bbox_depth_buffer=0.03,
        verbose=False,
    ):
        """
        Initializes the ProcessPose class with pose data, depth images, and bounding box coordinates.

        Parameters:
            pose (DataFrame): DataFrame containing poses for each frame.
            dataset (torch.Dataset): PyTorch Dataset of images and depth images for each frame.
            bbox_coordinates (dict): Dictionary of bounding boxes, keyed by frame index.
            img_size (int): Size of the images.
            depth_width (int): Width of the depth images.
            depth_height (int): Height of the depth images.
            depth_scale (int): Float, scale factor for depth (e.g., 100.0 if depth is in centimeters and you need meters)
        """
        self.pose = pose
        self.dataset = dataset
        self.bbox_coordinates = bbox_coordinates

        self.img_size = img_size
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.display_rgbd = display_rgbd
        self.display_3d = display_3d
        self.scale_depth = scale_depth
        self.bbox_depth_buffer = bbox_depth_buffer
        self.verbose = verbose

        # Instance util classes
        self.visualiser = Visualiser()
        self.transforms = VisualisationTransforms()

    def get_global_coordinates(self):
        global_bboxes = {}
        for frame_index, bboxes in self.bbox_coordinates.items():
            # Acquire images
            rgb_tensor, depth_tensor, camera_intrinsics = self.dataset[frame_index]
            rgb_image_cv, depth_image_cv, depth_image_norm_cv = self.visualiser.parse_images(rgb_tensor, depth_tensor)

            # Display the RGB+D images
            if self.display_rgbd:
                self.visualiser.display_imgs(
                    cv2.cvtColor(rgb_image_cv, cv2.COLOR_RGB2BGR),
                    depth_image_norm_cv
                )

            # Get pose information for the image
            pose_data = self.pose.iloc[frame_index][1:].to_numpy()

            # Get global coordinate of bounding boxes
            frame_global_bboxes = self._3d_processing(pose_data, rgb_image_cv, depth_image_cv, bboxes, camera_intrinsics)
            global_bboxes[frame_index] = frame_global_bboxes
            if frame_index == 10:
                break

        return global_bboxes

    def _3d_processing(self, pose_data, rgb_image_cv, depth_image_cv, bboxes, camera_intrinsics):
        # Get camera intrinsics
        depth_to_rgb_scale = camera_intrinsics["image_width"] / self.depth_width
        fx = camera_intrinsics["fx"] / depth_to_rgb_scale
        fy = camera_intrinsics["fy"] / depth_to_rgb_scale
        cx = camera_intrinsics["cx"] / depth_to_rgb_scale
        cy = camera_intrinsics["cy"] / depth_to_rgb_scale

        # Calculate extrinsics
        extrinsics = self.transforms.get_transformation_matrix(pose_data)
        extrinsics = np.linalg.inv(extrinsics)

        # Define intrinsics
        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            camera_intrinsics["image_width"],
            camera_intrinsics["image_height"],
            fx, fy, cx, cy,
        )

        # Generate RGBD image and point cloud
        rgbd_image = self.visualiser.gen_rgbd(
            rgb_image_cv,
            depth_image_cv,
            self.scale_depth,
        )
        point_cloud = self.visualiser.gen_point_cloud(
            rgbd_image,
            intrinsics,
            extrinsics,
        )

        # Configure the 3D visualizer
        if self.display_3d:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(point_cloud)

        # Store global coordinates
        frame_global_bboxes = []

        for bbox in bboxes:
            # Define the 2D bbox in 3D space
            corners = self.transforms.bbox_to_3d(bbox, self.img_size)

            # Scale bbox coordinates from initial image size to depth image width and height
            scaled_corners = self.transforms.scale_bounding_box(
                corners,
                (self.img_size, self.img_size),
                (self.depth_width, self.depth_height)
            )

            # Generate 3D corners with z-values from median over bbox (x, y) range
            corners_3d = [self._depth_to_3d(int(x), int(y), depth_image_cv, fx, fy, cx, cy) for x, y in scaled_corners]

            # Get global coordinates and apply a depth buffer for visualisation
            global_corners = [self._transform_to_global(corner, pose_data) for corner in corners_3d]
            visualise_corners_3d = self.transforms.create_3d_bounding_box(global_corners, self.bbox_depth_buffer)
            frame_global_bboxes.append(visualise_corners_3d)

            if self.verbose:
                print(f"\tOriginal 2D Corners: {corners}")
                print(f"\tScaled 2D Corners: {scaled_corners}")
                print(f"\t3D Corners before Mapping: {corners_3d}")
                print(f"\tGlobal 3D Coordinates: {global_corners}\n")

            if self.display_3d:
                # Overlay 3D bboxes onto point cloud
                line_set = self.visualiser.overlay_3d_bbox(visualise_corners_3d)
                vis.add_geometry(line_set)

                # Draw camera frustum
                frustum = self.visualiser._get_camera_frustum(
                    self.transforms.get_translation(pose_data),
                    self.transforms.get_rotation(pose_data),
                    fx,
                    fy,
                    self.depth_width,
                    self.depth_height,
                )
                vis.add_geometry(frustum)

        if self.display_3d:
            vis.poll_events()
            vis.update_renderer()
            vis.run()
            vis.destroy_window()

        return frame_global_bboxes

    def _depth_to_3d(self, x, y, depth_image, fx, fy, cx, cy):
        """
        Converts 2D pixel coordinates from the depth image to 3D space coordinates.

        Parameters:
            x (int): X-coordinate in the 2D image.
            y (int): Y-coordinate in the 2D image.
            depth_image (numpy.ndarray): Depth image to convert coordinates from.

        Returns:
            numpy.ndarray: 3D coordinates [X, Y, Z] relative to the camera frame.
        """
        Z = depth_image[y, x] / self.scale_depth

        # Convert (x, y) coordinates into 3D space based on camera intrinsic parameters
        X = (x - cx) * Z / fx
        Y = (y - cy) * Z / fy

        # Return the 3D point as a numpy array
        return np.array([X, Y, Z])

    def _transform_to_global(self, local_point, pose_data):
        """
        Transforms a 3D point from the camera frame to the global coordinate frame using the given pose.

        Parameters:
            local_point (numpy.ndarray): 3D coordinates [X, Y, Z] in the camera frame.
            pose_data (numpy.ndarray): Pose data containing translation and rotation (quaternion).

        Returns:
            numpy.ndarray: Transformed 3D point in the global coordinate frame.
        """
        # Extract the translation and quaternion rotation from the pose
        transformation = self.transforms.get_transformation_matrix(pose_data)

        local_point = np.array([(*local_point, 1)])

        # Apply rotation and translation to obtain the global coordinates
        global_point = (transformation @ local_point.T)[:3, 0]
        return global_point


if __name__ == '__main__':
    # Setup argparse config
    parser = argparse.ArgumentParser(description="Processing Configuration")
    parser.add_argument('--data', type=str, help='Data Folder Name.', default="gold_std")
    args = parser.parse_args()
    data_folder = args.data

    # Load the configuration
    os.chdir("../..")
    config_path = r"src/common/configs/variables.cfg"
    cfg = ConfigLoader(config_path, data_folder)

    with open(cfg.pickle_path, "rb") as file:
        variables = pickle.load(file)

    pose_df = variables["pose_df"]
    predictions = variables["predictions"]

    dataset = ImageDataset(
        image_dir=cfg.image_dir,
        depth_image_dir=cfg.depth_image_dir,
        calibration_dir=cfg.calibration_dir,
        img_size=cfg.img_size,
        processing=False,
    )
    print(f"Pose: {pose_df}\n\nDepth Images: {len(dataset)}")

    pose_processing = ProcessPose(
        pose=pose_df,
        dataset=dataset,
        bbox_coordinates=predictions,
        img_size=cfg.img_size,
        depth_width=cfg.depth_width,
        depth_height=cfg.depth_height,
        display_rgbd=True,
        display_3d=True,
    )
    global_bboxes_data = pose_processing.get_global_coordinates()

    # Save to pickle file
    data_to_save = {
        "global_bboxes_data": global_bboxes_data,
        "pose_df": pose_df,
    }

    try:
        with open(cfg.pickle_path, "wb") as file:
            pickle.dump(data_to_save, file)
            print("Variables stored to pickle file.", flush=True)
    except Exception as e:
        print(f"Failed to write to file: {e}")
