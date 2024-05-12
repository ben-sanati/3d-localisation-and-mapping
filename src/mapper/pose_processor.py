import os
import sys
import pickle
import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, r'../..')

from src.detector.dataset import ImageDataset


class ProcessPose:
    def __init__(
        self,
        pose,
        dataset,
        bbox_coordinates,
        img_size
    ):
        """
        Initializes the ProcessPose class with the pose data, depth images, and bounding box coordinates.

        Parameters:
            pose (DataFrame): DataFrame containing poses for each frame.
            dataset (torch.Dataset): PyTorch Dataset of images and depth images corresponding to each frame.
            bbox_coordinates (dict): Dictionary of bounding boxes, keyed by frame index.
        """
        self.img_size = img_size
        self.pose = pose
        self.dataset = dataset
        self.bbox_coordinates = bbox_coordinates

        # Acquired from rtabmap-databaseViewer
        self.fx = 673
        self.fy = 673
        self.cx = 357
        self.cy = 483.5

    def get_global_coordinates(self):
        """
        Converts all bounding box coordinates from local camera frames to global coordinates.

        Returns:
            dict: Dictionary mapping each frame index to lists of global bounding box coordinates.
        """
        global_bbox_lines = {}

        # Iterate through each frame index and corresponding bounding boxes
        for frame_index, bboxes in self.bbox_coordinates.items():
            # Extract the corresponding pose for the current frame
            frame_pose = self.pose.loc[frame_index, ['tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']].values
            depth_image = self.dataset[frame_index][1]
            print(f"Depth Image: {depth_image.size()}", flush=True)

            # Store the global bounding boxes for the current frame
            frame_global_bboxes = []

            # Convert each bounding box's corner points to global coordinates
            for bbox in bboxes:
                x_min, y_min, x_max, y_max, confidence, class_id = bbox

                # Define the corner points of the bounding box
                corners = [
                    (x_min, y_min), (x_min, y_max), 
                    (x_max, y_max), (x_max, y_min)
                ]

                # Convert each corner from 2D to 3D coordinates in the camera frame
                corners_3d = [self._depth_to_3d(int(x), int(y), depth_image) for x, y in corners]

                # Transform the 3D camera frame coordinates to global coordinates
                global_corners = [self._transform_to_global(corner, frame_pose) for corner in corners_3d]
                frame_global_bboxes.append(global_corners)

            # Map the frame index to the list of global bounding boxes
            global_bbox_lines[frame_index] = frame_global_bboxes

        return global_bbox_lines

    def _depth_to_3d(self, x, y, depth_image):
        """
        Converts 2D pixel coordinates from the depth image to 3D space coordinates.

        Parameters:
            x (int): X-coordinate in the 2D image.
            y (int): Y-coordinate in the 2D image.
            depth_image (numpy.ndarray): Depth image to convert coordinates from.

        Returns:
            numpy.ndarray: 3D coordinates [X, Y, Z] relative to the camera frame.
        """
        # Extract the depth value at (x, y)
        Z = depth_image[-1, x, y]

        # Convert (x, y) coordinates into 3D space based on camera intrinsic parameters
        X = (x - self.cx) * Z / self.fx
        Y = (y - self.cy) * Z / self.fy

        # Return the 3D point as a numpy array
        return np.array([X, Y, Z])

    def _transform_to_global(self, local_point, pose):
        """
        Transforms a 3D point from the camera frame to the global coordinate frame using the given pose.

        Parameters:
            local_point (numpy.ndarray): 3D coordinates [X, Y, Z] in the camera frame.
            pose (numpy.ndarray): Pose data containing translation and rotation (quaternion).

        Returns:
            numpy.ndarray: Transformed 3D point in the global coordinate frame.
        """
        # Extract the translation and quaternion rotation from the pose
        tx, ty, tz, qx, qy, qz, qw = pose
        translation = np.array([tx, ty, tz])
        rotation = R.from_quat([qx, qy, qz, qw])

        # Apply rotation and translation to obtain the global coordinates
        global_point = rotation.apply(local_point) + translation
        return global_point

    def view_3d_bbox(self):
        for frame_index, bboxes in self.bbox_coordinates.items():
            rgb_tensor, depth_tensor = self.dataset[frame_index]
            print(f"RGB: {rgb_tensor.size()}\nDepth: {depth_tensor.size()}")
            rgb_numpy = rgb_tensor.numpy().transpose(1, 2, 0)
            depth_numpy = depth_tensor.numpy().transpose(1, 2, 0)
            if np.max(rgb_numpy) <= 1.0:
                rgb_numpy = (rgb_numpy * 255).astype(np.uint8)
            else:
                rgb_numpy = rgb_numpy.astype(np.uint8)

            # Create Open3D Image objects
            rgb_image = o3d.geometry.Image(rgb_numpy.copy())
            depth_image = o3d.geometry.Image(depth_numpy.astype(np.uint16).copy())

            # Create an RGBD image
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_image,
                depth_image,
                depth_scale=1000.0,  # Adjust according to your depth unit
                depth_trunc=10000.0,  # Adjust truncation depth according to your needs
                convert_rgb_to_intensity=False
            )

            # Create point cloud
            point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                o3d.camera.PinholeCameraIntrinsic(width=self.img_size, height=self.img_size, fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy)
            )

            # Visualize the point cloud
            o3d.visualization.draw_geometries([point_cloud])


if __name__ == '__main__':
    os.chdir(r'../..')

    img_size = 640
    save_dir = r"src/common/data/gold_std/raw_img"
    image_dir = f"{save_dir}/images"
    depth_image_dir = f"{save_dir}/depth_images"
    print(f"{os.getcwd()}", flush=True)

    with open(r"src/common/data/gold_std/variables.pkl", "rb") as file:
        variables = pickle.load(file)

    pose_df = variables["pose_df"]
    predictions = variables["predictions"]
    dataset = ImageDataset(image_dir=image_dir, depth_image_dir=depth_image_dir, img_size=img_size)
    print(f"Pose: {pose_df}\n\nDepth Images: {len(dataset)}\n\nPredictions: {predictions}")

    pose_processing = ProcessPose(
        pose=pose_df,
        dataset=dataset,
        bbox_coordinates=predictions,
        img_size=img_size,
    )
    pose_processing.view_3d_bbox()
    # global_bboxes_data = pose_processing.get_global_coordinates()