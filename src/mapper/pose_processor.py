import os
import sys
import cv2
import pickle
import configparser
import numpy as np
import pandas as pd

import open3d as o3d
from scipy.spatial.transform import Rotation as R
from torchvision.transforms.functional import to_pil_image

sys.path.insert(0, r'../..')

from src.detector.dataset import ImageDataset


class ProcessPose:
    def __init__(
        self,
        pose,
        dataset,
        bbox_coordinates,
        img_size,
        depth_width,
        depth_height,
        depth_to_rgb_scale=3,
    ):
        """
        Initializes the ProcessPose class with the pose data, depth images, and bounding box coordinates.

        Parameters:
            pose (DataFrame): DataFrame containing poses for each frame.
            dataset (torch.Dataset): PyTorch Dataset of images and depth images corresponding to each frame.
            bbox_coordinates (dict): Dictionary of bounding boxes, keyed by frame index.
        """
        self.pose = pose
        self.dataset = dataset
        self.bbox_coordinates = bbox_coordinates

        self.img_size = img_size
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.depth_to_rgb_scale = depth_to_rgb_scale

        # Acquired from rtabmap-databaseViewer and scaled based on depth to RGB size difference (192,256) * 3 = (576, 768)
        self.fx = 673 / self.depth_to_rgb_scale
        self.fy = 673 / self.depth_to_rgb_scale
        self.cx = 357 / self.depth_to_rgb_scale
        self.cy = 483.5 / self.depth_to_rgb_scale

    def get_global_coordinates(self):
        for idx, (frame_index, bboxes) in enumerate(self.bbox_coordinates.items()):
            # Acquire images
            rgb_tensor, depth_tensor, camera_intrinsics = self.dataset[frame_index]
            rgb_image_pil = to_pil_image(rgb_tensor)
            depth_image_pil = to_pil_image(depth_tensor)
            rgb_image_cv = cv2.cvtColor(np.array(rgb_image_pil), cv2.COLOR_RGB2BGR)
            depth_image_cv = np.array(depth_image_pil)
            depth_image_norm_cv = cv2.normalize(depth_image_cv, None, 0, 255, cv2.NORM_MINMAX)
            depth_image_norm_cv = np.uint8(depth_image_norm_cv)

            # Display the images
            cv2.imshow("RGB Image", rgb_image_cv)
            cv2.imshow("Depth Image", depth_image_norm_cv)
            cv2.waitKey(0)  # Wait for key press to proceed to the next image
            cv2.destroyAllWindows()

            # Configure the 3D visualizer
            vis = o3d.visualization.Visualizer()
            vis.create_window()

            rgb_image_o3d = o3d.geometry.Image(np.array(rgb_image_pil))
            depth_image_o3d = o3d.geometry.Image(np.array(depth_image_cv).astype(np.uint16))
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_image_o3d,
                depth_image_o3d,
                depth_scale=1000.0,
                depth_trunc=1.0,
                convert_rgb_to_intensity=False
            )

            intrinsics = o3d.camera.PinholeCameraIntrinsic(self.depth_width, self.depth_height, self.fx, self.fy, self.cx, self.cy)
            point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                intrinsics
            )

            vis.add_geometry(point_cloud)
            print(f"\tPoint cloud: {np.asarray(point_cloud.points)}")

            for bbox in bboxes:
                corners = [
                    (bbox[0], bbox[1]), (bbox[0], bbox[3]),
                    (bbox[2], bbox[3]), (bbox[2], bbox[1])
                ]
                scaled_corners = self._scale_bounding_box(corners, (self.img_size, self.img_size), (self.depth_width, self.depth_height))
                corners_3d = [self._depth_to_3d(int(x), int(y), depth_image_norm_cv, 1000) for x, y in scaled_corners]

                print(f"\tCorners: {corners_3d}")

                # Define lines based on corner points for a flat box
                lines = [
                    [0, 1], [1, 2], [2, 3], [3, 0]  # Just the bottom rectangle
                ]

                # Create line set for bounding box
                line_set = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(corners_3d),
                    lines=o3d.utility.Vector2iVector(lines)
                )

                # Set colors (e.g., red) for each line
                line_set.paint_uniform_color([1, 0, 0])
                vis.add_geometry(line_set)

            # Visualize
            vis.poll_events()
            vis.update_renderer()
            vis.run()
            vis.destroy_window()
            break

    def _depth_to_3d(self, x, y, depth_image, scale_depth):
        """
        Converts 2D pixel coordinates from the depth image to 3D space coordinates.

        Parameters:
            x (int): X-coordinate in the 2D image.
            y (int): Y-coordinate in the 2D image.
            depth_image (numpy.ndarray): Depth image to convert coordinates from.
            depth_scale (int): Float, scale factor for depth (e.g., 1000.0 if depth is in millimeters and you need meters)

        Returns:
            numpy.ndarray: 3D coordinates [X, Y, Z] relative to the camera frame.
        """
        # Extract the depth value at (x, y) (rtabmap uses mm by default)
        Z = depth_image[x, y] / scale_depth

        # Convert (x, y) coordinates into 3D space based on camera intrinsic parameters
        X = (x - self.cx) * Z / self.fx
        Y = (y - self.cy) * Z / self.fy

        # Return the 3D point as a numpy array
        return np.array([X, Y, Z])

    @staticmethod
    def _scale_bounding_box(corners, original_size, new_size):
        """
        Scales bounding box coordinates to a new image size.

        :param corners: List of tuples [(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)]
        :param original_size: Tuple (original_width, original_height)
        :param new_size: Tuple (new_width, new_height)
        :return: List of scaled bounding box coordinates
        """
        original_width, original_height = original_size
        new_width, new_height = new_size

        # Calculate scale factors
        scale_x = new_width / original_width
        scale_y = new_height / original_height

        # Scale each corner
        scaled_corners = [(x * scale_x, y * scale_y) for (x, y) in corners]

        return scaled_corners

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


if __name__ == '__main__':
    # TODO: deal with 3D single image accuracy
    os.chdir(r'../..')
    config_path = r"src/common/configs/variables.cfg"
    config = configparser.ConfigParser()
    config.read(config_path)

    img_size = config.getint('detection', 'img_size')
    depth_width = config.getint('mapping', 'depth_width')
    depth_height = config.getint('mapping', 'depth_height')
    pickle_path = config['paths']['pickle_path']

    save_dir = config['paths']['save_dir']
    image_dir = config['paths']['image_dir']
    depth_image_dir = config['paths']['depth_image_dir']
    calibration_dir = config['paths']['calibration_dir']

    with open(pickle_path, "rb") as file:
        variables = pickle.load(file)

    pose_df = variables["pose_df"]
    predictions = variables["predictions"]

    dataset = ImageDataset(
        image_dir=image_dir,
        depth_image_dir=depth_image_dir,
        calibration_dir=calibration_dir,
        img_size=img_size,
        processing=False
    )
    print(f"Pose: {pose_df}\n\nDepth Images: {len(dataset)}\n\nPredictions: {predictions}")

    pose_processing = ProcessPose(
        pose=pose_df,
        dataset=dataset,
        bbox_coordinates=predictions,
        img_size=img_size,
        depth_width=depth_width,
        depth_height=depth_height,
    )
    pose_processing.get_global_coordinates()
    # global_bboxes_data = pose_processing.get_global_coordinates()
