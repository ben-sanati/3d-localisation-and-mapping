import os
import sys
import cv2
import pickle
import configparser
import numpy as np
import pandas as pd

import open3d as o3d
from scipy.spatial import KDTree
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
        display_rgbd=False,
        scale_depth=1000,
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
            depth_scale (int): Float, scale factor for depth (e.g., 1000.0 if depth is in millimeters and you need meters)
        """
        self.pose = pose
        self.dataset = dataset
        self.bbox_coordinates = bbox_coordinates

        self.img_size = img_size
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.display_rgbd = display_rgbd
        self.scale_depth = scale_depth

    def get_global_coordinates(self):
        for idx, (frame_index, bboxes) in enumerate(self.bbox_coordinates.items()):
            # Acquire images
            rgb_image_pil, rgb_image_cv, depth_image_cv, depth_image_norm_cv, camera_intrinsics = self._parse_images(frame_index)

            # Display the images
            if self.display_rgbd:
                cv2.imshow("RGB Image", rgb_image_cv)
                cv2.imshow("Depth Image", depth_image_norm_cv)
                cv2.waitKey(0)  # Wait for key press to proceed to the next image
                cv2.destroyAllWindows()

            self._3d_processing(rgb_image_pil, depth_image_cv, bboxes, camera_intrinsics)
            break

    def _parse_images(self, frame_index):
        # Acquire images
        rgb_tensor, depth_tensor, camera_intrinsics = self.dataset[frame_index]
        rgb_image_pil = to_pil_image(rgb_tensor)
        depth_image_pil = to_pil_image(depth_tensor)
        rgb_image_cv = cv2.cvtColor(np.array(rgb_image_pil), cv2.COLOR_RGB2BGR)
        depth_image_cv = np.array(depth_image_pil)
        depth_image_norm_cv = cv2.normalize(depth_image_cv, None, 0, 255, cv2.NORM_MINMAX)
        depth_image_norm_cv = np.uint8(depth_image_norm_cv)

        return rgb_image_pil, rgb_image_cv, depth_image_cv, depth_image_norm_cv, camera_intrinsics

    def _3d_processing(self, rgb_image_pil, depth_image_cv, bboxes, camera_intrinsics):
        # Configure the 3D visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        rgb_image_o3d = o3d.geometry.Image(np.array(rgb_image_pil))
        depth_image_o3d = o3d.geometry.Image(np.array(depth_image_cv).astype(np.uint16))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_image_o3d,
            depth_image_o3d,
            depth_scale=self.scale_depth,
            convert_rgb_to_intensity=False
        )

        depth_to_rgb_scale = camera_intrinsics["image_width"] / self.depth_width
        fx = camera_intrinsics["fx"] / depth_to_rgb_scale
        fy = camera_intrinsics["fy"] / depth_to_rgb_scale
        cx = camera_intrinsics["cx"] / depth_to_rgb_scale
        cy = camera_intrinsics["cy"] / depth_to_rgb_scale
        intrinsics = o3d.camera.PinholeCameraIntrinsic(self.depth_width, self.depth_height, fx, fy, cx, cy)
        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            intrinsics
        )
        vis.add_geometry(point_cloud)

        point_cloud_tree = KDTree(np.asarray(point_cloud.points))

        for bbox in bboxes:
            corners = [
                (bbox[0], bbox[1]), (bbox[0], bbox[3]),
                (bbox[2], bbox[3]), (bbox[2], bbox[1])
            ]

            # Scale bbox coordinates from initial image size to depth image width and height
            scaled_corners = self._scale_bounding_box(corners, (self.img_size, self.img_size), (self.depth_width, self.depth_height))

            # Calculate the 3d bbox coordinates based over the interquartile depth values within the bbox
            q1_depth, q3_depth = self._calculate_interquartile_depths(depth_image_cv, scaled_corners)

            # Generate 3D corners with z-values from q1 to q3 depth
            corners_3d_q1 = [self._depth_to_3d(int(x), int(y), q1_depth, fx, fy, cx, cy) for x, y in scaled_corners]
            corners_3d_q3 = [self._depth_to_3d(int(x), int(y), q3_depth, fx, fy, cx, cy) for x, y in scaled_corners]

            # Combine bottom and top corners
            corners_3d_combined = corners_3d_q1 + corners_3d_q3

            # Map to nearest points in the point cloud
            mapped_corners_3d = [self._map_to_nearest_point(corner, point_cloud_tree) for corner in corners_3d_combined]

            print(f"3D Corners: {mapped_corners_3d}", flush=True)

            # Define lines based on corner points for a flat box
            lines = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
                [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
                [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
            ]

            # Create line set for bounding box
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(mapped_corners_3d),
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

    @staticmethod
    def _calculate_interquartile_depths(depth_image, scaled_corners):
        x_min = int(min([c[0] for c in scaled_corners]))
        x_max = int(max([c[0] for c in scaled_corners]))
        y_min = int(min([c[1] for c in scaled_corners]))
        y_max = int(max([c[1] for c in scaled_corners]))

        # Extract the depth values in the bounding box area
        depth_values = depth_image[y_min:y_max+1, x_min:x_max+1].flatten()

        # Filter out zero values
        non_zero_depths = depth_values[depth_values > 0]

        if len(non_zero_depths) == 0:
            return 0, 0

        # Calculate the 1st and 3rd quartile of the non-zero depth values
        q1_depth = np.percentile(non_zero_depths, 25)
        q3_depth = np.percentile(non_zero_depths, 75)

        return q1_depth, q3_depth

    def _depth_to_3d(self, x, y, depth, fx, fy, cx, cy):
        """
        Converts 2D pixel coordinates from the depth image to 3D space coordinates.

        Parameters:
            x (int): X-coordinate in the 2D image.
            y (int): Y-coordinate in the 2D image.
            depth_image (numpy.ndarray): Depth image to convert coordinates from.

        Returns:
            numpy.ndarray: 3D coordinates [X, Y, Z] relative to the camera frame.
        """
        # Extract the depth value at (x, y) (rtabmap uses mm by default)
        Z = depth / self.scale_depth

        # Convert (x, y) coordinates into 3D space based on camera intrinsic parameters
        X = (x - cx) * Z / fx
        Y = (y - cy) * Z / fy

        # Return the 3D point as a numpy array
        return np.array([X, Y, Z])

    @staticmethod
    def _map_to_nearest_point(point, point_cloud_tree):
        dist, idx = point_cloud_tree.query(point)
        nearest_point = point_cloud_tree.data[idx]
        return nearest_point

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
        display_rgbd=True,
    )
    pose_processing.get_global_coordinates()
