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
        scale_depth=125,
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

    def get_global_coordinates(self):
        global_bboxes = {}
        for frame_index, bboxes in self.bbox_coordinates.items():
            # Acquire images
            rgb_image_pil, rgb_image_cv, depth_image_cv, depth_image_norm_cv, camera_intrinsics = self._parse_images(frame_index)

            # Display the images
            if self.display_rgbd:
                cv2.imshow("RGB Image", rgb_image_cv)
                cv2.imshow("Depth Image", depth_image_norm_cv)
                cv2.waitKey(0)  # Wait for key press to proceed to the next image
                cv2.destroyAllWindows()

            # Get pose information for the image
            pose_data = self.pose.iloc[frame_index][1:].to_numpy()
            print(f"Current Pose Data: {pose_data}")

            frame_global_bboxes = self._3d_processing(pose_data, rgb_image_pil, depth_image_cv, bboxes, camera_intrinsics)
            global_bboxes[frame_index] = frame_global_bboxes
            if frame_index == 10:
                break

        return global_bboxes

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

    def _3d_processing(self, pose_data, rgb_image_pil, depth_image_cv, bboxes, camera_intrinsics):
        # Get camera intrinsics
        depth_to_rgb_scale = camera_intrinsics["image_width"] / self.depth_width
        fx = camera_intrinsics["fx"] / depth_to_rgb_scale
        fy = camera_intrinsics["fy"] / depth_to_rgb_scale
        cx = camera_intrinsics["cx"] / depth_to_rgb_scale
        cy = camera_intrinsics["cy"] / depth_to_rgb_scale

        # Calculate extrinsics
        tx, ty, tz, qx, qy, qz, qw = pose_data
        translation = np.array([tx, ty, tz])
        rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()

        extrinsics = np.eye(4)
        extrinsics[:3, :3] = rotation
        extrinsics[:3, 3] = translation
        extrinsics = np.linalg.inv(extrinsics)

        # Configure the 3D visualizer
        vis = o3d.visualization.Visualizer()
        if self.display_3d:
            vis.create_window()

        frustum = self._get_camera_frustum(translation, rotation, fx, fy, self.depth_width, self.depth_height)
        rgb_image_o3d = o3d.geometry.Image(np.array(rgb_image_pil))
        depth_image_o3d = o3d.geometry.Image(np.array(depth_image_cv).astype(np.uint16))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_image_o3d,
            depth_image_o3d,
            depth_scale=self.scale_depth,
            depth_trunc=10,
            convert_rgb_to_intensity=False,
        )

        intrinsics = o3d.camera.PinholeCameraIntrinsic(720, 960, fx, fy, cx, cy)
        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            intrinsics,
            extrinsics,
        )
        if self.display_3d:
            vis.add_geometry(point_cloud)      

        # The 3D corners are refined by mapping them to the nearest points in the point cloud using the KDTree
        point_cloud_tree = KDTree(np.asarray(point_cloud.points))

        # Store global coordinates
        frame_global_bboxes = []

        for bbox in bboxes:
            # Define limits to bbox
            coordinates = np.array(bbox[:4])
            coordinates[coordinates >= self.img_size] = self.img_size - 0.1

            # Define the 2D bbox in 3D space
            corners = [
                (coordinates[0], coordinates[1]), (coordinates[0], coordinates[3]),
                (coordinates[2], coordinates[3]), (coordinates[2], coordinates[1])
            ]

            # Scale bbox coordinates from initial image size to depth image width and height
            scaled_corners = self._scale_bounding_box(corners, (self.img_size, self.img_size), (self.depth_width, self.depth_height))
            bbox_depth_buffer = 0.03

            # Generate 3D corners with z-values from median over bbox (x, y) range
            corners_3d = [self._depth_to_3d(int(x), int(y), rgbd_image, fx, fy, cx, cy) for x, y in scaled_corners]

            # Define lines based on corner points for a flat box
            lines = [
                [0, 1], [1, 2], [2, 3], [3, 0], # bottom face
                [4, 5], [5, 6], [6, 7], [7, 4], # top face
                [0, 4], [1, 5], [2, 6], [3, 7]  # vertical edges
            ]

            # Get global coordinates and apply a depth buffer for visualisation
            global_corners = [self._transform_to_global(corner, pose_data) for corner in corners_3d]
            visualise_corners_3d = self._create_3d_bounding_box(global_corners, bbox_depth_buffer)
            frame_global_bboxes.append(visualise_corners_3d)

            # Create line set for bounding box
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(visualise_corners_3d),
                lines=o3d.utility.Vector2iVector(lines)
            )

            if self.display_3d:
                # Overlay bboxes
                line_set.paint_uniform_color([1, 0, 0])
                vis.add_geometry(line_set)

                # Overlay pose
                pose_point_cloud = o3d.geometry.PointCloud()
                position = np.vstack((tx, ty, tz)).T
                pose_point_cloud.points = o3d.utility.Vector3dVector(position)
                vis.add_geometry(pose_point_cloud)

                # # Draw camera frustum
                vis.add_geometry(frustum)

            # Debugging prints
            print(f"\tOriginal 2D Corners: {corners}")
            print(f"\tScaled 2D Corners: {scaled_corners}")
            print(f"\t3D Corners before Mapping: {corners_3d}")
            print(f"\tGlobal 3D Coordinates: {global_corners}\n")

        if self.display_3d:
            # Visualize
            vis.poll_events()
            vis.update_renderer()
            vis.run()
            vis.destroy_window()

        return frame_global_bboxes

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

    def _depth_to_3d(self, x, y, rgbd_image, fx, fy, cx, cy):
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
        depth_image = np.asarray(rgbd_image.depth)
        Z = depth_image[y, x]

        # Convert (x, y) coordinates into 3D space based on camera intrinsic parameters
        X = (x - cx) * Z / fx
        Y = (y - cy) * Z / fy

        # Return the 3D point as a numpy array
        return np.array([X, Y, Z])

    @staticmethod
    def _create_3d_bounding_box(global_corners, buffer_depth):
        """
        Create a 3D bounding box from 2D corners with a specified buffer depth.
        
        Parameters:
            global_corners (list of np.array): List of 4 global 3D coordinates forming the 2D bounding box.
            buffer_depth (float): Depth to extend the bounding box along its normal.
        
        Returns:
            o3d.geometry.LineSet: LineSet object representing the 3D bounding box.
        """
        # Ensure we have exactly 4 corners
        assert len(global_corners) == 4, "global_corners should contain exactly 4 points."

        # Convert list of corners to numpy array
        corners = np.array(global_corners)

        # Compute the centroid of the bounding box
        centroid = np.mean(corners, axis=0)

        # Compute two vectors on the plane of the bounding box
        vec1 = corners[1] - corners[0]
        vec2 = corners[3] - corners[0]

        # Compute the normal vector to the plane of the bounding box
        normal = np.cross(vec1, vec2)
        normal = normal / np.linalg.norm(normal)  # Normalize the normal vector

        # Create 8 corners of the 3D bounding box
        box_corners = []
        for corner in corners:
            box_corners.append(corner + buffer_depth * normal)
        for corner in corners:
            box_corners.append(corner - buffer_depth * normal)

        return box_corners

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
        rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()

        transformation = np.eye(4)
        transformation[:3, :3] = rotation
        transformation[:3, 3] = translation

        local_point = np.array([(*local_point, 1)])

        # Apply rotation and translation to obtain the global coordinates
        global_point = (transformation @ local_point.T)[:3, 0]
        return global_point

    def _get_camera_frustum(self, position, rotation, fx, fy, width, height, length=0.3):
        """
        Draws the camera frustum in the 3D visualization.

        Parameters:
            position (np.array): The camera position (tx, ty, tz).
            rotation (np.array): The camera rotation matrix (3x3).
            fx (float): Focal length in x direction.
            fy (float): Focal length in y direction.
            width (int): Image width.
            height (int): Image height.
            length (float): Length of the frustum (default: 0.3).
        """
        # Get directions (in a right-handed coordinate system, the direction of the camera is the 3rd column of R)
        rotation = rotation[0:3, 2]

        # Normalise the direction
        direction = rotation / np.linalg.norm(rotation)
        corners_half = self._get_frustum_corners(position, direction, height, width, length / 2, fx, fy)
        corners_full = self._get_frustum_corners(position, direction, height, width, length, fx, fy)

        frustum_points = np.array([position, *corners_half, *corners_full])

        # # Define frustum lines connecting the camera position to the four corners of the far plane
        frustum_lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],
            [1, 2], [1, 3], [3, 4], [2, 4],
            [0, 5], [0, 6], [0, 7], [0, 8],
            [5, 6], [5, 7], [7, 8], [6, 8],
        ]

        # Create LineSet for the frustum
        frustum_line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(frustum_points),
            lines=o3d.utility.Vector2iVector(frustum_lines)
        )
        frustum_line_set.paint_uniform_color([0, 1, 0])  # Green color for the frustum

        return frustum_line_set

    @staticmethod
    def _get_frustum_corners(position, direction, height, width, length, fx, fy):
        # Calculate the size of the frustum base
        half_width = (width / (2.0 * fx)) * length
        half_height = (height / (2.0 * fy)) * length

        # Choose an arbitrary up vector that is not collinear with the direction
        arbitrary_up = np.array([0, 1, 0]) if abs(direction[1]) < 0.9 else np.array([1, 0, 0])

        # Calculate right and up vectors
        right = np.cross(direction, arbitrary_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, direction)
        up /= np.linalg.norm(up)

        # Calculate frustum base corners
        base_center = position + direction * length
        corner1 = base_center + half_width * right + half_height * up
        corner2 = base_center + half_width * right - half_height * up
        corner3 = base_center - half_width * right + half_height * up
        corner4 = base_center - half_width * right - half_height * up

        return corner1, corner2, corner3, corner4


if __name__ == '__main__':
    # Setup argparse config
    parser = argparse.ArgumentParser(description="Processing Configuration.")
    parser.add_argument('--data', type=str, help='Data Folder Name.', default="gold_std")
    args = parser.parse_args()
    data_folder = args.data

    # Load the configuration
    os.chdir(r'../..')
    config_path = r"src/common/configs/variables.cfg"
    config = configparser.ConfigParser()
    config.read(config_path)

    # Access configuration variables
    img_size = config.getint('detection', 'img_size')
    depth_width = config.getint('mapping', 'depth_width')
    depth_height = config.getint('mapping', 'depth_height')

    # Access paths from the 'paths' section
    root_dir = config['paths']['root_dir']
    data_path = os.path.join(root_dir, data_folder)
    db_path = os.path.join(data_path, config['paths']['db_path'])
    ply_path = os.path.join(data_path, config['paths']['ply_path'])
    pose_path = os.path.join(data_path, config['paths']['pose_path'])
    pickle_path = os.path.join(data_path, config['paths']['pickle_path'])
    image_dir = os.path.join(data_path, config['paths']['image_dir'])
    depth_image_dir = os.path.join(data_path, config['paths']['depth_image_dir'])
    calibration_dir = os.path.join(data_path, config['paths']['calibration_dir'])

    with open(pickle_path, "rb") as file:
        variables = pickle.load(file)

    pose_df = variables["pose_df"]
    predictions = variables["predictions"]

    dataset = ImageDataset(
        image_dir=image_dir,
        depth_image_dir=depth_image_dir,
        calibration_dir=calibration_dir,
        img_size=img_size,
        processing=False,
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
        display_3d=True,
    )
    pose_processing.get_global_coordinates()
