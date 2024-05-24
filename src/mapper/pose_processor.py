import argparse
import os
import pickle
import sys

import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import generic_filter

sys.path.insert(0, r"../..")

from src.detector.dataset import ImageDataset  # noqa
from src.utils.config import ConfigLoader  # noqa
from src.utils.transformations import VisualisationTransforms  # noqa
from src.utils.visualisation import Visualiser  # noqa


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
        scale_depth=1000,
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
            depth_scale (int): Float, scale factor for depth (e.g., 100.0 if depth is in centimeters and you
                            need meters)
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
            (
                rgb_image_cv,
                depth_image_cv,
            ) = self.visualiser.parse_images(rgb_tensor, depth_tensor)

            # Display the RGB+D images
            if self.display_rgbd:
                self.visualiser.display_imgs(
                    cv2.cvtColor(rgb_image_cv, cv2.COLOR_RGB2BGR),
                    depth_image_cv,
                    frame_index=frame_index,
                )

            # Get pose information for the image
            print(f"Frame {frame_index}:")
            pose_data = self.pose.iloc[frame_index][1:].to_numpy()

            # Get global coordinate of bounding boxes
            frame_global_bboxes = self._3d_processing(
                pose_data, rgb_image_cv, depth_image_cv, bboxes, camera_intrinsics
            )
            global_bboxes[frame_index] = frame_global_bboxes

        return global_bboxes

    def _3d_processing(
        self, pose_data, rgb_image_cv, depth_image_cv, bboxes, camera_intrinsics
    ):
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
            fx,
            fy,
            cx,
            cy,
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

        # Estimate normals and generate mesh
        point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30,
            )
        )
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=5)

        # ######## #
        # NEW CODE #
        # ######## #

        def project_3d_to_2d(points, fx, fy, cx, cy, extrinsics):
            # Transform points from world coordinates to camera coordinates
            points_camera = extrinsics @ np.hstack((points, np.ones((points.shape[0], 1)))).T
            points_camera = points_camera[:3, :].T

            # Project points from 3D to 2D
            points_2d = np.zeros((points_camera.shape[0], 3))
            points_2d[:, 0] = (points_camera[:, 0] * fx) / points_camera[:, 2] + cx
            points_2d[:, 1] = (points_camera[:, 1] * fy) / points_camera[:, 2] + cy
            points_2d[:, 2] = points_camera[:, 2]

            return points_2d

        def fill_depth_image(depth_image, projected_points):
            filled_depth_image = depth_image.copy()
            h, w = depth_image.shape

            i, m = 0, 0
            for point in projected_points:
                x, y, z = int(round(point[0])), int(round(point[1])), point[2]
                if 0 <= x < w and 0 <= y < h and filled_depth_image[y, x] == 0:
                    filled_depth_image[y, x] = z * 1000
                    i += 1

            for y in range(filled_depth_image.shape[0]):
                for x in range(filled_depth_image.shape[1]):
                    print(f"({y}, {x}): {filled_depth_image[y, x]}")
                    if filled_depth_image[y, x] == 0:
                        m += 1
            print(f"Number of filled in points: {i}/{m}")

            return filled_depth_image

        # Extract vertices from the mesh
        np.set_printoptions(threshold=np.inf)
        vertices = np.asarray(mesh.vertices)
        print(f"Vertices: {vertices.shape}")

        # Project 3D vertices to 2D image plane
        projected_points = project_3d_to_2d(vertices, fx, fy, cx, cy, extrinsics)

        # Debugging: Print some projected points
        for i in range(10):
            print(f"3D Point: {vertices[i]}, Projected 2D Point: {projected_points[i]}")

        # Fill the depth image with the depth values from the mesh
        depth_image_cv = fill_depth_image(depth_image_cv, projected_points)
        print(f"Depth image: {depth_image_cv.shape}")

        # ######## #
        # NEW CODE #
        # ######## #

        # Configure the 3D visualizer
        if self.display_3d:
            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window()
            vis.add_geometry(point_cloud)
            vis.add_geometry(mesh)

            for key in range(64, 127):
                vis.register_key_callback(key, lambda vis: vis.close())  # Close on any char key

        # Store global coordinates
        frame_global_bboxes = []

        for bbox in bboxes:
            # Define the 2D bbox in 3D space
            corners = self.transforms.bbox_to_3d(bbox, self.img_size)

            # Scale bbox coordinates from initial image size to depth image width and height
            scaled_corners = self.transforms.scale_bounding_box(
                corners,
                (self.img_size, self.img_size),
                (self.depth_width, self.depth_height),
            )

            # Generate 3D corners with z-values from median over bbox (x, y) range
            corners_3d = [
                self._depth_to_3d(int(x), int(y), depth_image_cv, fx, fy, cx, cy)
                for x, y in scaled_corners
            ]

            # Get global coordinates and apply a depth buffer for visualisation
            global_corners = [
                self._transform_to_global(corner, pose_data) for corner in corners_3d
            ]
            # visualise_corners_3d = self.transforms.create_3d_bounding_box(
            #     global_corners, self.bbox_depth_buffer
            # )
            print(global_corners, flush=True)
            visualise_corners_3d = global_corners
            frame_global_bboxes.append(global_corners)

            # ######## #
            # NEW CODE #
            # ######## #

            # ######## #
            # NEW CODE #
            # ######## #

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
            vis.run()

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

    def get_depth_from_mesh(self, mesh, bbox_corner, camera_position):
        # Calculate the direction vector
        direction = bbox_corner - camera_position
        direction /= np.linalg.norm(direction)

        # Generate t values
        t_values = np.linspace(0, 5, num=500)

        # Project points onto the line (use parametric equation)
        line_points = camera_position + np.outer(t_values, direction)

        return line_points


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

    with open(cfg.pickle_path, "rb") as read_file:
        variables = pickle.load(read_file)

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
        display_rgbd=False,
        display_3d=True,
    )
    global_bboxes_data = pose_processing.get_global_coordinates()

    # Save to pickle file
    data_to_save = {
        "global_bboxes_data": global_bboxes_data,
        "pose_df": pose_df,
    }

    with open(cfg.pickle_path, "wb") as write_file:
        pickle.dump(data_to_save, write_file)
        print("Variables stored to pickle file.", flush=True)
