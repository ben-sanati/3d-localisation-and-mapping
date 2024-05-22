import argparse
import os
import pickle
import sys

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

sys.path.insert(0, r"../..")
sys.path.append("/home/phoenix/base/active/3D-Mapping-ATK")

from src.utils.config import ConfigLoader  # noqa
from src.utils.transformations import VisualisationTransforms  # noqa
from src.utils.visualisation import Visualiser  # noqa


class Mapping:
    def __init__(
        self,
        global_bboxes_data,
        pose,
        eps=0.04,
        min_points=10,
        ply_filepath=r"../common/data/gold_std/cloud.ply",
        preprocess_point_cloud=True,
        overlay_pose=False,
        radius=0.1,
        max_nn=30,
        depth=8,
        scale_factor=1.0,
        bbox_depth_buffer=0.02,  # 3cm
        area_bbox_min_th=0.001,
        cam_to_bbox_min_th=0.01,
    ):
        self.eps = eps
        self.min_points = min_points
        self.ply_filepath = ply_filepath
        self.overlay_pose = overlay_pose
        self.global_bboxes_data = global_bboxes_data
        self.preprocess_point_cloud = preprocess_point_cloud

        # Mesh data
        self.radius = radius
        self.max_nn = max_nn
        self.depth = depth
        self.scale_factor = scale_factor
        self.bbox_depth_buffer = bbox_depth_buffer

        # Bbox threshold values
        self.area_bbox_min_th = area_bbox_min_th
        self.cam_to_bbox_min_th = cam_to_bbox_min_th

        # Remove timestamp column
        self.pose = pose.drop(["timestamp"], axis=1)

        # Load the point cloud
        self.pcd = o3d.io.read_point_cloud(
            self.ply_filepath,
            remove_nan_points=True,
            remove_infinite_points=True,
            print_progress=True,
        )

        # Instance util classes
        self.visualiser = Visualiser()
        self.transforms = VisualisationTransforms()

    def make_point_cloud(self):
        if self.preprocess_point_cloud:
            # DBSCAN clustering
            self._clustering()

        # Visualise mesh
        self._visualiser(self.pcd)

    def make_mesh(self):
        if self.preprocess_point_cloud:
            # DBSCAN clustering
            self._clustering()

        # Create mesh
        print("\tMaking mesh...")
        mesh = self._poisson_surface_recon()

        # Optimise mesh
        print("\tOptimising mesh...")
        mesh = self._optimise_mesh(mesh)

        # Visualise mesh
        print("\tVisualising mesh...")
        self._visualiser(mesh)

    def _clustering(self):
        # Execute DBSCAN algorithm
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
            labels = np.array(
                self.pcd.cluster_dbscan(
                    eps=self.eps, min_points=self.min_points, print_progress=True
                )
            )

        # Calculate the number of point clouds
        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters", flush=True)

        # Find the largest cluster
        largest_cluster_label = np.argmax(np.bincount(labels[labels >= 0]))

        # Filter the point cloud to only include points from the largest cluster
        largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
        self.pcd = self.pcd.select_by_index(largest_cluster_indices)

    def _poisson_surface_recon(self):
        # Estimate normals
        self.pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.radius, max_nn=self.max_nn
            )
        )

        # Apply Poisson surface reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            self.pcd, depth=self.depth, scale=self.scale_factor
        )

        return mesh

    def _optimise_mesh(self, mesh):
        # Smooth mesh -> less noise in mesh (using Laplacian filter)
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=1)

        # Mesh denoising -> further cleans surface by removing noise
        mesh = mesh.filter_smooth_simple(number_of_iterations=2)

        # Mesh refinement -> increases resolution and improves smoothness
        mesh = mesh.subdivide_midpoint(number_of_iterations=1)

        # Scale the mesh
        mesh.scale(self.scale_factor, center=(0, 0, 0))

        return mesh

    def _visualiser(self, data):
        # Create a visualizer object
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Make mesh/point_cloud
        vis.add_geometry(data)

        # Overlay 3D bboxes onto point cloud
        point_cloud_points = np.asarray(self.pcd.points)
        kd_tree = KDTree(point_cloud_points)
        for frame_index, bbox_list in self.global_bboxes_data.items():
            pose_data = self.pose.iloc[frame_index]
            camera_position = np.array(pose_data[:3])
            for bbox in bbox_list:
                bbox_area = self.transforms.calculate_bbox_area(bbox)

                if self._is_within_threshold(bbox, camera_position, self.cam_to_bbox_min_th) or bbox_area < self.area_bbox_min_th:
                    # Reason for removal
                    if self._is_within_threshold(bbox, camera_position, self.cam_to_bbox_min_th):
                        print("\t\tBBox removed. At least one point is within the threshold distance from the camera position.")
                    elif bbox_area < self.area_bbox_min_th:
                        print("\t\tBBox removed. BBox area too small.")
                    continue

                # Map corner to point cloud from camera pose
                transformed_bbox = [
                    self.transforms.closest_point_to_corner(camera_position, corner, kd_tree, point_cloud_points)
                    for corner in bbox
                ]

                # Turn 2D corners into 3D corners (with a buffer)
                bbox_3d = self.transforms.create_3d_bounding_box(
                    transformed_bbox, self.bbox_depth_buffer
                )
                bbox_lines = self.visualiser.overlay_3d_bbox(bbox_3d)
                vis.add_geometry(bbox_lines)

        if self.overlay_pose:
            # Overlay pose onto 3d map
            pose_point_cloud = self.visualiser.overlay_pose(self.pose)
            vis.add_geometry(pose_point_cloud)

            # # Get directions
            directions = self.transforms.get_camera_direction(self.pose)

            # Get pose camera directions
            pose_directions = self.visualiser.overlay_pose_directions(
                pose_point_cloud.points, directions
            )
            vis.add_geometry(pose_directions)

        # Run the visualizer
        vis.run()
        vis.destroy_window()

    @staticmethod
    def _is_within_threshold(points, camera_position, threshold):
        """
        Check if any point in the list is within the threshold distance from the camera position.
        """
        return any(np.linalg.norm(point - camera_position) < threshold for point in points)


if __name__ == "__main__":
    # Setup argparse config
    parser = argparse.ArgumentParser(description="Processing Configuration.")
    parser.add_argument(
        "--data", type=str, help="Data Folder Name.", default="gold_std"
    )
    parser.add_argument(
        "--model", type=str, help="The Type of 3D Model to Create [mesh or pc]", default="mesh"
    )
    args = parser.parse_args()
    data_folder = args.data
    model_type = args.model

    # Load the configuration
    os.chdir("../..")
    config_path = r"src/common/configs/variables.cfg"
    cfg = ConfigLoader(config_path, data_folder)

    eps = 0.02
    min_points = 10

    # Read the variables file
    with open(cfg.pickle_path, "rb") as file:
        variables = pickle.load(file)

    global_bboxes_data = variables["global_bboxes_data"]
    pose_df = variables["pose_df"]

    # Create the map
    mapper = Mapping(
        global_bboxes_data=global_bboxes_data,
        pose=pose_df,
        eps=eps,
        min_points=min_points,
        ply_filepath=cfg.ply_path,
        preprocess_point_cloud=True,
        overlay_pose=False,
    )

    # Define the type of map to be made
    make_map = {
        "mesh": mapper.make_mesh,
        "pc": mapper.make_point_cloud,
    }

    # Create the mesh/point cloud
    make_map[model_type]()
