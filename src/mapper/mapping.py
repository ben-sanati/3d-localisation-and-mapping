import sys

sys.path.insert(0, r"../..") # noqa: E402
sys.path.append("/home/phoenix/base/active/3D-Mapping-ATK") # noqa: E402

import argparse
import os
import pickle

import numpy as np
import open3d as o3d
from src.utils.config import ConfigLoader
from src.utils.transformations import VisualisationTransforms
from src.utils.visualisation import Visualiser


class Mapping:
    def __init__(
        self,
        global_bboxes_data=None,
        pose=None,
        eps=0.04,
        min_points=10,
        ply_filepath=r"../common/data/gold_std/cloud.ply",
        preprocess_point_cloud=True,
        overlay_pose=False,
        radius=0.1,
        max_nn=30,
        depth=8,
        scale_factor=1.0,
        bbox_depth_buffer=0.03,  # 3cm
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
        for frame_index, bbox_list in self.global_bboxes_data.items():
            for bbox in bbox_list:
                points = [corner for corner in bbox]

                # Turn 2D corners into 3D corners (with a buffer)
                bbox_3d = self.transforms.create_3d_bounding_box(
                    points, self.bbox_depth_buffer
                )
                bbox_lines = self.visualiser.overlay_3d_bbox(bbox_3d)
                vis.add_geometry(bbox_lines)

        if self.overlay_pose:
            # Overlay pose onto 3d map
            pose_point_cloud = self.visualiser.overlay_pose(self.pose)
            vis.add_geometry(pose_point_cloud)

            # # Get directions (in a right-handed coordinate system, the direction of the camera is the 3rd column of R)
            directions = self.transforms.get_camera_direction(self.pose)

            # Get pose camera directions
            pose_directions = self.visualiser.overlay_pose_directions(
                pose_point_cloud.points, directions
            )
            vis.add_geometry(pose_directions)

        # Run the visualizer
        vis.run()
        vis.destroy_window()


if __name__ == "__main__":
    # Setup argparse config
    parser = argparse.ArgumentParser(description="Processing Configuration.")
    parser.add_argument(
        "--data", type=str, help="Data Folder Name.", default="gold_std"
    )
    args = parser.parse_args()
    data_folder = args.data

    # Load the configuration
    os.chdir("../..")
    config_path = r"src/common/configs/variables.cfg"
    cfg = ConfigLoader(config_path, data_folder)

    eps = 0.02
    min_points = 10

    with open(cfg.pickle_path, "rb") as file:
        variables = pickle.load(file)

    global_bboxes_data = variables["global_bboxes_data"]
    pose_df = variables["pose_df"]

    mapper = Mapping(
        global_bboxes_data=global_bboxes_data,
        pose=pose_df,
        eps=eps,
        min_points=min_points,
        ply_filepath=cfg.ply_path,
        preprocess_point_cloud=True,
        overlay_pose=True,
    )

    # Either make a point cloud or a mesh
    # mapper.make_point_cloud()
    mapper.make_mesh()
