import argparse
import os
import pickle
import sys
import logging

import numpy as np
import open3d as o3d

sys.path.insert(0, r"../..")

from src.utils.config import ConfigLoader  # noqa
from src.utils.alignment import AlignmentTransforms  # noqa


class Alignment:
    def __init__(
        self,
        base_pose_df,
        comparison_pose_df,
        base_bboxes,
        comparison_bboxes,
        voxel_size=0.05,
        bbox_depth_buffer=0.02,
        distance_threshold=50,
        base_map_name="gold_std",
    ):
        self.pose_df = {"base": base_pose_df, "comparison": comparison_pose_df}
        self.optimised_bboxes = {"base": base_bboxes, "comparison": comparison_bboxes}
        self.voxel_size = voxel_size
        self.bbox_depth_buffer = bbox_depth_buffer
        self.distance_threshold = distance_threshold  # Distance threshold for ICP
        self.base_map_filepath = self._parse_filepaths(base_map_name)

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Comparison class initialized")

        # Instance util classes
        self.alignment = AlignmentTransforms()

        # Set verbosity and setup visualisation
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
        self.vis = o3d.visualization.Visualizer()

    @staticmethod
    def _parse_filepaths(data_folder):
        data_dir = "src/common/data"
        return os.path.join(data_dir, data_folder, "cloud.ply")

    @staticmethod
    def _load_pc(map_name):
        map = o3d.io.read_point_cloud(
            map_name,
            remove_nan_points=True,
            remove_infinite_points=True,
            print_progress=True,
        )
        return map

    def compare(self, comparison_map_name):
        # Load point clouds
        self.logger.info("Loading base and comparison point clouds")
        base_pcd = self._load_pc(self.base_map_filepath)
        comparison_pcd = self._load_pc(self._parse_filepaths(comparison_map_name))

        # Initial point cloud alignment
        self.logger.info("Calculating point cloud alignment.")
        pca_rotation_matrix, center = self.alignment._align_principal_axes(comparison_pcd, base_pcd)
        mean_position_transformation = self.alignment._align_mean_positions(comparison_pcd, base_pcd)
        best_transformation = self.alignment._shift_along_principal_components(comparison_pcd, base_pcd, self.voxel_size)
        result_icp = self.alignment._refine_registration(comparison_pcd, base_pcd, self.voxel_size)

        self.logger.info("Performing point cloud alignment.")
        comparison_pcd.rotate(pca_rotation_matrix, center=center)
        comparison_pcd.transform(mean_position_transformation)
        comparison_pcd.transform(best_transformation)
        comparison_pcd.transform(result_icp.transformation)

        self.logger.info("Transforming comparison bounding boxes.")
        transformed_bboxes_comparison = self.alignment._rotate_bboxes(pca_rotation_matrix, self.optimised_bboxes['comparison'], center)
        transformed_bboxes_comparison = self.alignment._transform_bboxes(mean_position_transformation, transformed_bboxes_comparison)
        # transformed_bboxes_comparison = self.alignment._transform_bboxes(best_transformation, transformed_bboxes_comparison)
        # transformed_bboxes_comparison = self.alignment._transform_bboxes(result_icp.transformation, transformed_bboxes_comparison)

        self.logger.info("Converting point clouds to meshes")
        base_mesh = self.alignment._create_mesh_from_point_cloud(base_pcd)
        comparison_mesh = self.alignment._create_mesh_from_point_cloud(comparison_pcd)

        self.logger.info("Creating line sets for bounding boxes")
        base_bboxes_lines = self.alignment._visualize_bboxes(self.optimised_bboxes['base'], colour=[1, 0, 0])  # Red for base
        comparison_bboxes_lines = self.alignment._visualize_bboxes(transformed_bboxes_comparison, colour=[0, 1, 0])  # Green for comparison

        self.logger.info("Coloring meshes for visualization")
        base_mesh.paint_uniform_color([1, 0, 0])  # Red
        comparison_mesh.paint_uniform_color([0, 1, 0])  # Green

        self.logger.info("Visualizing aligned meshes and bounding boxes")
        self.vis.create_window()
        # self.vis.add_geometry(base_mesh)
        self.vis.add_geometry(comparison_mesh)
        for bbox_lines in base_bboxes_lines:
            self.vis.add_geometry(bbox_lines)
        for bbox_lines in comparison_bboxes_lines:
            self.vis.add_geometry(bbox_lines)
        self.vis.run()
        self.vis.destroy_window()

        # Restore the verbosity level after the process is done
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


if __name__ == "__main__":
    # Setup argparse config
    parser = argparse.ArgumentParser(description="Processing Configuration")
    parser.add_argument(
        "--data", type=str, help="Data Folder Name.", default="ideal_scan"
    )
    args = parser.parse_args()
    data_folder = args.data

    if data_folder == "gold_std":
        raise ValueError("The parameter 'gold_std' is not allowed for --data.")

    # Load the configuration
    os.chdir("../..")
    config_path = r"src/common/configs/variables.cfg"
    cfg = ConfigLoader(config_path, data_folder)

    with open(cfg.pickle_path, "rb") as read_file:
        comparison_variables = pickle.load(read_file)

    comparison_pose_df = comparison_variables["pose_df"]
    comparison_optimised_bboxes = comparison_variables["optimised_bboxes"]

    cfg = ConfigLoader(config_path, "gold_std")
    with open(cfg.pickle_path, "rb") as read_file:
        base_variables = pickle.load(read_file)

    base_pose_df = base_variables["pose_df"]
    base_optimised_bboxes = base_variables["optimised_bboxes"]

    map_alignment = Alignment(
        base_pose_df=base_pose_df,
        comparison_pose_df=comparison_pose_df,
        base_bboxes=base_optimised_bboxes,
        comparison_bboxes=comparison_optimised_bboxes,
    )
    map_alignment.compare(data_folder)
