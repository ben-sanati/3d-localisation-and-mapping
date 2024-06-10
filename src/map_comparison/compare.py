import argparse
import os
import pickle
import sys
import logging
import copy

import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, r"../..")

from src.utils.config import ConfigLoader  # noqa


class Comparison:
    def __init__(
        self,
        base_pose_df,
        comparison_pose_df,
        base_bboxes,
        comparison_bboxes,
        voxel_size=0.05,
        distance_threshold=50,
        base_map_name="gold_std",
    ):
        self.pose_df = {"base": base_pose_df, "comparison": comparison_pose_df}
        self.optimised_bboxes = {"base": base_bboxes, "comparison": comparison_bboxes}
        self.voxel_size = voxel_size
        self.distance_threshold = distance_threshold  # Distance threshold for ICP
        self.base_map_filepath = self._parse_filepaths(base_map_name)

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Comparison class initialized")
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

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
        # TODO: Change algorithm such that
        # 1. We orient the bbox positions such that they first align
        # 2. Calculate the transformation to the bbox positions to do this
        # 3. Use this as the initial transformation and then perform ICP
        # 4. Apply the ICP transformation to the bboxes
        # Load point clouds
        self.logger.info("Loading base and comparison point clouds")
        base_pcd = self._load_pc(self.base_map_filepath)
        comparison_pcd = self._load_pc(self._parse_filepaths(comparison_map_name))

        self.logger.info("Aligning principal axes of point clouds")
        aligned_comparison_pcd = self._align_principal_axes(comparison_pcd, base_pcd)

        self.logger.info("Aligning mean positions of point clouds")
        aligned_comparison_pcd = self._align_mean_positions(aligned_comparison_pcd, base_pcd)

        self.logger.info("Iteratively shifting along principal components for optimal fit")
        best_transformation = self._shift_along_principal_components(aligned_comparison_pcd, base_pcd)

        self.logger.info("Applying the best initial transformation")
        aligned_comparison_pcd.transform(best_transformation)

        self.logger.info("Performing fine registration (ICP)")
        result_icp = self._refine_registration(aligned_comparison_pcd, base_pcd, self.voxel_size)

        self.logger.info(f"ICP Transformation:\n{result_icp.transformation}")

        self.logger.info("Transforming comparison point cloud")
        aligned_comparison_pcd.transform(result_icp.transformation)

        self.logger.info("Converting point clouds to meshes")
        base_mesh = self._create_mesh_from_point_cloud(base_pcd)
        comparison_mesh = self._create_mesh_from_point_cloud(aligned_comparison_pcd)

        self.logger.info("Coloring meshes for visualization")
        base_mesh.paint_uniform_color([1, 0, 0])  # Red
        comparison_mesh.paint_uniform_color([0, 1, 0])  # Green

        self.logger.info("Visualizing aligned meshes")
        o3d.visualization.draw_geometries([base_mesh, comparison_mesh])

        # Restore the verbosity level after the process is done
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    @staticmethod
    def _downsample(pcd, voxel_size):
        logging.info(f"Downsampling point cloud with voxel size: {voxel_size}")
        return pcd.voxel_down_sample(voxel_size)

    def _align_mean_positions(self, source, target):
        source_mean = np.mean(np.asarray(source.points), axis=0)
        target_mean = np.mean(np.asarray(target.points), axis=0)

        logging.info(f"Source mean position: {source_mean}")
        logging.info(f"Target mean position: {target_mean}")

        translation = target_mean - source_mean
        logging.info(f"Translating source by: {translation}")

        source.translate(translation)

        return source

    def _align_principal_axes(self, source, target):
        source_points = np.asarray(source.points)
        target_points = np.asarray(target.points)

        pca_source = PCA(n_components=3).fit(source_points)
        pca_target = PCA(n_components=3).fit(target_points)

        source_rotation = pca_source.components_.T
        target_rotation = pca_target.components_.T

        logging.info(f"Source principal axes:\n{source_rotation}")
        logging.info(f"Target principal axes:\n{target_rotation}")

        # Compute rotation matrix to align source to target
        rotation_matrix = np.dot(target_rotation, source_rotation.T)
        center = np.mean(source_points, axis=0)
        source.rotate(rotation_matrix, center=center)

        return source

    def _shift_along_principal_components(self, source, target):
        pca_source = PCA(n_components=3).fit(np.asarray(source.points))
        principal_components = pca_source.components_

        best_fit_score = -np.inf
        best_transformation = np.eye(4)

        # Define the range and step size for shifting along the principal component
        shift_range = np.linspace(-5, 5, num=100)
        for shift in shift_range:
            temp_source = copy.deepcopy(source)
            temp_source.translate(principal_components[0] * shift)
            fit_score = self._evaluate_fit(temp_source, target)
            logging.info(f"Testing shift: {shift}\tFit score: {fit_score}")
            if fit_score > best_fit_score:
                best_fit_score = fit_score
                best_transformation = np.eye(4)
                best_transformation[:3, 3] = principal_components[0] * shift

        logging.info(f"Best transformation found with fit score: {best_fit_score}")
        return best_transformation

    def _evaluate_fit(self, source, target):
        distance_threshold = self.voxel_size * 0.4
        # Perform ICP with 0 iterations to get the fitness score
        result = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1)
        )
        return result.fitness

    def _refine_registration(self, source, target, voxel_size, initial_transformation=np.eye(4)):
        distance_threshold = voxel_size * 0.4
        logging.info("Refining registration using ICP with convergence criteria")

        result = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold,
            initial_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria()
        )

        logging.info(f"ICP registration complete with fitness: {result.fitness}")
        return result

    @staticmethod
    def _create_mesh_from_point_cloud(pcd):
        logging.info("Creating mesh from point cloud using Poisson reconstruction")
        pcd.estimate_normals()
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        
        # Optional: remove low density vertices to clean up the mesh
        densities = np.asarray(densities)
        vertices_to_remove = densities < np.mean(densities)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        mesh.compute_vertex_normals()
        return mesh


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

    map_comparison = Comparison(
        base_pose_df=base_pose_df,
        comparison_pose_df=comparison_pose_df,
        base_bboxes=base_optimised_bboxes,
        comparison_bboxes=comparison_optimised_bboxes,
    )
    map_comparison.compare(data_folder)
