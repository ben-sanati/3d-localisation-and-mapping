import argparse
import os
import pickle
import sys

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, r"../..")

from src.utils.config import ConfigLoader  # noqa


class Comparison:
    def __init__(
        self,
        base_pose_df,
        comparison_pose_df,
        optimised_bboxes,
        voxel_size=0.05,
        base_map_name="gold_std",
    ):
        # TODO: Change the variable structure such that you pass in a dict of {base: val, comparison: val}
        self.base_pose_df = base_pose_df
        self.comparison_pose_df = comparison_pose_df
        self.voxel_size = voxel_size
        self.optimised_bboxes = optimised_bboxes
        self.base_map_filepath = self._parse_filepaths(base_map_name)

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
        comparison_map_filepath = self._parse_filepaths(comparison_map_name)

        # Load Point Clouds
        print("Loading point clouds.")
        base_map = self._load_pc(self.base_map_filepath)
        comparison_map = self._load_pc(comparison_map_filepath)

        # Preprocess Point Clouds
        print("Preprocessing point clouds.")
        base_map_down = self._preprocess_point_cloud(base_map)
        comparison_map_down = self._preprocess_point_cloud(comparison_map)

        # Get extreme poses for initial alignment
        start_pose_base, end_pose_base = self._get_extreme_poses(self.base_pose_df)
        start_pose_comparison, end_pose_comparison = self._get_extreme_poses(
            self.comparison_pose_df,
        )

        initial_transformation = self._compute_initial_transformation(
            start_pose_base,
            end_pose_base,
            start_pose_comparison,
            end_pose_comparison,
        )

        # Apply initial transformation to the comparison map
        comparison_map_down.transform(initial_transformation)

        # Align Point Clouds
        print("Aligning point clouds.")
        transformation = self._align_point_clouds(
            comparison_map_down, base_map_down, initial_transformation,
        )

        # Apply transformation to the comparison map
        print("Applying transformations.")
        comparison_map.transform(transformation)

        # Set colors for visualization
        base_map.paint_uniform_color([1, 0, 0])  # Red for base map
        comparison_map.paint_uniform_color([0, 1, 0])  # Green for comparison map

        # Visualize the aligned point clouds (optional)
        print("Visualising the maps.")
        o3d.visualization.draw_geometries([base_map, comparison_map])

        # Returning the transformation for further use if needed
        return transformation

    def _preprocess_point_cloud(self, pcd):
        # Downsample the point cloud
        pcd_down = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        
        # Estimate normals
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.voxel_size * 2, max_nn=30
            )
        )
        return pcd_down

    def _get_extreme_poses(self, pose_df):
        # Identify poses at the extremities (e.g., first and last rows)
        start_pose = pose_df.iloc[0]
        end_pose = pose_df.iloc[-1]

        # Extract translation and rotation (quaternion) for both poses
        start_translation = start_pose[["tx", "ty", "tz"]].values
        start_quaternion = start_pose[["qw", "qx", "qy", "qz"]].values

        end_translation = end_pose[["tx", "ty", "tz"]].values
        end_quaternion = end_pose[["qw", "qx", "qy", "qz"]].values

        return (start_translation, start_quaternion), (end_translation, end_quaternion)

    def _compute_initial_transformation(
        self,
        start_pose_base,
        end_pose_base,
        start_pose_comparison,
        end_pose_comparison,
    ):
        # Compute translation
        translation = start_pose_comparison[0] - start_pose_base[0]

        # Compute rotation (assuming both rotations are provided as quaternions)
        rotation_base = R.from_quat(start_pose_base[1])
        rotation_comparison = R.from_quat(start_pose_comparison[1])
        rotation = rotation_comparison * rotation_base.inv()

        # Create the 4x4 transformation matrix
        transformation = np.eye(4)
        transformation[:3, :3] = rotation.as_matrix()
        transformation[:3, 3] = translation

        return transformation

    def _align_point_clouds(self, source, target, initial_transformation):
        threshold = 50  # Distance threshold for ICP

        # Perform ICP alignment (point-to-plane method)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source,
            target,
            threshold,
            initial_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )
        return reg_p2p.transformation


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

    base_pose_df = comparison_variables["pose_df"]

    map_comparison = Comparison(
        base_pose_df=base_pose_df,
        comparison_pose_df=comparison_pose_df,
        optimised_bboxes=comparison_optimised_bboxes,
    )
    map_comparison.compare(data_folder)
