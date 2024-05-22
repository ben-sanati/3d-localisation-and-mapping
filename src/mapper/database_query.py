import sys

import open3d as o3d
import pandas as pd

sys.path.insert(0, r"../..")

from src.utils.transformations import VisualisationTransforms  # noqa
from src.utils.visualisation import Visualiser  # noqa


class PoseDataExtractor:
    def __init__(self, pose_path):
        self.pose_path = pose_path
        self.pcd = o3d.geometry.PointCloud()

        self.visualiser = Visualiser()
        self.transforms = VisualisationTransforms()

    def fetch_data(self):
        df = pd.read_csv(self.pose_path, sep=" ", skiprows=1, header=None)
        df.columns = ["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw", "id"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.drop(["id"], axis=1)
        return df

    def plot_pose(self, df):
        # Drop timestamp column
        df = df.drop(["timestamp"], axis=1)

        # Create a visualizer object
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        # Get pose points and camera directions
        pose_point_cloud = self.visualiser.overlay_pose(df)
        directions = self.transforms.get_camera_direction(df)

        # Get pose camera directions
        pose_directions = self.visualiser.overlay_pose_directions(
            pose_point_cloud.points, directions
        )

        # Visualise pose
        o3d.visualization.draw_geometries([pose_point_cloud, pose_directions])
        self.vis.destroy_window()


if __name__ == "__main__":
    pose_path = "../common/data/gold_std/poses.txt"

    extractor = PoseDataExtractor(pose_path)
    df = extractor.fetch_data()
    print(f"Pose Data:\n{df}")
    extractor.plot_pose(df)
