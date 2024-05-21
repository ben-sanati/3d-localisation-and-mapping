import sys
import pickle
import sqlite3
import pandas as pd
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, r'../..')

from src.utils.transformations import Transformations


class PoseDataExtractor:
    def __init__(self, pose_path):
        self.pose_path = pose_path
        self.pcd = o3d.geometry.PointCloud()

        self.transformations = Transformations()

    def fetch_data(self):
        df = pd.read_csv(self.pose_path, sep=' ', skiprows=1, header=None)
        df.columns = ['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw', 'id']
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.drop(['id'], axis=1)
        return df

    def plot_pose(self, df):
        # Create a visualizer object
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        # Get pose points and camera directions
        self.pcd.points = o3d.utility.Vector3dVector(df[['tx', 'ty', 'tz']].values)
        colours = np.random.uniform(0, 1, size=(len(df), 3))
        self.pcd.colors = o3d.utility.Vector3dVector(colours)

        # directions = np.array([self._quaternion_to_rotation_matrix(q)[0:3, 2] for q in df[['qw', 'qx', 'qy', 'qz']].to_numpy()])
        directions = self.transformations.get_camera_direction(df)

        # Create line set for orientations
        lines = []
        line_colors = []
        for i, (point, direction) in enumerate(zip(self.pcd.points, directions)):
            lines.append([i, i + len(self.pcd.points)])
            line_colors.append([0, 1, 0])

        # Create line set geometry
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.vstack((self.pcd.points, self.pcd.points + 0.4 * directions)))  # Concatenate points for lines
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(line_colors)

        o3d.visualization.draw_geometries([self.pcd, line_set])
        self.vis.destroy_window()


if __name__ == "__main__":
    pose_path = "../common/data/gold_std/poses_id_new.txt"

    extractor = PoseDataExtractor(pose_path)
    df = extractor.fetch_data()
    print(f"Pose Data:\n{df}")
    extractor.plot_pose(df)
