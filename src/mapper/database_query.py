import pandas as pd
import numpy as np
import open3d as o3d


class PoseDataExtractor:
    def __init__(self, pose_path):
        self.pose_path = pose_path
        self.pcd = o3d.geometry.PointCloud()

    def fetch_data(self):
        df = pd.read_csv(self.pose_path, sep=' ', skiprows=1, header=None)
        df.columns = ['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        return df

    def plot_pose(self, df):
        # Create a visualizer object
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        # Get the rendering options
        opt = self.vis.get_render_option()
        opt.point_size = 2.0  # Set the point size to smaller values

        # Get pose points and camera directions
        self.pcd.points = o3d.utility.Vector3dVector(df[['tx', 'ty', 'tz']].values)
        colours = np.random.uniform(0, 1, size=(len(df), 3))
        self.pcd.colors = o3d.utility.Vector3dVector(colours)

        directions = np.array([self._quaternion_to_rotation_matrix(q)[0] for q in df[['qw', 'qx', 'qy', 'qz']].to_numpy()])

        # Create line set for orientations
        lines = []
        line_colors = []
        for i, (point, direction) in enumerate(zip(self.pcd.points, directions)):
            lines.append([i, i + len(self.pcd.points)])
            line_colors.append([1, 0, 0])  # Red color for direction vectors

        # Create line set geometry
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.vstack((self.pcd.points, self.pcd.points + 0.4 * directions)))  # Concatenate points for lines
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(line_colors)
        o3d.visualization.draw_geometries([self.pcd, line_set])
        self.vis.destroy_window()

    @staticmethod
    def _quaternion_to_rotation_matrix(q):
        qw, qx, qy, qz = q
        return np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])


if __name__ == '__main__':
    pose_path = '../common/data/gold_std/poses.txt'
    extractor = PoseDataExtractor(pose_path)
    df = extractor.fetch_data()
    print(df)
    extractor.plot_pose(df)
