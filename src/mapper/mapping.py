import os
import sys
import pickle
import psutil
import argparse
import configparser
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, r'../..')
sys.path.append('/home/phoenix/base/active/3D-Mapping-ATK')


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
        alpha=0.02,
        radii = [0.005, 0.01, 0.02, 0.04],
        scale_factor = 1.0,
    ):
        self.eps = eps
        self.pose = pose
        self.min_points = min_points
        self.ply_filepath = ply_filepath
        self.overlay_pose = overlay_pose
        self.global_bboxes_data = global_bboxes_data
        self.preprocess_point_cloud = preprocess_point_cloud

        self.lines = [
            [0, 1], [1, 2], [2, 3], [3, 0], # bottom face
            [4, 5], [5, 6], [6, 7], [7, 4], # top face
            [0, 4], [1, 5], [2, 6], [3, 7]  # vertical edges
        ]

        # Mesh data
        self.radius = radius
        self.max_nn = max_nn
        self.depth = depth
        self.alpha = alpha
        self.radii = radii
        self.scale_factor = scale_factor

        # Load the point cloud
        self.pcd = o3d.io.read_point_cloud(
            self.ply_filepath,
            remove_nan_points=True,
            remove_infinite_points=True,
            print_progress=True,
        )

    def make_point_cloud(self):
        if self.preprocess_point_cloud:
            # DBSCAN clustering
            self._clustering()

        # Visualise mesh
        self._visualiser(self.pcd)

    def make_mesh(self, algo_method="Poisson"):
        mesh_methods = {
            "Poisson": self._poisson_surface_recon,
            "Alpha": self._alpha_shapes,
            "BPA": self._ball_pivoting_algo
        }

        if self.preprocess_point_cloud:
            # DBSCAN clustering
            self._clustering()

        # Create mesh
        print("\tMaking mesh...")
        mesh = mesh_methods[algo_method]()

        # Optimise mesh
        print("\tOptimising mesh...")
        mesh = self._optimise_mesh(mesh)

        # Visualise mesh
        print("\tVisualising mesh...")
        self._visualiser(mesh)

    def _clustering(self):
        # Execute DBSCAN algorithm
        with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug
        ) as cm:
            labels = np.array(self.pcd.cluster_dbscan(eps=self.eps, min_points=self.min_points, print_progress=True))

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
        self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius, max_nn=self.max_nn))

        # Apply Poisson surface reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(self.pcd, depth=self.depth, scale=self.scale_factor)

        return mesh

    def _alpha_shapes(self):
        # Estimate normals
        self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius, max_nn=self.max_nn))

        # Compute alpha shape
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(self.pcd, self.alpha)

        return mesh

    def _ball_pivoting_algo(self):
        # Estimate normals
        self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius, max_nn=self.max_nn))

        # Ball Pivoting
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(self.pcd, o3d.utility.DoubleVector(self.radii))

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

        print(f"Point Cloud Data: {data}")

        # Make mesh/point_cloud
        vis.add_geometry(data)

        # Add bounding boxes to the visualizer
        for frame_index, bbox_list in self.global_bboxes_data.items():
            for bbox in bbox_list:
                points = [corner for corner in bbox]
                print(f"Global BBox Coordinates for Frame {frame_index}: {points}")

                line_set = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(points),
                    lines=o3d.utility.Vector2iVector(self.lines)
                )
                line_set.paint_uniform_color([1, 0, 0])
                vis.add_geometry(line_set)

        if self.overlay_pose:
            pose_point_cloud = o3d.geometry.PointCloud()
            pose_point_cloud.points = o3d.utility.Vector3dVector(self.pose[['tx', 'ty', 'tz']].values)
            vis.add_geometry(pose_point_cloud)

            # Get directions (in a right-handed coordinate system, the direction of the camera is the 3rd column of R)
            directions = np.array([self._quaternion_to_rotation_matrix(q)[0:3, 2] for q in self.pose[['qw', 'qx', 'qy', 'qz']].to_numpy()])

            lines = []
            line_colors = []
            for i, (point, direction) in enumerate(zip(pose_point_cloud.points, directions)):
                lines.append([i, i + len(pose_point_cloud.points)])
                if i == 1:
                    line_colors.append([1, 0, 0])
                else:
                    line_colors.append([0, 1, 0])

            # Create line set geometry
            pose_line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(np.vstack((pose_point_cloud.points, pose_point_cloud.points + 0.4 * directions))),
                lines=o3d.utility.Vector2iVector(lines),
            )
            pose_line_set.colors = o3d.utility.Vector3dVector(line_colors)
            vis.add_geometry(pose_line_set)

        # Run the visualizer
        vis.run()
        vis.destroy_window()

    @staticmethod
    def _quaternion_to_rotation_matrix(q):
        qw, qx, qy, qz = q
        rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()
        return rotation


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

    eps = 0.02
    min_points = 10

    with open(pickle_path, "rb") as file:
        variables = pickle.load(file)

    global_bboxes_data = variables["global_bboxes_data"]
    pose_df = variables["pose_df"]

    mapper = Mapping(
        global_bboxes_data=global_bboxes_data,
        pose=pose_df,
        eps=eps,
        min_points=min_points,
        ply_filepath=ply_path,
        preprocess_point_cloud=True,
        overlay_pose=True,
    )

    # Either make a point cloud or a mesh
    # mapper.make_point_cloud()
    mapper.make_mesh()
