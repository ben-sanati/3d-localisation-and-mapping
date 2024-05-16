import os
import sys
import pickle
import psutil
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, r'../..')


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
        depth=10,
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
            # [4, 5], [5, 6], [6, 7], [7, 4], # top face
            # [0, 4], [1, 5], [2, 6], [3, 7]  # vertical edges
        ]

        # Mesh data
        self.radius = radius
        self.max_nn = max_nn
        self.depth = depth
        self.alpha = alpha
        self.radii = radii
        self.scale_factor = scale_factor

        # Load the point cloud
        self.pcd = o3d.io.read_point_cloud(self.ply_filepath)

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
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(self.pcd, depth=self.depth)

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

        # Make mesh/point_cloud
        print(data)
        vis.add_geometry(data)

        # Add bounding boxes to the visualizer
        for frame_index, bbox_list in self.global_bboxes_data.items():
            for bbox in bbox_list:
                points = [corner for corner in bbox]

                line_set = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(points),
                    lines=o3d.utility.Vector2iVector(self.lines)
                )
                render_option = vis.get_render_option()
                render_option.line_width = 10.0
                line_set.paint_uniform_color([1, 0, 0])
                vis.add_geometry(line_set)

        if self.overlay_pose:
            pose_point_cloud = o3d.geometry.PointCloud()
            pose_point_cloud.points = o3d.utility.Vector3dVector(self.pose[['tx', 'ty', 'tz']].values)
            vis.add_geometry(pose_point_cloud)

        # Run the visualizer
        vis.run()
        vis.destroy_window()


if __name__ == '__main__':
    eps = 0.02
    min_points = 10

    with open(r"../common/data/gold_std/variables.pkl", "rb") as file:
        variables = pickle.load(file)

    global_bboxes_data = variables["global_bboxes_data"]
    pose_df = variables["pose_df"]

    mapper = Mapping(
        global_bboxes_data=global_bboxes_data,
        pose=pose_df,
        eps=eps,
        min_points=min_points,
        ply_filepath=r"../common/data/gold_std/cloud.ply",
        preprocess_point_cloud=True,
        overlay_pose=True,
    )

    # Either make a point cloud or a mesh
    # mapper.make_point_cloud()
    mapper.make_mesh()
