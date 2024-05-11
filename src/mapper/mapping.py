import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


class Mapping:
    def __init__(
        self,
        eps=0.04,
        min_points=10,
        ply_filepath=r"../common/data/gold_std/cloud.ply",
        preprocess_point_cloud=True,
        radius=0.1,
        max_nn=30,
        depth=9,
        alpha=0.02,
        radii = [0.005, 0.01, 0.02, 0.04]
    ):
        self.eps = eps
        self.min_points = min_points
        self.ply_filepath = ply_filepath
        self.preprocess_point_cloud = preprocess_point_cloud

        # Mesh data
        self.radius = radius
        self.max_nn = max_nn
        self.depth = depth
        self.alpha = alpha
        self.radii = radii

        # Load the point cloud
        self.pcd = o3d.io.read_point_cloud(self.ply_filepath)

    def make_point_cloud(self):
        if self.preprocess_point_cloud:
            # DBSCAN clustering
            self._clustering()

        # visualise mesh
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

        return mesh

    def _visualiser(self, data):
        # Create a visualizer object
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        # Make mesh
        self.vis.add_geometry(data)

        # Run the visualizer
        self.vis.run()
        self.vis.destroy_window()


if __name__ == '__main__':
    mapper = Mapping(
        eps=0.04,
        min_points=10,
        ply_filepath=r"../common/data/gold_std/cloud.ply"
    )

    # Either make a point cloud or a mesh
    # mapper.make_point_cloud()
    mapper.make_mesh()
