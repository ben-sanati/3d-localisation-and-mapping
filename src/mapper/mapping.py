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
    ):
        self.eps = eps
        self.min_points = min_points
        self.ply_filepath = ply_filepath
        self.preprocess_point_cloud = preprocess_point_cloud

        # Load the point cloud
        self.pcd = o3d.io.read_point_cloud(self.ply_filepath)

    def make_point_cloud(self):
        if self.preprocess_point_cloud:
            # DBSCAN clustering
            self._clustering()

        # Create a visualizer object
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        # Make pointcloud
        self.vis.add_geometry(self.pcd)

        # Run the visualizer
        self.vis.run()
        self.vis.destroy_window()

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


if __name__ == '__main__':
    mapper = Mapping(
        eps=0.04,
        min_points=10,
        ply_filepath=r"../common/data/gold_std/cloud.ply"
    )
    mapper.make_point_cloud()
