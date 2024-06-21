import numpy as np
import open3d as o3d
import time

class VisualiseAlignment:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.frames = []

    def _apply_incremental_transformation(self, point_cloud, transformation, steps=50, delay=0.05):
        """
        Apply the given transformation incrementally to the point cloud for smooth transition visualization.
        Args:
            point_cloud (open3d.geometry.PointCloud): The point cloud to transform.
            transformation (numpy.ndarray): The 4x4 transformation matrix.
            steps (int): Number of incremental steps.
            delay (float): Time delay between each step in seconds.
        """
        incremental_transformation = np.linalg.matrix_power(transformation, 1 / steps)
        for _ in range(steps):
            point_cloud.transform(incremental_transformation)
            self._capture_frame()
            self.vis.update_geometry(point_cloud)
            self.vis.poll_events()
            self.vis.update_renderer()
            time.sleep(delay)

    def _capture_frame(self):
        """
        Capture a frame from the visualizer and store it in the frames list.
        """
        image = self.vis.capture_screen_float_buffer(do_render=True)
        self.frames.append((np.asarray(image) * 255).astype(np.uint8))

    def visualize(self, base_pcd, comparison_pcd, transformations):
        """
        Visualize the incremental transformations.
        Args:
            base_pcd (open3d.geometry.PointCloud): The base point cloud.
            comparison_pcd (open3d.geometry.PointCloud): The comparison point cloud.
            transformations (list): List of 4x4 transformation matrices.
        """
        self.vis.create_window()
        self.vis.add_geometry(base_pcd)
        self.vis.add_geometry(comparison_pcd)
        
        for transformation in transformations:
            self._apply_incremental_transformation(comparison_pcd, transformation)
        
        self.vis.run()
        self.vis.destroy_window()

    def create_gif(self, output_path="alignment.gif"):
        """
        Create a GIF from the captured frames.
        Args:
            output_path (str): Path to save the output GIF.
        """
        import imageio
        imageio.mimsave(output_path, self.frames, fps=10)
