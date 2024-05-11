import open3d as o3d


class Mapping:
    def __init__(self, ply_filepath):
        self.ply_filepath = ply_filepath

        # Load the point cloud
        self.pcd = o3d.io.read_point_cloud(self.ply_filepath)

        # Create a visualizer object
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        # Get the rendering options
        opt = self.vis.get_render_option()
        opt.point_size = 1.0  # Set the point size to smaller values

    def make_point_cloud(self):
        # Add the point cloud to the visualizer
        self.vis.add_geometry(self.pcd)

        # Run the visualizer
        self.vis.run()
        self.vis.destroy_window()


if __name__ == '__main__':
    mapper = Mapping(r"../common/data/gold_std/cloud.ply")
    mapper.make_point_cloud()
