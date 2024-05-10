import open3d as o3d

# Load the point cloud
pcd = o3d.io.read_point_cloud("src/common/data/gold_std/cloud.ply")

# Create a visualizer object
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add the point cloud to the visualizer
vis.add_geometry(pcd)

# Get the rendering options
opt = vis.get_render_option()
opt.point_size = 1.0  # Set the point size to smaller values

# Run the visualizer
vis.run()
vis.destroy_window()
