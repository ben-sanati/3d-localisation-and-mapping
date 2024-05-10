import sqlite3
import numpy as np
import cv2
import open3d as o3d
from io import StringIO

def connect_to_database(db_path):
    """ Connect to the SQLite database and return the connection. """
    return sqlite3.connect(db_path)

def fetch_depth_images_and_pose(conn):
    """ Fetch depth images and pose data from the database. """
    cursor = conn.cursor()
    cursor.execute("SELECT Data.depth, Node.pose FROM Data JOIN Node ON Data.id = Node.id WHERE Data.depth IS NOT NULL")
    depth_images_and_pose = []
    for row in cursor.fetchall():
        if len(row[0]) % 2 != 0:
            continue  # Skip this row if the data size is not correct

        depth_data = np.frombuffer(row[0], dtype=np.uint16)
        depth_image = cv2.imdecode(depth_data, cv2.IMREAD_UNCHANGED)

        # Use np.loadtxt to read pose data from a string
        try:
            pose_data_str = row[1].decode('utf-8')  # Decoding bytes to string
            pose = np.loadtxt(StringIO(pose_data_str))
        except ValueError:
            continue  # Skip rows where pose data is malformed

        depth_images_and_pose.append((depth_image, pose.reshape((4, 4))))
    return depth_images_and_pose

def generate_point_clouds(depth_images_and_pose, intrinsics):
    """Convert depth images to point clouds using the given intrinsics and pose, and compute normals."""
    point_clouds = []
    for depth_image, pose in depth_images_and_pose:
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(depth_image),
            o3d.camera.PinholeCameraIntrinsic(intrinsics['width'], intrinsics['height'], intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']),
            depth_scale=1000.0,
            depth_trunc=3.0,
            stride=1
        )
        # Apply pose transformation to point cloud
        pcd.transform(pose)
        # Estimate normals for each individual point cloud
        if not pcd.is_empty():
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        point_clouds.append(pcd)
    return point_clouds

def create_mesh_from_point_clouds(point_clouds):
    """Create a mesh from combined point clouds after ensuring normals are present."""
    combined_pcd = o3d.geometry.PointCloud()
    # Combine only point clouds that have data and normals
    for pcd in point_clouds:
        if not pcd.is_empty() and pcd.has_normals():
            combined_pcd += pcd

    if combined_pcd.is_empty():
        print("No valid data in point clouds. Check your input data and parameters.")
        return None

    # Ensure normals are estimated on the combined point cloud if not already there
    if not combined_pcd.has_normals():
        combined_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    if not combined_pcd.has_normals():
        print("Failed to estimate normals on the combined point cloud.")
        return None

    # Reorient normals consistently
    combined_pcd.orient_normals_consistent_tangent_plane(k=50)

    # Define the radius for ball pivoting
    radii = [0.05, 0.1, 0.2]  # Adjust these radii based on your specific point cloud scale
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        combined_pcd,
        o3d.utility.DoubleVector(radii)
    )
    return mesh

if __name__ == '__main__':
    db_path = 'src/common/data/gold_std/data.db'
    conn = connect_to_database(db_path)
    depth_images_and_pose = fetch_depth_images_and_pose(conn)
    camera_intrinsics = {'width': 720, 'height': 960, 'fx': 525, 'fy': 525, 'cx': 319.5, 'cy': 239.5}  # Adjust these values
    # Main execution part of your script
    point_clouds = generate_point_clouds(depth_images_and_pose, camera_intrinsics)
    mesh = create_mesh_from_point_clouds(point_clouds)

    if mesh is not None:
        o3d.io.write_triangle_mesh("output_mesh.ply", mesh)
        o3d.visualization.draw_geometries([mesh])
    else:
        print("Mesh generation failed. No mesh to save or visualize.")

