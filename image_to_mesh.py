import sqlite3
import open3d as o3d
import numpy as np
import cv2

def connect_to_database(db_path):
    """ Connect to the SQLite database and return the connection. """
    return sqlite3.connect(db_path)

def fetch_rgb_image(conn):
    """ Fetch images along with calibration and pose data from the database. """
    cursor = conn.cursor()
    cursor.execute("SELECT Data.image, Data.calibration, Node.pose FROM Data JOIN Node ON Data.id = Node.id")
    data = []
    for row in cursor.fetchall():
        image_data = np.frombuffer(row[0], dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        break
    return image

def fetch_depth_images(conn):
    """ Fetch depth images and pose data from the database. """
    cursor = conn.cursor()
    cursor.execute("SELECT Data.depth, Node.pose FROM Data JOIN Node ON Data.id = Node.id WHERE Data.depth IS NOT NULL")
    for row in cursor.fetchall():
        if len(row[0]) % 2 != 0:
            continue  # Skip this row if the data size is not correct

        depth_data = np.frombuffer(row[0], dtype=np.uint16)
        depth_image = cv2.imdecode(depth_data, cv2.IMREAD_UNCHANGED)
        break
    return depth_image


if __name__ == '__main__':
    db_path = 'src/common/data/gold_std/data.db'
    conn = connect_to_database(db_path)
    rgb_image = fetch_rgb_image(conn)
    depth_image = fetch_depth_images(conn)

    # Target size - choose based on your requirement, e.g., size of RGB image
    target_size = rgb_image.shape[1], rgb_image.shape[0]
    depth_image_resized = cv2.resize(depth_image, target_size, interpolation=cv2.INTER_NEAREST)

    # # Show the RGB image
    # cv2.imshow('RGB Image', rgb_image)
    # cv2.waitKey(0)  # Wait for a key press

    # # Show the Depth image
    # cv2.imshow('Depth Image', depth_image_resized)
    # cv2.waitKey(0)
    
    # print(f"Depth: {depth_image_resized.shape}\tRGB: {rgb_image.shape}")

    # # Close all windows
    # cv2.destroyAllWindows()

    # put into open3d
    rgb_image_o3d = o3d.geometry.Image(rgb_image)
    depth_image_o3d = o3d.geometry.Image(depth_image_resized)

    # create an rgbd image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_image_o3d,
        depth_image_o3d,
        depth_scale=1000.0,  # Adjust this depending on your depth image's scale
        depth_trunc=100.0,     # Adjust the truncation depth according to your setup
        convert_rgb_to_intensity=False
    )

    # Define the intrinsic parameters of the camera (example values, adjust accordingly)
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        target_size[0],   # Width of the depth image
        target_size[1],  # Height of the depth image
        525.0,    # Focal length x
        525.0,    # Focal length y
        target_size[0] / 2,    # Principal point x
        target_size[1] / 2     # Principal point y
    )

    # Create a point cloud from the RGBD image
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsics
    )

    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud])