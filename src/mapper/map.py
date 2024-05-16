import os
import sys
import cv2
import pickle
import numpy as np
import configparser
import open3d as o3d
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image

sys.path.insert(0, r'../..')

from src.detector.dataset import ImageDataset


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    rotation = o3d.geometry.get_rotation_matrix_from_quaternion([qw, qx, qy, qz])
    return rotation

def transform_point_cloud(pcd, transformation):
    return pcd.transform(transformation)

def main(pose_df, dataset, depth_width, depth_height, scale_depth=100):
    pcd_combined = o3d.geometry.PointCloud()

    loop = tqdm(enumerate(pose_df.iterrows()), total=len(pose_df))
    for frame_idx, (_, pose) in loop:
        # Read images
        rgb_tensor, depth_tensor, camera_intrinsics = dataset[frame_idx]
        rgb_image_pil = to_pil_image(rgb_tensor)
        depth_image_pil = to_pil_image(depth_tensor)
        depth_image_cv = np.array(depth_image_pil)

        # Create RGBD image
        rgb_image_o3d = o3d.geometry.Image(np.array(rgb_image_pil))
        depth_image_o3d = o3d.geometry.Image(np.array(depth_image_cv).astype(np.uint16))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_image_o3d,
            depth_image_o3d,
            depth_scale=scale_depth,
            convert_rgb_to_intensity=False
        )

        # Create point cloud from RGBD image
        depth_to_rgb_scale = camera_intrinsics["image_width"] / depth_width
        fx = camera_intrinsics["fx"] / depth_to_rgb_scale
        fy = camera_intrinsics["fy"] / depth_to_rgb_scale
        cx = camera_intrinsics["cx"] / depth_to_rgb_scale
        cy = camera_intrinsics["cy"] / depth_to_rgb_scale
        intrinsics = o3d.camera.PinholeCameraIntrinsic(depth_width, depth_height, fx, fy, cx, cy)
        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            intrinsics
        )
        
        # Extract transformation data from the DataFrame
        tx, ty, tz = pose['tx'], pose['ty'], pose['tz']
        qx, qy, qz, qw = pose['qx'], pose['qy'], pose['qz'], pose['qw']
        
        # Convert quaternion to rotation matrix
        rotation_matrix = quaternion_to_rotation_matrix(qx, qy, qz, qw)

        # Form the 4x4 transformation matrix
        transformation = np.eye(4)
        transformation[:3, :3] = rotation_matrix
        transformation[:3, 3] = [tx, ty, tz]

        # Apply transformation
        pcd = transform_point_cloud(point_cloud, transformation)

        # Combine point clouds
        pcd_combined += pcd

        # Update progress bar
        loop.set_description(f"Frame [{frame_idx + 1}/{len(pose_df)}]")

    # Downsample the combined point cloud
    pcd_combined = pcd_combined.voxel_down_sample(voxel_size=0.02)

    # Configure the 3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd_combined)

    # Visualize
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    os.chdir(r'../..')
    config_path = r"src/common/configs/variables.cfg"
    config = configparser.ConfigParser()
    config.read(config_path)

    img_size = config.getint('detection', 'img_size')
    depth_width = config.getint('mapping', 'depth_width')
    depth_height = config.getint('mapping', 'depth_height')
    pickle_path = config['paths']['pickle_path']

    save_dir = config['paths']['save_dir']
    image_dir = config['paths']['image_dir']
    depth_image_dir = config['paths']['depth_image_dir']
    calibration_dir = config['paths']['calibration_dir']

    with open(pickle_path, "rb") as file:
        variables = pickle.load(file)

    pose_df = variables["pose_df"]
    predictions = variables["predictions"]

    dataset = ImageDataset(
        image_dir=image_dir,
        depth_image_dir=depth_image_dir,
        calibration_dir=calibration_dir,
        img_size=img_size,
        processing=False
    )
    print(f"Pose: {pose_df}\n\nDepth Images: {len(dataset)}\n\nPredictions: {predictions}")
    main(pose_df, dataset, depth_width, depth_height)
