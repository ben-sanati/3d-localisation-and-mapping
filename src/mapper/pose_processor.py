import os
import sys
import cv2
import pickle
import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from torchvision.transforms.functional import to_pil_image

sys.path.insert(0, r'../..')

from src.detector.dataset import ImageDataset


class ProcessPose:
    def __init__(
        self,
        pose,
        dataset,
        bbox_coordinates,
        img_size
    ):
        """
        Initializes the ProcessPose class with the pose data, depth images, and bounding box coordinates.

        Parameters:
            pose (DataFrame): DataFrame containing poses for each frame.
            dataset (torch.Dataset): PyTorch Dataset of images and depth images corresponding to each frame.
            bbox_coordinates (dict): Dictionary of bounding boxes, keyed by frame index.
        """
        self.img_size = img_size
        self.pose = pose
        self.dataset = dataset
        self.bbox_coordinates = bbox_coordinates

        # Acquired from rtabmap-databaseViewer
        self.fx = 673 / 3
        self.fy = 673 / 3
        self.cx = 357 / 3
        self.cy = 483.5 / 3

    def get_global_coordinates(self):
        """
        Converts all bounding box coordinates from local camera frames to global coordinates.

        Returns:
            dict: Dictionary mapping each frame index to lists of global bounding box coordinates.
        """
        global_bbox_lines = {}

        # Iterate through each frame index and corresponding bounding boxes
        for frame_index, bboxes in self.bbox_coordinates.items():
            # Extract the corresponding pose for the current frame
            frame_pose = self.pose.loc[frame_index, ['tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']].values

            try:
                depth_image_path = f"src/common/data/gold_std/rtabmap_extract/depth/{frame_index+1}.png"
                depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)  # Use cv2.IMREAD_UNCHANGED or -1 to load raw data
                depth_image = cv2.resize(depth_image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                print(f"Depth Image: {depth_image.size()}", flush=True)
            except Exception:
                continue

            # Store the global bounding boxes for the current frame
            frame_global_bboxes = []

            # Convert each bounding box's corner points to global coordinates
            for bbox in bboxes:
                x_min, y_min, x_max, y_max, confidence, class_id = bbox

                # Define the corner points of the bounding box
                corners = [
                    (x_min, y_min), (x_min, y_max), 
                    (x_max, y_max), (x_max, y_min)
                ]

                # Convert each corner from 2D to 3D coordinates in the camera frame
                corners_3d = [self._depth_to_3d(int(x), int(y), depth_image) for x, y in corners]

                # Transform the 3D camera frame coordinates to global coordinates
                global_corners = [self._transform_to_global(corner, frame_pose) for corner in corners_3d]
                frame_global_bboxes.append(global_corners)

            # Map the frame index to the list of global bounding boxes
            global_bbox_lines[frame_index] = frame_global_bboxes

        return global_bbox_lines

    def _depth_to_3d(self, x, y, depth_image):
        """
        Converts 2D pixel coordinates from the depth image to 3D space coordinates.

        Parameters:
            x (int): X-coordinate in the 2D image.
            y (int): Y-coordinate in the 2D image.
            depth_image (numpy.ndarray): Depth image to convert coordinates from.

        Returns:
            numpy.ndarray: 3D coordinates [X, Y, Z] relative to the camera frame.
        """
        # Extract the depth value at (x, y) (rtabmap uses mm by default)
        Z = depth_image[1, x, y]

        # Convert (x, y) coordinates into 3D space based on camera intrinsic parameters
        X = (x - self.cx) * Z / self.fx
        Y = (y - self.cy) * Z / self.fy

        # Return the 3D point as a numpy array
        return np.array([X, Y, Z])

    def _transform_to_global(self, local_point, pose):
        """
        Transforms a 3D point from the camera frame to the global coordinate frame using the given pose.

        Parameters:
            local_point (numpy.ndarray): 3D coordinates [X, Y, Z] in the camera frame.
            pose (numpy.ndarray): Pose data containing translation and rotation (quaternion).

        Returns:
            numpy.ndarray: Transformed 3D point in the global coordinate frame.
        """
        # Extract the translation and quaternion rotation from the pose
        tx, ty, tz, qx, qy, qz, qw = pose
        translation = np.array([tx, ty, tz])
        rotation = R.from_quat([qx, qy, qz, qw])

        # Apply rotation and translation to obtain the global coordinates
        global_point = rotation.apply(local_point) + translation
        return global_point

    def view_3d_bbox(self, dataset):
        for frame_index, bboxes in self.bbox_coordinates.items():
            # Fetch images from the dataset
            rgb_tensor, depth_tensor = dataset[frame_index]

            # Convert tensors to PIL Images for display (assuming they're already in the correct size)
            rgb_image_pil = to_pil_image(rgb_tensor)
            depth_image_pil = to_pil_image(depth_tensor)

            # Convert PIL Images to OpenCV format for display
            rgb_image_cv = cv2.cvtColor(np.array(rgb_image_pil), cv2.COLOR_RGB2BGR)
            depth_image_cv = np.array(depth_image_pil)

            # Print depth image stats
            print(f"Depth ({depth_image_cv.shape}) : [{depth_image_cv.min()} - {depth_image_cv.max()}]")
            print(depth_image_cv, flush=True)

            # Normalize depth image for better visualization
            depth_image_norm_cv = cv2.normalize(depth_image_cv, None, 0, 255, cv2.NORM_MINMAX)
            depth_image_norm_cv = np.uint8(depth_image_norm_cv)

            # Display the images
            cv2.imshow("RGB Image", rgb_image_cv)
            cv2.imshow("Depth Image", depth_image_norm_cv)
            cv2.waitKey(0)  # Wait for key press to proceed to the next image
            cv2.destroyAllWindows()

            # Convert each bounding box's corner points to global coordinates
            print(f"Img 1:")
            for bbox in bboxes:
                x_min, y_min, x_max, y_max, confidence, class_id = bbox
                print(f"\tBBox: {bbox}")
                print(f"\tXmin: {x_min}\tYmin: {y_min}\tXmax: {x_max}\tYmax: {y_max}")

            # Convert PIL Images to Open3D images
            rgb_image_o3d = o3d.geometry.Image(np.array(rgb_image_pil))
            depth_image_o3d = o3d.geometry.Image(np.array(depth_image_cv).astype(np.uint16))  # Ensure depth is uint16

            # Create an RGBD image
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_image_o3d,
                depth_image_o3d,
                depth_scale=1000.0,  # Adjust this based on your depth unit
                depth_trunc=1.0,     # Adjust truncation for better visualization
                convert_rgb_to_intensity=False
            )

            # Generate point cloud
            point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                o3d.camera.PinholeCameraIntrinsic(256, 192, self.fx, self.fy, self.cx/2, self.cy/2)
            )

            # Visualize the point cloud
            o3d.visualization.draw_geometries([point_cloud])
            break


if __name__ == '__main__':
    # TODO: check 3d map
    os.chdir(r'../..')

    img_size = 640
    pickle_file = r"src/common/data/gold_std/variables.pkl"

    save_dir = r"src/common/data/gold_std"
    image_dir = f"{save_dir}/rtabmap_extract/rgb"
    depth_image_dir = f"{save_dir}/db_extract/depth"
    print(f"{os.getcwd()}", flush=True)

    with open(pickle_file, "rb") as file:
        variables = pickle.load(file)

    pose_df = variables["pose_df"]
    predictions = variables["predictions"]

    dataset = ImageDataset(image_dir=image_dir, depth_image_dir=depth_image_dir, img_size=img_size, processing=False)
    print(f"Pose: {pose_df}\n\nDepth Images: {len(dataset)}\n\nPredictions: {predictions}")

    pose_processing = ProcessPose(
        pose=pose_df,
        dataset=dataset,
        bbox_coordinates=predictions,
        img_size=img_size,
    )
    pose_processing.view_3d_bbox(dataset)
    # global_bboxes_data = pose_processing.get_global_coordinates()
