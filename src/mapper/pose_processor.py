import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


class ProcessPose:
    def __init__(
        self,
        pose,
        depth_images,
        bbox_coordinates,
        img_size
    ):
        """
        Initializes the ProcessPose class with the pose data, depth images, and bounding box coordinates.

        Parameters:
            pose (DataFrame): DataFrame containing poses for each frame.
            depth_images (List[numpy.ndarray]): List of depth images corresponding to each frame.
            bbox_coordinates (dict): Dictionary of bounding boxes, keyed by frame index.
        """
        self.img_size = img_size
        self.pose = pose
        self.depth_images = depth_images
        self.bbox_coordinates = bbox_coordinates

        # Acquired from rtabmap-databaseViewer
        self.fx = 673
        self.fy = 673
        self.cx = 357
        self.cy = 483.5

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
            depth_image = self.depth_images[frame_index]

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
                corners_3d = [self.depth_to_3d(int(x), int(y), depth_image) for x, y in corners]

                # Transform the 3D camera frame coordinates to global coordinates
                global_corners = [self.transform_to_global(corner, frame_pose) for corner in corners_3d]
                frame_global_bboxes.append(global_corners)

            # Map the frame index to the list of global bounding boxes
            global_bbox_lines[frame_index] = frame_global_bboxes

        return global_bbox_lines

    def depth_to_3d(self, x, y, depth_image):
        """
        Converts 2D pixel coordinates from the depth image to 3D space coordinates.

        Parameters:
            x (int): X-coordinate in the 2D image.
            y (int): Y-coordinate in the 2D image.
            depth_image (numpy.ndarray): Depth image to convert coordinates from.

        Returns:
            numpy.ndarray: 3D coordinates [X, Y, Z] relative to the camera frame.
        """
        # Extract the depth value at (x, y), scaled down to meters
        max_bbx_coord = self.img_size - 1
        x = max_bbx_coord if x > max_bbx_coord else x
        y = max_bbx_coord if y > max_bbx_coord else y
        Z = depth_image[y, x, -1]

        # Convert (x, y) coordinates into 3D space based on camera intrinsic parameters
        X = (x - self.cx) * Z / self.fx
        Y = (y - self.cy) * Z / self.fy

        # Return the 3D point as a numpy array
        return np.array([X, Y, Z])

    def transform_to_global(self, local_point, pose):
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
