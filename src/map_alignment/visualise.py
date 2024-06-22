import argparse
import logging
import os
import pickle
import sys
import time
import imageio
import cv2
import numpy as np
import open3d as o3d
from scipy.linalg import logm, expm

sys.path.insert(0, r"../..")

from src.utils.config import ConfigLoader  # noqa


class VisualiseAlignment:
    def __init__(self, base_map_filepath, comparison_map_filepath):
        self.base_pcd = o3d.io.read_point_cloud(base_map_filepath)
        self.comparison_pcd = o3d.io.read_point_cloud(comparison_map_filepath)
        self.frames = []
        self.total_frames = 0

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info("VisualiseAlignment class initialized")

    def _apply_incremental_transformation(self, transformation, steps=20):
        """
        Apply the given transformation incrementally to the point cloud for smooth transition visualization.
        Args:
            transformation (numpy.ndarray): The 4x4 transformation matrix.
            steps (int): Number of incremental steps.
        """
        incremental_transformation = self._compute_incremental_transformation(transformation, steps)

        for step in range(steps):
            self.logger.info(f"Applying incremental transformation step {step + 1}/{steps}")
            self.comparison_pcd.transform(incremental_transformation)
            self._capture_frame()

    def _compute_incremental_transformation(self, transformation, steps):
        log_transformation = logm(transformation)
        return expm(log_transformation / steps)

    def _capture_frame(self):
        """
        Capture the current state of the comparison point cloud for visualization.
        """
        self.logger.info(f"Capturing frame {len(self.frames) + 1}/{self.total_frames}")

        # Render the point cloud to image buffer
        img = self._render_point_cloud_to_image(self.comparison_pcd)

        # Append to frames list
        self.frames.append(img)

    def _render_point_cloud_to_image(self, point_cloud):
        """
        Render the given point cloud to an image buffer for frame capture.
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(self.base_pcd)
        vis.add_geometry(point_cloud)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.1)  # Small delay to ensure the frame is captured correctly

        # Capture screen as float buffer and convert to numpy array
        img = vis.capture_screen_float_buffer(True)
        vis.destroy_window()

        # Convert float image to uint8
        img_array = np.asarray(img) * 255
        img_array = img_array.astype(np.uint8)

        # Ensure image is 3 channels (RGB)
        if len(img_array.shape) == 2 or img_array.shape[2] == 1:
            img_array = np.stack((img_array, img_array, img_array), axis=-1)

        # Resize the image to be divisible by 16
        height, width, _ = img_array.shape
        new_height = (height + 15) // 16 * 16
        new_width = (width + 15) // 16 * 16
        resized_img = cv2.resize(img_array, (new_width, new_height))

        return resized_img

    def create_video(self, transformations, output_video="alignment_animation.mp4", fps=30):
        """
        Create a video of the accumulated transformations as meshes.
        Args:
            transformations (list): List of transformations to apply.
            output_video (str): Output video filename.
            fps (int): Frames per second for the video.
        """
        self.logger.info("Creating video")
        self.total_frames = len(transformations) * 20

        # Apply all transformations step by step
        for i, transformation in enumerate(transformations):
            self.logger.info(f"Processing transformation {i + 1}/{len(transformations)}")
            if isinstance(transformation, tuple):  # Handle (rotation, center) tuple
                rotation, center = transformation
                self.comparison_pcd.rotate(rotation, center)
                self._capture_frame()
            else:
                self._apply_incremental_transformation(transformation)

        # Save frames as a video
        self.logger.info(f"Saving video to {output_video}")
        with imageio.get_writer(output_video, fps=fps, format='mp4') as writer:
            for i, frame in enumerate(self.frames):
                self.logger.info(f"Writing frame {i + 1}/{self.total_frames}")
                writer.append_data(frame)

        self.logger.info("Video creation completed")


if __name__ == "__main__":
    # Setup argparse config
    parser = argparse.ArgumentParser(description="Processing Configuration")
    parser.add_argument(
        "--data", type=str, help="Data Folder Name.", default="ideal_scan"
    )
    args = parser.parse_args()
    data_folder = args.data

    if data_folder == "gold_std":
        raise ValueError("The parameter 'gold_std' is not allowed for --data.")

    # Load the configuration
    os.chdir("../..")
    config_path = r"src/common/configs/variables.cfg"
    cfg = ConfigLoader(config_path, data_folder)
    save_pickle_path = cfg.pickle_path

    with open(cfg.pickle_path, "rb") as read_file:
        comparison_variables = pickle.load(read_file)

    transformations = comparison_variables["transformations"]
    base_map_filepath = comparison_variables["base_map_filepath"]
    comparison_map_filepath = comparison_variables["comparison_map_filepath"]

    # Visualise alignment
    output_video = "alignment_animation.mp4"
    visualiser = VisualiseAlignment(base_map_filepath, comparison_map_filepath)
    visualiser.create_video(transformations, output_video)
