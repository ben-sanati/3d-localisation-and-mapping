import argparse
import gc
import os
import pickle
import logging
import sys

import torch
from src.detector.database_query import ImageExtractor
from src.detector.dataset import ImageDataset
from src.detector.detector import ObjectDetector
from src.mapper.bbox_optimiser import BoundingBoxProcessor
from src.mapper.database_query import PoseDataExtractor
from src.mapper.mapping import Mapping
from src.mapper.pose_processor import ProcessPose
from src.utils.config import ConfigLoader
from torch.utils.data import DataLoader

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        r"src",
        r"src/detector/yolov7",
    )
)


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_to_save = {}

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def run(self):
        # TODO: integrate bbox comparison methods into pipeline
        # TODO: define pipeline for gold_std vs. maintenance runs

        # Extract images
        dataset, dataloader = self._extract_images()

        # Detecting signs
        predictions = self._detect_signs(dataloader)

        # Map detected objects
        dataset = ImageDataset(
            image_dir=cfg.image_dir,
            depth_image_dir=cfg.depth_image_dir,
            calibration_dir=cfg.calibration_dir,
            img_size=cfg.img_size,
            processing=False,
        )
        global_bboxes_data, optimised_bboxes, pose_df = self._map_detected_objects(
            predictions,
            dataset,
        )

        # Save as pickle file and load later to use in another script 
        # Useful during development
        self.data_to_save["dataset"] = dataset
        self.data_to_save["dataloader"] = dataloader
        self.data_to_save["predictions"] = predictions
        self.data_to_save["global_bboxes_data"] = global_bboxes_data
        self.data_to_save["optimised_bboxes"] = optimised_bboxes
        self.data_to_save["pose_df"] = pose_df

        try:
            with open(cfg.pickle_path, "wb") as file:
                pickle.dump(self.data_to_save, file)
                self.logger.info("Variables stored to pickle file.")
        except Exception as e:
            self.logger.info(f"Failed to write to file: {e}")

        # Plot 3D Global Map
        if self.cfg.visualise:
            self._plot_map(
                global_bboxes_data,
                optimised_bboxes,
                pose_df,
            )

    def _extract_images(self):
        self.logger.info("Extracting frames...")
        extractor = ImageExtractor(self.cfg.db_path, self.cfg.depth_image_dir)
        extractor.fetch_data()

        # Create dataset
        dataset = ImageDataset(
            image_dir=self.cfg.image_dir,
            depth_image_dir=self.cfg.depth_image_dir,
            calibration_dir=self.cfg.calibration_dir,
            img_size=self.cfg.img_size,
        )
        dataloader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=False)
        self.logger.info("Frames extracted.")

        return dataset, dataloader

    def _detect_signs(self, dataloader):
        # Instance model
        self.logger.info("Detecting Signs...")
        model = ObjectDetector(
            conf_thresh=self.cfg.conf_thresh,
            iou_thresh=self.cfg.iou_thresh,
            img_size=self.cfg.img_size,
            batch_size=self.cfg.batch_size,
            view_img=self.cfg.view_img,
            save_img=self.cfg.processing_path,
        )

        # Run inference
        predictions = model(dataloader)
        self.logger.info("Inference Complete.")

        return predictions

    def _map_detected_objects(self, predictions, dataset):
        # Get the node information from the table
        self.logger.info("Extracting Pose Information...")
        extractor = PoseDataExtractor(self.cfg.pose_path)
        pose_df = extractor.fetch_data()
        self.logger.info("Pose Information Extracted.")

        # Transform bbox coordinates to global coordinates
        self.logger.info("Processing Pose...")
        pose_processing = ProcessPose(
            pose=pose_df,
            dataset=dataset,
            bbox_coordinates=predictions,
            img_size=self.cfg.img_size,
            depth_width=self.cfg.depth_width,
            depth_height=self.cfg.depth_height,
            display_3d=self.cfg.display_3d,
        )
        global_bboxes_data = pose_processing.get_global_coordinates()
        self.logger.info("Pose Processed.")

        # Perform 3D NMS
        self.logger.info("Executing 3D NMS...")
        optimise_bboxes = BoundingBoxProcessor(global_bboxes_data, pose_df)
        optimised_bboxes = optimise_bboxes.suppress_bboxes()
        self.logger.info("3D NMS Executed.")

        return global_bboxes_data, optimised_bboxes, pose_df

    def _plot_map(self, global_bboxes_data, optimised_bboxes, pose_df):
        # Map the bounding box information to the global 3D map
        self.logger.info("Generating 3D Map...")
        mapper = Mapping(
            global_bboxes_data=global_bboxes_data,
            optimised_bboxes=optimised_bboxes,
            pose=pose_df,
            eps=self.cfg.eps,
            min_points=self.cfg.min_points,
            ply_filepath=self.cfg.ply_path,
            preprocess_point_cloud=self.cfg.preprocess_point_cloud,
            overlay_pose=self.cfg.overlay_pose,
        )
        mapper.make_mesh()
        self.logger.info("3D Map Generated.")

    def _goldstd_vs_maintenance(self):
        pass


if __name__ == "__main__":
    # Setup argparse config
    parser = argparse.ArgumentParser(description="Processing Configuration")
    parser.add_argument(
        "--data", type=str, help="Data Folder Name.", default="gold_std"
    )
    args = parser.parse_args()
    data_folder = args.data

    # Load the configuration
    config_path = r"src/common/configs/variables.cfg"
    cfg = ConfigLoader(config_path, data_folder)

    pipeline = Pipeline(cfg)
    pipeline.run()
