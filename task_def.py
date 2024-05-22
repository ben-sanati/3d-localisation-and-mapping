import argparse
import gc
import os
import pickle
import sys

import torch
from src.detector.database_query import ImageExtractor
from src.detector.dataset import ImageDataset
from src.detector.detector import ObjectDetector
from src.mapper.database_query import PoseDataExtractor
from src.mapper.mapping import Mapping
from src.mapper.pose_processor import ProcessPose

from src.utils.config import ConfigLoader


sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        r"src",
        r"src/detector/yolov7",
    )
)


def extract_images(
    db_path, img_size, batch_size, image_dir, depth_image_dir, calibration_dir
):
    print("Extracting frames...", flush=True)
    extractor = ImageExtractor(db_path, depth_image_dir)
    extractor.fetch_data()

    # Create dataset
    dataset = ImageDataset(
        image_dir=image_dir,
        depth_image_dir=depth_image_dir,
        calibration_dir=calibration_dir,
        img_size=img_size,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Garbage collection
    del extractor
    gc.collect()
    print("Frames extracted.\n", flush=True)

    return dataset, dataloader


def detect_signs(
    dataloader, img_size, batch_size, conf_thresh, iou_thresh, view_img, processing_path
):
    # Instance model
    print("Detecting Signs...", flush=True)
    model = ObjectDetector(
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        img_size=img_size,
        batch_size=batch_size,
        view_img=view_img,
        save_img=processing_path,
    )

    # Run inference
    predictions = model(dataloader)

    # Perform garbage collection
    torch.cuda.empty_cache()
    del model
    gc.collect()
    print("Inference Complete!\n", flush=True)

    return predictions


def map_detected_objects(
    pose_path, dataset, predictions, img_size, depth_width, depth_height, display_3d
):
    # Get the node information from the table
    print("Extracting Pose Information...", flush=True)
    extractor = PoseDataExtractor(pose_path)
    pose_df = extractor.fetch_data()
    del extractor
    gc.collect()
    print("Pose Information Extracted!\n", flush=True)

    # Transform bbox coordinates to global coordinates
    print("Processing Pose", flush=True)
    pose_processing = ProcessPose(
        pose=pose_df,
        dataset=dataset,
        bbox_coordinates=predictions,
        img_size=img_size,
        depth_width=depth_width,
        depth_height=depth_height,
        display_3d=display_3d,
    )
    global_bboxes_data = pose_processing.get_global_coordinates()

    # Garbage collection
    del pose_processing
    gc.collect()
    print("Pose Processed!\n", flush=True)

    return global_bboxes_data, pose_df


def plot_map(
    global_bboxes_data,
    pose_df,
    eps,
    min_points,
    ply_path,
    preprocess_point_cloud,
    overlay_pose,
):
    # Map the bounding box information to the global 3D map
    print("Generating 3D Map...", flush=True)
    mapper = Mapping(
        global_bboxes_data=global_bboxes_data,
        pose=pose_df,
        eps=eps,
        min_points=min_points,
        ply_filepath=ply_path,
        preprocess_point_cloud=preprocess_point_cloud,
        overlay_pose=overlay_pose,
    )
    mapper.make_mesh()

    # Garbage collection
    del mapper
    gc.collect()
    print("3D Map Generated!", flush=True)


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

    data_to_save = {}

    # Extract images
    dataset, dataloader = extract_images(
        cfg.db_path,
        cfg.img_size,
        cfg.batch_size,
        cfg.image_dir,
        cfg.depth_image_dir,
        cfg.calibration_dir,
    )
    data_to_save["dataset"] = dataset
    data_to_save["dataloader"] = dataloader

    # Detecting signs
    predictions = detect_signs(
        dataloader,
        cfg.img_size,
        cfg.batch_size,
        cfg.conf_thresh,
        cfg.iou_thresh,
        cfg.view_img,
        cfg.processing_path,
    )
    data_to_save["predictions"] = predictions
    del dataloader
    gc.collect()

    # Map detected objects
    dataset = ImageDataset(
        image_dir=cfg.image_dir,
        depth_image_dir=cfg.depth_image_dir,
        calibration_dir=cfg.calibration_dir,
        img_size=cfg.img_size,
        processing=False,
    )
    global_bboxes_data, pose_df = map_detected_objects(
        cfg.pose_path,
        dataset,
        predictions,
        cfg.img_size,
        cfg.depth_width,
        cfg.depth_height,
        cfg.display_3d,
    )

    # Plot 3D Global Map
    plot_map(
        global_bboxes_data,
        pose_df,
        cfg.eps,
        cfg.min_points,
        cfg.ply_path,
        cfg.preprocess_point_cloud,
        cfg.overlay_pose,
    )

    # Save as pickle file and load later to use in another script
    data_to_save["global_bboxes_data"] = global_bboxes_data
    data_to_save["pose_df"] = pose_df

    try:
        with open(cfg.pickle_path, "wb") as file:
            pickle.dump(data_to_save, file)
            print("Variables stored to pickle file.", flush=True)
    except Exception as e:
        print(f"Failed to write to file: {e}")
