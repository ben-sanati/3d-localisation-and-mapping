import os
import gc
import sys
import pickle
import configparser

import torch
from torch.utils.data import DataLoader

from src.detector.database_query import ImageExtractor
from src.detector.dataset import ImageDataset
from src.detector.detector import ObjectDetector

from src.mapper.mapping import Mapping
from src.mapper.pose_processor import ProcessPose
from src.mapper.database_query import PoseDataExtractor


sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        r"src",
        r"src/detector/yolov7",
    )
)


def load_config(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config

def extract_images(db_path, img_size, batch_size, save_dir, image_dir, depth_image_dir):
    print("Extracting frames...", flush=True)
    extractor = ImageExtractor(db_path, depth_image_dir)
    extractor.fetch_data()

    # Create dataset
    dataset = ImageDataset(image_dir=image_dir, depth_image_dir=depth_image_dir, calibration_dir=calibration_dir, img_size=img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Garbage collection
    del extractor
    gc.collect()
    print("Frames extracted.\n", flush=True)

    return dataset, dataloader

def detect_signs(dataloader, img_size, batch_size, conf_thresh, iou_thresh):
    # Instance model
    print("Detecting Signs...", flush=True)
    model = ObjectDetector(
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        img_size=img_size,
        batch_size=batch_size,
        view_img=True,
        save_img="src/common/data/gold_std/processed_img",
    )

    # Run inference
    predictions = model(dataloader)

    # Perform garbage collection
    torch.cuda.empty_cache()
    del model
    gc.collect()
    print("Inference Complete!\n", flush=True)

    return predictions

def map_detected_objects(pose_path, dataset, predictions, img_size, depth_width, depth_height):
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
    )
    global_bboxes_data = pose_processing.get_global_coordinates()

    # Garbage collection
    del pose_processing
    gc.collect()
    print("Pose Processed!", flush=True)

    return global_bboxes_data, pose_df

def plot_map(global_bboxes_data, pose_df, eps, min_points, ply_path, preprocess_point_cloud, overlay_pose):
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


if __name__ == '__main__':
    # Load the configuration
    config_path = r"src/common/configs/variables.cfg"
    config = load_config(config_path)

    # Access configuration variables from the 'detection' section
    img_size = config.getint('detection', 'img_size')
    batch_size = config.getint('detection', 'batch_size')
    conf_thresh = config.getfloat('detection', 'conf_thresh')
    iou_thresh = config.getfloat('detection', 'iou_thresh')

    # Access configuration variables from the 'mapping' section
    eps = float(config['mapping']['eps'])
    min_points = config.getint('mapping', 'min_points')
    preprocess_point_cloud = config.getboolean('mapping', 'preprocess_point_cloud')
    overlay_pose = config.getboolean('mapping', 'overlay_pose')
    depth_width = config.getint('mapping', 'depth_width')
    depth_height = config.getint('mapping', 'depth_height')

    # Access paths from the 'paths' section
    db_path = config['paths']['db_path']
    ply_path = config['paths']['ply_path']
    pose_path = config['paths']['pose_path']
    pickle_path = config['paths']['pickle_path']
    save_dir = config['paths']['save_dir']
    image_dir = config['paths']['image_dir']
    depth_image_dir = config['paths']['depth_image_dir']
    calibration_dir = config['paths']['calibration_dir']

    data_to_save = {}

    # Extract images
    dataset, dataloader = extract_images(db_path, img_size, batch_size, save_dir, image_dir, depth_image_dir)
    data_to_save["dataset"] = dataset
    data_to_save["dataloader"] = dataloader

    # Detecting signs
    predictions = detect_signs(dataloader, img_size, batch_size, conf_thresh, iou_thresh)
    data_to_save["predictions"] = predictions
    del dataloader
    gc.collect()

    # Map detected objects
    dataset = ImageDataset(image_dir=image_dir, depth_image_dir=depth_image_dir, calibration_dir=calibration_dir, img_size=img_size, processing=False)
    global_bboxes_data, pose_df = map_detected_objects(pose_path, dataset, predictions, img_size, depth_width, depth_height)

    # Plot 3D Global Map (RAM runs out so save as pickle file and run independently instead)
    # plot_map(global_bboxes_data, pose_df, eps, min_points, ply_path, preprocess_point_cloud, overlay_pose)

    # Save as pickle file and load later to use in another script
    data_to_save["global_bboxes_data"] = global_bboxes_data
    data_to_save["pose_df"] = pose_df

    try:
        with open(pickle_path, "wb") as file:
            pickle.dump(data_to_save, file)
            print("Variables stored to pickle file.", flush=True)
    except Exception as e:
        print(f"Failed to write to file: {e}")
