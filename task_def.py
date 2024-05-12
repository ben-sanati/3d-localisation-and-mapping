import os
import gc
import sys
import pickle
from memory_profiler import profile

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


def extract_images(db_path):
    print("Extracting frames...", flush=True)
    extractor = ImageExtractor(db_path, img_size)
    images, depth_images = extractor.fetch_data()

    # Garbage collection
    del extractor
    gc.collect()
    print("Frames extracted.\n", flush=True)

    return images, depth_images

def detect_signs(images, img_size, batch_size, conf_thresh, iou_thresh):
    # Create dataset
    dataset = ImageDataset(images, img_size=img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Instance model
    print("Detecting Signs...", flush=True)
    model = ObjectDetector(
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        img_size=img_size,
        batch_size=batch_size,
        view_img=False,
        save_img="src/common/data/gold_std/processed_img",
    )

    # Run inference
    predictions = model(dataloader)

    # Perform garbage collection
    torch.cuda.empty_cache()
    del images
    del dataset
    del dataloader
    del model
    gc.collect()
    print("Inference Complete!\n", flush=True)

    return predictions

def map_detected_objects(pose_path, depth_images, predictions):
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
        depth_images=depth_images,
        bbox_coordinates=predictions,
        img_size=img_size,
    )
    global_bboxes_data = pose_processing.get_global_coordinates()

    for frame_index, bbox_list in global_bboxes_data.items():
        for bbox in bbox_list:
            print(f"{frame_index}: {bbox}")

    # Garbage collection
    del pose_df
    del pose_processing
    del depth_images
    gc.collect()
    print("Pose Processed!", flush=True)

    return global_bboxes_data

def plot_map(global_bboxes_data, eps, min_points, ply_path, preprocess_point_cloud):
    # Map the bounding box information to the global 3D map
    print("Generating 3D Map...")
    mapper = Mapping(
        global_bboxes_data=global_bboxes_data,
        eps=eps,
        min_points=min_points,
        ply_filepath=ply_path,
        preprocess_point_cloud=preprocess_point_cloud,
    )
    mapper.make_point_cloud()

    # Garbage collection
    del mapper
    gc.collect()
    print("3D Map Generated!", flush=True)


if __name__ == '__main__':
    img_size = 640
    batch_size = 2
    conf_thresh = 0.5
    iou_thresh = 0.65
    eps = 0.02
    min_points = 10
    preprocess_point_cloud = False

    db_path = r"src/common/data/gold_std/data.db"
    pose_path = r"src/common/data/gold_std/poses.txt"
    ply_path = r"src/common/data/gold_std/cloud.ply"

    # Extract images
    images, depth_images = extract_images(db_path)

    # Detecting signs
    predictions = detect_signs(images, img_size, batch_size, conf_thresh, iou_thresh)

    # Map detected objects
    global_bboxes_data = map_detected_objects(pose_path, depth_images, predictions)

    # Save as pickle file and load later to use in another script
    data_to_save = {
        "global_bboxes_data": global_bboxes_data,
        "eps": eps,
        "min_points": min_points,
        "ply_path": ply_path,
        "preprocess_point_cloud": preprocess_point_cloud,
    }
    with open("src/common/data/gold_std/variables.pkl", "wb") as file:
        pickle.dump(data_to_save, file)

    print("Variables stored to pickle file.", flush=True)

    # Plot 3D map
    # plot_map(global_bboxes_data, eps, min_points, ply_path, preprocess_point_cloud)
