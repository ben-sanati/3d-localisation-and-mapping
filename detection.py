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


def extract_images(db_path, img_size, batch_size, save_dir, image_dir, depth_image_dir):
    # print("Extracting frames...", flush=True)
    # extractor = ImageExtractor(db_path, img_size, save_dir)
    # extractor.fetch_data()

    # Create dataset
    dataset = ImageDataset(image_dir=image_dir, depth_image_dir=depth_image_dir, img_size=img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Garbage collection
    # del extractor
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
        view_img=False,
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

def map_detected_objects(pose_path, dataset, predictions, img_size):
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
    )
    global_bboxes_data = pose_processing.get_global_coordinates()

    for frame_index, bbox_list in global_bboxes_data.items():
        for bbox in bbox_list:
            print(f"{frame_index}: {bbox}")

    # Garbage collection
    del pose_processing
    gc.collect()
    print("Pose Processed!", flush=True)

    return global_bboxes_data, pose_df

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
    eps = 0.02
    img_size = 640
    batch_size = 2
    min_points = 10
    conf_thresh = 0.5
    iou_thresh = 0.65
    preprocess_point_cloud = False

    db_path = r"src/common/data/gold_std/data.db"
    ply_path = r"src/common/data/gold_std/cloud.ply"
    pose_path = r"src/common/data/gold_std/poses.txt"
    pickle_path = r"src/common/data/gold_std/variables.pkl"

    save_dir = r"src/common/data/gold_std/rtabmap_extract"
    image_dir = f"{save_dir}/rgb"
    depth_image_dir = f"{save_dir}/depth"

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
    global_bboxes_data, pose_df = map_detected_objects(pose_path, dataset, predictions, img_size)

    # # Save as pickle file and load later to use in another script
    data_to_save["global_bboxes_data"] = global_bboxes_data
    data_to_save["pose_df"] = pose_df

    try:
        with open(pickle_path, "wb") as file:
            pickle.dump(data_to_save, file)
            print("Variables stored to pickle file.", flush=True)
    except Exception as e:
        print(f"Failed to write to file: {e}")
