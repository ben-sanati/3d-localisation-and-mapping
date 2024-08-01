import argparse
import os
import pickle
import logging
import sys

from src.detector.database_query import ImageExtractor
from src.detector.dataset import ImageDataset
from src.detector.detector import ObjectDetector
from src.mapper.bbox_optimiser import BoundingBoxProcessor
from src.mapper.database_query import PoseDataExtractor
from src.mapper.mapping import Mapping
from src.map_alignment.align import Alignment
from src.map_alignment.comparison import BBoxComparison
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
    def __init__(self, data_folder, cfg, cfg_goldstd, goldstd_var=None):
        self.cfg = cfg
        self.cfg_goldstd = cfg_goldstd
        self.data_folder = data_folder
        self.goldstd_var = goldstd_var
        self.data_to_save = {}

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def run(self):
        # Extract images
        dataset, dataloader = self._extract_images()

        # Detecting signs
        predictions = self._detect_signs(dataloader)

        # Map detected objects
        dataset = ImageDataset(
            image_dir=self.cfg.image_dir,
            depth_image_dir=self.cfg.depth_image_dir,
            calibration_dir=self.cfg.calibration_dir,
            img_size=self.cfg.img_size,
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
            with open(self.cfg.pickle_path, "wb") as file:
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

        # Compare maps if Gold-Std. is given in (proxy for setup being already completed)
        if self.cfg_goldstd:
            self._goldstd_vs_maintenance(pose_df, optimised_bboxes)

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
            img_size=self.cfg.img_size,
            batch_size=self.cfg.batch_size,
            view_img=self.cfg.view_img,
            save_img=self.cfg.processing_path,
            data_root=self.cfg.image_dir,
        )

        # Run inference
        predictions = model()
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
            display_3d=self.cfg.display_3d_pose,
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

    def _goldstd_vs_maintenance(self, maintenance_pose_df, maintenance_optimised_bboxes):
        # Align bboxes from maintenance scan onto the gold-std scan for comparison
        map_alignment = Alignment(
            base_pose_df=self.goldstd_var["pose_df"],
            comparison_pose_df=maintenance_pose_df,
            base_bboxes=self.goldstd_var["optimised_bboxes"],
            comparison_bboxes=maintenance_optimised_bboxes,
            visualise=self.cfg.alignment_vis,
        )
        (
            aligned_maintenance_bboxes,
            _,
            goldstd_mesh,
            _,
        ) = map_alignment.compare(self.data_folder)

        # Compare the bboxes and output results to a csv file
        compare_bboxes = BBoxComparison(
            self.goldstd_var["optimised_bboxes"],
            aligned_maintenance_bboxes,
            goldstd_mesh,
            visualise=self.cfg.comparison_vis,
            csv_output_file=self.cfg.csv_output,
        )
        compare_bboxes.match_bboxes()


def load_gold_std(pickle_path):
    try:
        with open(pickle_path, "rb") as read_file:
            return pickle.load(read_file)
    except FileNotFoundError:
        logging.error(f"The file {pickle_path} was not found.")
        return None
    except pickle.UnpicklingError:
        logging.error(f"Failed to unpickle the file {pickle_path}.")
        return None

def setup_pipeline(data_folder, cfg, cfg_goldstd, goldstd_var=None):
    pipeline = Pipeline(data_folder, cfg, cfg_goldstd, goldstd_var)
    pipeline.run()
    return pipeline


if __name__ == "__main__":
    # Setup argparse config
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Processing Configuration")
    parser.add_argument(
        "--data", type=str, help="Data Folder Name.", default="gold_std"
    )
    args = parser.parse_args()
    data_folder = args.data

    # Load the configs
    config_path = r"src/common/configs/variables.cfg"
    cfg = ConfigLoader(config_path, data_folder)
    cfg_goldstd = ConfigLoader(config_path, "gold_std")

    # Automated setup check logic
    if data_folder == "gold_std":
        setup_pipeline(data_folder, cfg_goldstd, None)
    else:
        # Make sure gold-std setup is done
        if os.path.exists(cfg_goldstd.pickle_path) == False:
            # We first have to run the setup with Gold-Std. before run
            logging.info("Performing setup before maintenance check.")
            setup_pipeline(data_folder, cfg_goldstd, None)

        # Fetch stored variables
        goldstd_var = load_gold_std(cfg_goldstd.pickle_path)
        logging.info("Fetched Gold-Std. Data.")

        # We can perform a maintenance run
        logging.info("Executing maintenance check.")
        setup_pipeline(data_folder, cfg, cfg_goldstd, goldstd_var=goldstd_var)
