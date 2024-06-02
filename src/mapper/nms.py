import os
import sys
import pickle
import argparse
import numpy as np
from collections import Counter
from itertools import combinations

sys.path.insert(0, r"../..")

from src.utils.config import ConfigLoader  # noqa
from src.utils.transformations import VisualisationTransforms  # noqa


class BoundingBoxProcessor:
    def __init__(self, global_bboxes, conf_th=0.5, iou_th=0.5, bbox_depth_buffer=0.03,):
        self.global_bboxes = global_bboxes
        self.conf_th = conf_th
        self.iou_th = iou_th
        self.bbox_depth_buffer = bbox_depth_buffer
        
        self.transforms = VisualisationTransforms()

    def suppress_bboxes(self):
        # 1. Calculate the new confidence for each bbox
        new_bboxes = self._cube_likeness()

        # 2. Remove bboxes with confidence < threshold
        # 3. Perform 3D NMS on new bboxes
        return new_bboxes

    def _cube_likeness(self):
        new_bboxes = {}
        for key in self.global_bboxes:
            for item in self.global_bboxes[key]:
                coords = item[:4]
                cube_coords = self.transforms.create_3d_bounding_box(
                    coords, self.bbox_depth_buffer
                )
                score = self._calculate_rectangular_cuboid_likeness(cube_coords)
                if score > 0.9:
                    new_bboxes[key] = self.global_bboxes[key]
                else:
                    print(f"Removed BBox")
                print(f"Cube-likeness score for {cube_coords}: {score:.2f}")

        return new_bboxes

    def _calculate_rectangular_cuboid_likeness(self, coords, tolerance=0.2):
        edge_lengths = []

        for (i, j) in combinations(range(4), 2):
            dist = np.linalg.norm(coords[i] - coords[j])
            edge_lengths.append(dist)

        if len(edge_lengths) != 6:
            raise ValueError("Incorrect number of edges. Expected 6 edges.")

        # Check how parallel the edges are
        direction_vectors = self._calculate_direction_vectors(coords)
        parallelism_score = self._calculate_parallelism(direction_vectors)
        print(f"Parallelism: {parallelism_score}")

        rectangular_cuboid_likeness_score = parallelism_score
        return rectangular_cuboid_likeness_score

    @staticmethod
    def _calculate_direction_vectors(vertices):
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0), # Bottom face edges
            (4, 5), (5, 6), (6, 7), (7, 4), # Top face edges
            (0, 4), (1, 5), (2, 6), (3, 7)  # Vertical edges
        ]
        direction_vectors = []
        for edge in edges:
            v1, v2 = edge
            direction_vectors.append(vertices[v2] - vertices[v1])
        return np.array(direction_vectors)

    def _calculate_parallelism(self, vectors):
        parallel_pairs = [
            (0, 2), (1, 3), (4, 6), (5, 7), # Pairs of vectors on the bottom and top faces
            (0, 4), (1, 5), (2, 6), (3, 7)  # Vertical pairs
        ]
        cosines = []
        for pair in parallel_pairs:
            v1, v2 = vectors[pair[0]], vectors[pair[1]]
            dot_product = np.dot(v1, v2)
            cos_angle = dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cosines.append(abs(cos_angle)) # Use absolute value to handle parallel vectors in opposite directions
        return np.mean(cosines)

    def _calculate_mean_relative_deviation(self, measured_lengths, expected_length):
        deviations = [abs(length - expected_length) / expected_length for length in measured_lengths]
        return np.mean(deviations)


if __name__ == "__main__":
    # Setup argparse config
    parser = argparse.ArgumentParser(description="Processing Configuration")
    parser.add_argument(
        "--data", type=str, help="Data Folder Name.", default="gold_std"
    )
    args = parser.parse_args()
    data_folder = args.data

    # Load the configuration
    os.chdir("../..")
    config_path = r"src/common/configs/variables.cfg"
    cfg = ConfigLoader(config_path, data_folder)

    with open(cfg.pickle_path, "rb") as read_file:
        variables = pickle.load(read_file)

    pose_df = variables["pose_df"]
    global_bboxes_data = variables["global_bboxes_data"]

    optimise_bboxes = BoundingBoxProcessor(global_bboxes_data)
    optimised_bboxes = optimise_bboxes.suppress_bboxes()

    # Save to pickle file
    data_to_save = {
        "global_bboxes_data": global_bboxes_data,
        "pose_df": pose_df,
        "optimised_bboxes": optimised_bboxes,
    }

    with open(cfg.pickle_path, "wb") as write_file:
        pickle.dump(data_to_save, write_file)
        print("Variables stored to pickle file.", flush=True)
