import logging
import sys
import csv
import random
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

sys.path.insert(0, r"../..")

from src.utils.config import ConfigLoader  # noqa
from src.utils.transformations import Transforms  # noqa
from src.utils.visualisation import Visualiser  # noqa


class BBoxComparison:
    def __init__(
        self,
        base_bboxes,
        comparison_bboxes,
        base_mesh,
        comparison_mesh,
        bbox_depth_buffer=0.05,
        area_threshold=1e-2, # 1cm^2 diff ~ 10cm difference in bbox lengths
        k_closest=5,
        visualise=False,
        output_file="src/common/results/debug.csv",
    ):
        self.base_bboxes = base_bboxes
        self.comparison_bboxes = comparison_bboxes
        self.base_mesh = base_mesh
        self.comparison_mesh = comparison_mesh
        self.bbox_depth_buffer = bbox_depth_buffer
        self.area_threshold = area_threshold
        self.k_closest = k_closest
        self.visualise = visualise
        self.output_file = output_file

        self.matches = {}
        self.color_map = {}
        self.matched_base_indices = set()

        # Instance util classes
        self.visualiser = Visualiser()
        self.transforms = Transforms()

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Comparison class initialized")

        # Set logging verbosity
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    def _calculate_centroid(self, bbox):
        # Calculate the centroid of a bbox
        return np.mean(bbox[:4], axis=0)

    def _calculate_area(self, bbox):
        # Calculate the area of the bbox assuming it is a rectangle
        p1, p2, p3, p4 = bbox[:4]
        width = np.linalg.norm(p1 - p2)
        height = np.linalg.norm(p2 - p3)
        return width * height

    def match_bboxes(self):
        self.logger.info("Matching bounding boxes")

        # Collect all base bboxes in a list
        base_bbox_list = [bbox for bbox_list in self.base_bboxes.values() for bbox in bbox_list]

        # Initialize KDTree for fast spatial matching
        base_centroids = [self._calculate_centroid(bbox) for bbox in base_bbox_list]
        base_tree = KDTree(base_centroids)

        for frame_id, comp_bbox_list in self.comparison_bboxes.items():
            for comp_bbox in comp_bbox_list:
                comp_centroid = self._calculate_centroid(comp_bbox)
                distances, indices = base_tree.query(comp_centroid, k=self.k_closest)
                
                # Find the best match among the k nearest neighbors
                best_match = None
                for i in range(self.k_closest):
                    base_bbox = base_bbox_list[indices[i]]
                    if (
                        comp_bbox[-1] == base_bbox[-1]
                        and abs(self._calculate_area(comp_bbox) - self._calculate_area(base_bbox))
                        < self.area_threshold
                    ):
                        best_match = (
                            indices[i],
                            (base_bbox[-1], tuple(base_centroids[indices[i]]))  # Use tuple for hashable key
                        )
                        self.matches[
                            (frame_id, (comp_bbox[-1], tuple(comp_centroid)))  # Use tuple for hashable key
                        ] = best_match
                        self.matched_base_indices.add(indices[i])
                        break

        self.logger.info(f"Matches: {self.matches}")

        # Identify unmatched base bboxes
        unmatched_base_bboxes = [
            {
                'base_classification': base_bbox_list[i][-1],
                'base_coordinates': self._calculate_centroid(base_bbox_list[i])
            }
            for i in range(len(base_bbox_list)) if i not in self.matched_base_indices
        ]
        self.logger.info(f"Unmatched base bboxes: {unmatched_base_bboxes}")

        # Output results to csv
        self.generate_csv()

        # Visualise the bboxes and meshes
        if self.visualise:
            self._visualise_bboxes()

        return self.matches, unmatched_base_bboxes

    def _visualise_bboxes(self):
        self.logger.info("Visualizing bounding boxes and meshes")
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.add_geometry(self.base_mesh)
        # vis.add_geometry(self.comparison_mesh)

        # Add base bboxes
        for frame_id, bbox_list in self.base_bboxes.items():
            for bbox in bbox_list:
                vertices = bbox[:4]
                classification = bbox[-1]
                color = self._get_colour_by_classification(classification)
                bbox_3d = self.transforms.create_3d_bounding_box(vertices, self.bbox_depth_buffer)
                bbox_line_set = self.visualiser.overlay_3d_bbox(bbox_3d, colour=color)
                vis.add_geometry(bbox_line_set)

        # Add comparison bboxes
        for frame_id, bbox_list in self.comparison_bboxes.items():
            for bbox in bbox_list:
                vertices = bbox[:4]
                classification = bbox[-1]
                color = self._get_colour_by_classification(classification)
                bbox_3d = self.transforms.create_3d_bounding_box(vertices, self.bbox_depth_buffer)
                bbox_line_set = self.visualiser.overlay_3d_bbox(bbox_3d, colour=color)
                vis.add_geometry(bbox_line_set)

        # Draw lines connecting matched bboxes and add spheres at centroids
        for (comp_frame_id, comp_data), (base_index, base_data) in self.matches.items():
            comp_classification, comp_centroid = comp_data
            base_classification, base_centroid = base_data
            
            # Ensure the centroids are numpy arrays
            comp_centroid = np.array(comp_centroid)
            base_centroid = np.array(base_centroid)
            
            # Line connecting centroids
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector([comp_centroid, base_centroid]),
                lines=o3d.utility.Vector2iVector([[0, 1]]),
            )
            line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1]])  # Blue color for lines
            vis.add_geometry(line_set)

            # Sphere at comparison centroid
            sphere_comp = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
            sphere_comp.translate(comp_centroid)
            sphere_comp.paint_uniform_color([0, 1, 0])  # Green color for comparison centroid
            vis.add_geometry(sphere_comp)

            # Sphere at base centroid
            sphere_base = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
            sphere_base.translate(base_centroid)
            sphere_base.paint_uniform_color([1, 0, 0])  # Red color for base centroid
            vis.add_geometry(sphere_base)

        vis.run()
        vis.destroy_window()

    def _get_colour_by_classification(self, classification):
        # Generate random color if not already defined
        if classification not in self.color_map:
            self.color_map[classification] = [random.random() for _ in range(3)]
        return self.color_map[classification]

    def generate_csv(self):
        """
        Generate a CSV file with columns for base bboxes and matched bboxes.
        """
        base_bboxes_data = [
            {
                'classification': bbox[-1],
                'coordinates': tuple(self._calculate_centroid(bbox))
            }
            for bbox_list in self.base_bboxes.values() for bbox in bbox_list
        ]
        
        matched_bboxes_data = [
            {
                'classification': self.matches[(comp_frame_id, comp_data)][1][0],
                'coordinates': tuple(self.matches[(comp_frame_id, comp_data)][1][1])
            }
            if (comp_frame_id, comp_data) in self.matches else None
            for comp_frame_id, comp_bbox_list in self.comparison_bboxes.items()
            for comp_data in [(bbox[-1], tuple(self._calculate_centroid(bbox))) for bbox in comp_bbox_list]
        ]

        with open(self.output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Signs', 'Matched Signs'])

            for base_data, matched_data in zip(base_bboxes_data, matched_bboxes_data):
                base_entry = f"{base_data['classification']} - {base_data['coordinates']}"
                matched_entry = f"{matched_data['classification']} - {matched_data['coordinates']}" if matched_data else ''
                writer.writerow([base_entry, matched_entry])

        self.logger.info(f"CSV file generated at {self.output_file}")
