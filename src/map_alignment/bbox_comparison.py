import argparse
import os
import pickle
import sys
import random
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

sys.path.insert(0, r"../..")

from src.utils.config import ConfigLoader  # noqa
from src.utils.transformations import Transforms  # noqa
from src.utils.visualisation import Visualiser  # noqa


class BBoxComparison:
    def __init__(self, base_data, comparison_data, base_mesh, comparison_mesh):
        self.base_data = base_data
        self.comparison_data = comparison_data
        self.base_mesh = base_mesh
        self.comparison_mesh = comparison_mesh
        self.matches = {}

        # Instance util classes
        self.visualiser = Visualiser()
        self.transforms = Transforms()

    def match_bboxes(self):
        print(f"Base: {self.base_data}\n\nComparison: {self.comparison_data}")
