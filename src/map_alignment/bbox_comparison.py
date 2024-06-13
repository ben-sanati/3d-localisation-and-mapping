import argparse
import os
import pickle
import sys

sys.path.insert(0, r"../..")

from src.utils.config import ConfigLoader  # noqa


class BBoxComparison:
    def __init__(self, base_data, comparison_data):
        self.base_data = base_data
        self.comparison_data = comparison_data

    def match_bboxes(self):
        print(f"Base: {self.base_data}\n\nComparison: {self.comparison_data}")


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

    with open(cfg.pickle_path, "rb") as read_file:
        comparison_variables = pickle.load(read_file)

    print(comparison_variables)

    base_bboxes = comparison_variables["base_bboxes"]
    comparison_bboxes = comparison_variables["comparison_bboxes"]
