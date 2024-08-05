# Localisation and Mapping of Signage on Train Vehicles - Innovate UK AKT Project

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) &emsp;
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) &emsp;
![VSCODE](https://img.shields.io/badge/Visual_Studio_Code-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white) &emsp;
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white) &emsp;
![iOS](https://img.shields.io/badge/iOS-000000?style=for-the-badge&logo=ios&logoColor=white)

## Table of Contents

- [Localisation and Mapping of Signage on Train Vehicles - Innovate UK AKT Project](#localisation-and-mapping-of-signage-on-train-vehicles---innovate-uk-akt-project)
  - [Bespoke Algorithms](#bespoke-algorithms)
  - [Usage](#usage)
  - [Folder Hierarchy](#folder-hierarchy)

This was an Innovate UK AKT project that was a collaboration between the University of Southampton and an industry client. The key goals of the project were as follows

- **Curating Gold-Standard and Maintenance Data**
  - Accomplished at a train depot, forming the basis for solution testing and validation.
- **Self-Localization, Detection, and Mapping on Mobile Devices**
  - Implemented using open-source software integrated into the backend processing pipeline, thoroughly tested for real-world performance.
- **Detection and Labelling in 3D Point Clouds**
  - Utilized advanced computer vision and machine learning techniques to accurately map signage within point clouds.
- **Prototyping Non-Conformance Detection in Vehicles**
  - Developed a system to identify missing and damaged signage within train vehicles, crucial for the auditing process.

<p align="center">
  <img src="readme_img/LiDARMap.gif" alt="Demo Video"/>
  <br>
  <em>RTAB-Map Generated LiDAR scan of vehicle interior.</em>
</p>

Beyond these core objectives, we delivered a strategic roadmap to guide our clients in transitioning to a data-driven organization. This roadmap included recommendations on integrating the developed technologies, adopting data-driven decision-making practices, and scaling product deployment for maximum impact.

The AKT project not only met its initial goals but also positioned our client for future growth and innovation by equipping them with the necessary tools, insights, and strategic direction to achieve their long-term objectives.

## Bespoke Algorithms

1. **2D Object Detection Bounding Box Transform to 3D Space** - this algorithm extends traditional 2D object detection by projecting the detected bounding boxes into 3D space. It converts 2D pixel coordinates from the depth image to 3D space coordinates such that it can estimate the real-world dimensions and positions of objects. This is performed in `src/utils/transformations.py`, within the `BBoxTransforms` class (`_depth_to_3d` method).

<table style="width: 100%;">
  <tr>
    <td align="center" style="width: 50%;">
      <img id="firstImage" src="readme_img/gold_std_map.png" style="width: 100%; height: auto;"/>
    </td>
    <td align="center" style="width: 50%;">
      <img id="secondImage" src="readme_img/seating_map.png" style="width: 100%; height: auto;"/>
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <em>Processed 3D and signs localized in the global space.</em>
    </td>
  </tr>
</table>

2. **Map Alignment Algorithm** - the map alignment algorithm is designed to accurately align the gold-standard and maintenance map representations. It employs image processing and machine learning techniques to perform the alignment. This is performed in `src/map_alignment/align.py`.

<p align="center">
  <img src="readme_img/alignment_visualisation.gif" alt="Demo Video" style="width: 60%"/>
  <br>
  <em>Alignment of 2 separate 3D point clouds using our algorithm.</em>
</p>

3. **Bounding Box Bipartite Matching for Missing Box Identification** - this algorithm addresses the problem of identifying missing signs in the maintenance scan compared to the gold-standard scan. By using a bipartite matching approach, it pairs detected bounding boxes from the maintenance scan to the gold-standard scans detected bounding boxes. This is performed in `src/map_alignment/comparison.py`.

<p align="center">
  <img src="readme_img/bp_match_algo.png" style="width: 60%"/>
  <br>
  <em>Bounding box matching of gold-standard signage (red) to maintenance signage (green).</em>
</p>

## Usage

All Python dependencies are listed in the `requirements.txt` file. Install with `pip install -r requirements.txt` while in the root directory.

### Data Folder Setup & Run

1. **Create a New Folder**: Create a new folder with `<folder_name>` in the `src/common/data` directory.
2. **Place the .db File**: Put the `.db` file in this folder and name it `data.db`.
3. **Navigate to Root**: Change directory back to the root of the project.
4. **Run the Shell Script**:
    - Use `./run.sh --setup --data <folder_name>` if running the particular `folder_name` for the first time.
    - Use `./run.sh --data <folder_name>` for subsequent runs.

> **Note**: If you use the `--setup` flag after the first time, it still works fine, you're just wasting time with setup.

> **Note**: There should never be a need to run the `gold_std` folder name as this should run automatically if it has not been done before (you still can if you wish).

<details>
  <summary>Using the RTAB-Map GUI Instead</summary>

    You can also use the RTAB-Map GUI to do this manually.

    cd src/common/data/<folder_name>

    Extract Point Cloud
    -------------------------------
    rtabmap-databaseViewer data.db
    Yes
    File -> Export 3D map
    Save

    Extract Pose
    -------------------------------
    File -> Export Poses
    Maps graph (see Graph View)
    Camera

</details>

---

## Folder Hierarchy


```
.
├── run.sh
└── task_def.py
├── src
│   ├── common
│   │   ├── configs
│   │   │   └── variables.cfg
│   │   ├── data
│   │   │   ├── gold_std
│   │   │   ├── ideal_scan
│   │   │   ├── quick_a
│   │   │   ├── quick_b
│   │   │   └── setup.py
│   │   ├── finetuned_models
│   │   └── results
│   ├── damage
│   ├── detector
│   ├── map_alignment
│   ├── mapper
│   └── utils
```

- `task_def.py` - contains both the pipeline definition and the main file used by `run.sh`
- `src` - contains all relevant packages utulised by the program
  - `common` - this contains all files used by the entire program, including:
    - Config files
    - Data folders for each scan
    - Trained computer vision models
    - Results from program processing
  - `damage` - the damage detection inference module
  - `detector` - the object detection inference module
  - `map_alignment` - the alignment of the maintenance map with respect to the gold-std scan in 3D space
  - `mapper` - visualisation of a single 3D map with detected signs overlayed
  - `utils` - a general utilities package used by the entire program, including transformations, visualisations, and more.

  ---