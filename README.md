# Localisation and Mapping of Signage on Train Vehicles - Innovate UK ATK Project

<div style="display: flex; justify-content: center;">
  <div style="text-align: center; margin: 10px;">
    <img src="readme_img/gold_std_map.png" width="450"/>
  </div>
  <div style="text-align: center; margin: 10px;">
    <img src="readme_img/seating_map.png" width="420"/>
  </div>
</div>
<p align="center">
  <em>Generated 3D map and signs localised in the global space.</em>
</p>



## Usage

### Data Folder Setup

1. **Create a New Folder**: Create a new folder with `<folder_name>` in the `src/common/data` directory.
2. **Place the .db File**: Put the `.db` file in this folder and name it `data.db`.
3. **Navigate to Root**: Change directory back to the root of the project.
4. **Run the Shell Script**:
    - Use `./run.sh --setup --data <folder_name>` if running for the first time.
    - Use `./run.sh --data <folder_name>` for subsequent runs.

<br />

> **Note**: If you use the `--setup` flag after the first time, it still works fine, you're just wasting time with setup.

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
