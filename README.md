# 3D-Mapping-ATK

## Usage

### Data Folder Setup

Once you have created your `.db` file, create a new folder in the `data` directory and put the `.db` file in it, renaming the file as `data.db`. Then, perform the following steps.

<details>
    <summary>Extract Data from RTAB-Map</summary>

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

<details>
    <summary>Run Setup Script</summary>

    cd src/common/data
    python3 setup.py --data <folder_name>

</details>

---