import os
import cv2
import sqlite3
import numpy as np


class ImageExtractor:
    def __init__(self, db_path, depth_dir):
        self.db_path = db_path
        self.depth_dir = depth_dir
        self._prepare_directories()

        self.conn = self._connect_to_database()

    def _connect_to_database(self):
        """
        Connect to the SQLite database and return the connection.
        """
        return sqlite3.connect(self.db_path)

    def _prepare_directories(self):
        """
        Ensure that the directories for storing images and depth images exist.
        """
        os.makedirs(self.depth_dir, exist_ok=True)

    def fetch_data(self):
        """
        Fetch depth images.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT Data.image, Data.depth FROM Data JOIN Node ON Data.id = Node.id")
        for i, (row) in enumerate(cursor.fetchall()):
            # Fetch the depth data
            idx = i + 1
            depth = np.frombuffer(row[1], dtype=np.uint8)
            depth = cv2.imdecode(depth, cv2.IMREAD_UNCHANGED)
            depth_path = os.path.join(self.depth_dir, f"{idx}.png")
            cv2.imwrite(depth_path, depth)

    @staticmethod
    def view_images(directory):
        """
        View images in .db file from RTAB-MAP.

        Args:
            data (_type_): data acquired from fetch_data method
        """
        for image_name in os.listdir(directory):
            image_path = os.path.join(directory, image_name)
            image = cv2.imread(image_path)
            cv2.imshow('Image', image)
            cv2.waitKey(0)
            break


if __name__ == '__main__':
    db_path = '../common/data/gold_std/data.db'
    extractor = ImageExtractor(db_path, depth_dir="../common/data/gold_std/db_extract/depth")
    data = extractor.fetch_data()
    extractor.view_images(data)
