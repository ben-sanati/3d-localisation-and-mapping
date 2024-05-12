import os
import cv2
import sqlite3
import numpy as np


class ImageExtractor:
    def __init__(self, db_path, image_size, save_dir):
        self.db_path = db_path
        self.image_size = image_size
        self.save_dir = save_dir
        self.image_dir = os.path.join(save_dir, 'images')
        self.depth_dir = os.path.join(save_dir, 'depth_images')
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
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)

    def fetch_data(self):
        """
        Fetch images along with calibration and pose data from the database.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT Data.image, Data.depth FROM Data JOIN Node ON Data.id = Node.id")
        for idx, (row) in enumerate(cursor.fetchall()):
            # Fetch the image data
            image = np.frombuffer(row[0], dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            image_path = os.path.join(self.image_dir, f"image_{idx}.jpg")
            cv2.imwrite(image_path, image)

            # Fetch the depth data
            depth = np.frombuffer(row[1], dtype=np.uint8)
            depth = cv2.imdecode(depth, cv2.IMREAD_UNCHANGED)
            depth = cv2.resize(depth, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            depth_path = os.path.join(self.depth_dir, f"depth_{idx}.png")
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
    extractor = ImageExtractor(db_path, img_size=640, save_dir="../common/data/gold_std/raw_img")
    data = extractor.fetch_data()
    extractor.view_images(data)
