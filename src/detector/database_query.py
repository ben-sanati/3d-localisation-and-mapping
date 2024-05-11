import sqlite3
import cv2
import numpy as np


class ImageExtractor:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = self._connect_to_database()

    def _connect_to_database(self):
        """
        Connect to the SQLite database and return the connection.
        """
        return sqlite3.connect(self.db_path)

    def fetch_data(self):
        """
        Fetch images along with calibration and pose data from the database.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT Data.image, Data.calibration, Node.pose FROM Data JOIN Node ON Data.id = Node.id")
        data = []
        for row in cursor.fetchall():
            image_data = np.frombuffer(row[0], dtype=np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            data.append(image)

        return data

    @staticmethod
    def view_images(data):
        """
        View images in .db file from RTAB-MAP.

        Args:
            data (_type_): data acquired from fetch_data method
        """
        for idx, (image) in enumerate(data):
            # Display the image using OpenCV
            cv2.imshow('Image', image)
            cv2.waitKey(0)

            if idx >= 3:
                break


if __name__ == '__main__':
    db_path = '../common/data/gold_std/data.db'
    extractor = ImageExtractor(db_path)
    data = extractor.fetch_data()
    extractor.view_images(data)
