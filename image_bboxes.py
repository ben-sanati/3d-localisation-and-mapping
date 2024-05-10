import sqlite3
import cv2
import numpy as np

def connect_to_database(db_path):
    """ Connect to the SQLite database and return the connection. """
    return sqlite3.connect(db_path)

def fetch_images_and_data(conn):
    """ Fetch images along with calibration and pose data from the database. """
    cursor = conn.cursor()
    cursor.execute("SELECT Data.image, Data.calibration, Node.pose FROM Data JOIN Node ON Data.id = Node.id")
    data = []
    for row in cursor.fetchall():
        image_data = np.frombuffer(row[0], dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        calibration = row[1]
        pose = row[2]
        data.append((image, calibration, pose))
    return data

def detect_and_annotate_images(data, detector):
    """ Run object detection and annotate images. """
    annotated_images = []
    for image, calibration, pose in data:
        boxes = detector.detect(image)  # Your detector method
        for (x, y, w, h) in boxes:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        annotated_images.append(image)
    return annotated_images

def view_images(data):
    for index, (image, calibration, pose) in enumerate(data):
        # Display the image using OpenCV
        cv2.imshow('Image', image)

        # Wait for a key press and close the window
        filename = f'src/common/out/content/image_{index}.png'
        cv2.imwrite(filename, image)

if __name__ == '__main__':
    db_path = 'src/common/data/gold_std/data.db'
    conn = connect_to_database(db_path)
    data = fetch_images_and_data(conn)
    view_images(data)
