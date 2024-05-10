import os
import sys

from torch.utils.data import DataLoader

from src.detector.image_bboxes import ImageExtractor
from src.detector.dataset import ImageDataset
from src.detector.detector import ObjectDetector

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        r"src",
        r"src/detector/yolov7",
    )
)


if __name__ == '__main__':
    img_size = 640
    batch_size = 16
    conf_thresh = 0.5
    iou_thresh = 0.65

    # ################# #
    # Extracting Images #
    # ################# #

    print("Extracting frames...", flush=True)
    db_path = r"src/common/data/gold_std/data.db"
    extractor = ImageExtractor(db_path)
    data = extractor.fetch_data()
    print("Frames extracted.\n", flush=True)

    # ############### #
    # Detecting Signs #
    # ############### #

    # Create dataset
    dataset = ImageDataset(data, img_size=img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Instance model
    print("Detecting Signs...", flush=True)
    model = ObjectDetector(
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        img_size=img_size,
        view_img=False,
        save_img="src/common/out/content"
    )

    # Run inference
    model(dataloader)
    print("Inference Complete!", flush=True)
