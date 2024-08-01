import logging
import torch
from ultralytics import YOLOv10

logging.basicConfig(level=logging.INFO)
data_root = r"../common/data/quick_b/rtabmap_extract/data_rgb"
weights = r"../common/finetuned_models/yolov10/best.pt"
model = YOLOv10(weights).to(torch.device("cuda"))
results = model(source=f"{data_root}", batch=16, conf=0.25, save_txt=True, verbose=False)

for result in results:
    logging.info(f"Result: {result.summary()}")
    input()
