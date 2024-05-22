import os

import yaml
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class ImageDataset(Dataset):
    def __init__(
        self, image_dir, depth_image_dir, calibration_dir, img_size, processing=True
    ):
        self.image_dir, self.depth_image_dir, self.calibration_dir = (
            image_dir,
            depth_image_dir,
            calibration_dir,
        )
        self.img_size = img_size
        self.processing = processing

        self.image_filenames = natsorted(os.listdir(image_dir))
        self.depth_image_filenames = natsorted(os.listdir(depth_image_dir))
        self.paired_filenames = self._pair_filenames()

        self.depth_transform = transforms.ToTensor()

    def _pair_filenames(self):
        # Filter to keep only those files where both image and corresponding depth image exist
        paired_filenames = []
        for image_filename in self.image_filenames:
            # Assuming the naming convention matches except for the extension and directory
            depth_filename = image_filename.replace(".jpg", ".png")
            if depth_filename in self.depth_image_filenames:
                paired_filenames.append((image_filename, depth_filename))

        return paired_filenames

    def __len__(self):
        return len(self.paired_filenames)

    def __getitem__(self, idx):
        image_filename, depth_filename = self.paired_filenames[idx]

        image_path = os.path.join(self.image_dir, image_filename)
        depth_image_path = os.path.join(self.depth_image_dir, depth_filename)
        calibration_path = os.path.join(
            self.calibration_dir, image_filename.replace(".jpg", ".yaml")
        )

        depth_image_tensor = self._load_depth_image(depth_image_path)
        image_tensor = self._load_image(image_path, depth_image_tensor)
        camera_intrinsics = self._load_calibration(calibration_path)

        return image_tensor, depth_image_tensor, camera_intrinsics

    def _load_depth_image(self, path):
        with Image.open(path) as img:
            return self.depth_transform(img)[0, :, :]  # the depth channel is channel 0

    def _load_image(self, path, depth_img):
        with Image.open(path) as img:
            if self.processing:
                transform = transforms.Compose(
                    [
                        transforms.Resize((self.img_size, self.img_size)),
                        transforms.ToTensor(),
                    ]
                )
                return transform(img)
            else:
                transform = transforms.Compose(
                    [
                        transforms.Resize((depth_img.size()[0], depth_img.size()[1])),
                        transforms.ToTensor(),
                    ]
                )
                return transform(img)

    def _load_calibration(self, calibration_path):
        with open(calibration_path, "r") as file:
            calibration_data = yaml.safe_load(file)

        image_width = calibration_data.get("image_width", None)
        image_height = calibration_data.get("image_height", None)

        fx = calibration_data["camera_matrix"]["data"][0]
        fy = calibration_data["camera_matrix"]["data"][4]
        cx = calibration_data["camera_matrix"]["data"][2]
        cy = calibration_data["camera_matrix"]["data"][5]

        return {
            "image_width": image_width,
            "image_height": image_height,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        }
