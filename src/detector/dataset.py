import os
from PIL import Image

from torch.utils.data import Dataset
from torchvision.transforms import transforms

class ImageDataset(Dataset):
    def __init__(self, image_dir, depth_image_dir, img_size):
        self.image_dir, self.depth_image_dir = image_dir, depth_image_dir
        self.img_size = img_size
        self.image_filenames = sorted(os.listdir(image_dir))
        self.depth_image_filenames = sorted(os.listdir(depth_image_dir))
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        depth_image_path = os.path.join(self.depth_image_dir, self.depth_image_filenames[idx])

        image_tensor = self.load_image(image_path, self.transform)
        depth_image_tensor = self.load_image(depth_image_path, self.transform)

        return image_tensor, depth_image_tensor

    def load_image(self, path, transform):
        with Image.open(path) as img:
            img = img
            return transform(img)
