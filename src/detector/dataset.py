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
        self.paired_filenames = self._pair_filenames()

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])

    def _pair_filenames(self):
        # Filter to keep only those files where both image and corresponding depth image exist
        paired_filenames = []
        for image_filename in self.image_filenames:
            # Assuming the naming convention matches except for the extension and directory
            depth_filename = image_filename.replace('.jpg', '.png')  # Change extensions accordingly
            if depth_filename in self.depth_image_filenames:
                paired_filenames.append((image_filename, depth_filename))

        return paired_filenames

    def __len__(self):
        return len(self.paired_filenames)

    def __getitem__(self, idx):
        image_filename, depth_filename = self.paired_filenames[idx]
        
        image_path = os.path.join(self.image_dir, image_filename)
        depth_image_path = os.path.join(self.depth_image_dir, depth_filename)

        image_tensor = self._load_image(image_path)
        depth_image_tensor = self._load_image(depth_image_path, is_depth=True)

        return image_tensor, depth_image_tensor

    def _load_image(self, path, is_depth=False):
        with Image.open(path) as img:
            if is_depth:
                img = img.convert('L')
            return self.transform(img)
