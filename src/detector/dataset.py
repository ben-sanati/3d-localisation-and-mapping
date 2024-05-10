from PIL import Image

from torch.utils.data import Dataset
from torchvision.transforms import transforms

class ImageDataset(Dataset):
    def __init__(self, data, img_size):
        self.data = data
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        image_pil = Image.fromarray(image)
        image_tensor = self.transform(image_pil)
        return image_tensor