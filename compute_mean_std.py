import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from PIL import Image
import os

IMG_SIZE = 64

class DiceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for label in range(1, 7):
            class_dir = os.path.join(root_dir, str(label))
            if not os.path.exists(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append(os.path.join(class_dir, fname))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image = Image.open(self.samples[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

stats_transform = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

dataset = DiceDataset("data/cropped/train", transform=stats_transform)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

print(f"Computing mean/std over {len(dataset)} training images...")

channel_sum = torch.zeros(3)
channel_sum_sq = torch.zeros(3)
pixel_count = 0

for images in loader:
    b, c, h, w = images.shape
    pixel_count += b * h * w
    channel_sum += images.sum(dim=[0, 2, 3])
    channel_sum_sq += (images ** 2).sum(dim=[0, 2, 3])

mean = channel_sum / pixel_count
std = torch.sqrt(channel_sum_sq / pixel_count - mean ** 2)

print("\nDataset-specific normalization stats:")
print(f"  mean = [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
print(f"  std  = [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")