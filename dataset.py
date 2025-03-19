import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from glob import glob
from PIL import Image

class CASIABDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Read all image paths from subject and condition subfolders
        self.image_paths = glob(os.path.join(root_dir, "*/*/*.png"))  # Now includes walking conditions
        self.labels = [int(os.path.basename(path).split('_')[0]) for path in self.image_paths]

        # Debugging: Print dataset size
        print(f"📂 Loaded {len(self.image_paths)} images from {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def get_dataloaders(train_dir, test_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = CASIABDataset(train_dir, transform=transform)
    test_dataset = CASIABDataset(test_dir, transform=transform)

    print(f"✅ Training Samples: {len(train_dataset)} | Testing Samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
