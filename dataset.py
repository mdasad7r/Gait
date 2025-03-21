import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from glob import glob
from PIL import Image

class CASIABSequenceDataset(Dataset):
    def __init__(self, root_dir, transform=None, sequence_len=None):
        self.root_dir = root_dir
        self.transform = transform
        self.sequence_paths = glob(os.path.join(root_dir, "*/*/*"))  # Subject/condition/sequence folders
        self.sequence_paths = [p for p in self.sequence_paths if os.path.isdir(p)]
        self.sequence_len = sequence_len  # Optional truncation or padding
        self.label_map = self._create_label_map()

        print(f"📂 Found {len(self.sequence_paths)} sequences in {root_dir}")

    def _create_label_map(self):
        subjects = sorted(set(os.path.basename(os.path.normpath(p).split(os.sep)[-3]) for p in self.sequence_paths))
        return {sid: idx for idx, sid in enumerate(subjects)}

    def __len__(self):
        return len(self.sequence_paths)

    def __getitem__(self, idx):
        seq_dir = self.sequence_paths[idx]
        image_paths = sorted(glob(os.path.join(seq_dir, "*.png")))
        frames = []

        for img_path in image_paths:
            img = Image.open(img_path).convert("L")
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        sequence = torch.stack(frames)

        if self.sequence_len:
            T = sequence.shape[0]
            if T < self.sequence_len:
                pad = torch.zeros((self.sequence_len - T, *sequence.shape[1:]))
                sequence = torch.cat([sequence, pad], dim=0)
            elif T > self.sequence_len:
                sequence = sequence[:self.sequence_len]

        subject_id = os.path.basename(os.path.normpath(seq_dir).split(os.sep)[-3])
        label = self.label_map[subject_id]

        return sequence, label

def get_dataloaders(train_dir, test_dir, batch_size=4, sequence_len=None):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = CASIABSequenceDataset(train_dir, transform=transform, sequence_len=sequence_len)
    test_dataset = CASIABSequenceDataset(test_dir, transform=transform, sequence_len=sequence_len)

    print(f"✅ Training Sequences: {len(train_dataset)} | Testing Sequences: {len(test_dataset)}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
