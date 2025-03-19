import os
import torch
import torch.optim as optim
import torch.nn as nn
from dataset import get_dataloaders
from model import GaitRecognitionModel
from config import *
from torch.utils.tensorboard import SummaryWriter

def download_casia_b():
    """Downloads CASIA-B dataset from Kaggle using API if not already downloaded."""
    if not os.path.exists("/content/casia-b"):
        print("Downloading CASIA-B dataset from Kaggle...")
        os.system("mkdir -p ~/.kaggle")
        os.system("cp /content/drive/MyDrive/kaggle.json ~/.kaggle/")  # Ensure kaggle.json is in Google Drive
        os.system("chmod 600 ~/.kaggle/kaggle.json")
        os.system("kaggle datasets download -d trnquanghuyn/casia-b -p /content")
        os.system("unzip /content/casia-b.zip -d /content/casia-b")
        print("Dataset downloaded and extracted successfully!")

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GaitRecognitionModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter()

    # Download dataset if not available
    download_casia_b()

    # Load Data
    train_loader, test_loader = get_dataloaders(TRAIN_DIR, TEST_DIR, BATCH_SIZE)

    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device).float(), labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        writer.add_scalar("Training Loss", avg_loss, epoch)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print("Model saved successfully.")
    writer.close()

if __name__ == "__main__":
    train_model()
