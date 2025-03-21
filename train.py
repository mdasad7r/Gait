"""import os
os.environ["KERAS_BACKEND"] = "torch"

import os
import torch
import torch.optim as optim
import torch.nn as nn
from dataset import get_dataloaders
from model import GaitRecognitionModel
from config import *
from torch.utils.tensorboard import SummaryWriter

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GaitRecognitionModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Learning rate scheduler
    writer = SummaryWriter()

    #train_loader, test_loader = get_dataloaders(TRAIN_DIR, TEST_DIR, BATCH_SIZE)
    train_loader, test_loader = get_dataloaders(
    TRAIN_DIR, TEST_DIR, BATCH_SIZE,
    sequence_len=60  # or 50, 64, etc. depending on what fits in memory
)
    
    best_loss = float("inf")  # Track best loss

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

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print("✅ Best model saved!")

        scheduler.step()  # Adjust learning rate

    writer.close()

if __name__ == "__main__":
    train_model()
"""

import os
os.environ["KERAS_BACKEND"] = "torch"

import os
import torch
import torch.optim as optim
import torch.nn as nn
from dataset import get_dataloaders
from model import GaitRecognitionModel
from config import *
from torch.utils.tensorboard import SummaryWriter

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GaitRecognitionModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    writer = SummaryWriter()

    train_loader, test_loader = get_dataloaders(
        TRAIN_DIR, TEST_DIR, BATCH_SIZE, sequence_len=10
    )

    best_val_loss = float("inf")
    patience = 3  # Stop after 3 epochs of no improvement
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device).float(), labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        writer.add_scalar("Training Loss", avg_train_loss, epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device).float(), labels.to(device).long()
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(test_loader)
        writer.add_scalar("Validation Loss", avg_val_loss, epoch)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print("✅ Best model saved!")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

        scheduler.step()

    writer.close()

if __name__ == "__main__":
    train_model()
