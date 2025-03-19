import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from dataset import get_dataloaders
from model import GaitRecognitionModel
from config import *
import utils

# Load trained model
def load_model():
    if not os.path.exists(SAVE_MODEL_PATH):
        print("❌ Model file not found. Please train the model first.")
        exit()
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GaitRecognitionModel().to(device)
    model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=device))
    model.eval()
    return model, device

# Evaluate model
def evaluate_model():
    model, device = load_model()
    _, test_loader = get_dataloaders(TRAIN_DIR, TEST_DIR, BATCH_SIZE)

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device).float(), labels.to(device).long()
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Compute Accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")

    # Classification Report
    print("\n🔹 Classification Report:\n")
    print(classification_report(all_labels, all_preds))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm)

# Plot Confusion Matrix
def plot_confusion_matrix(cm):
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=False, cmap="Blues", fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    evaluate_model()
