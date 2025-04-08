import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from dataset import get_dataloaders
from model import GaitRecognitionModel
from config import *
from utils import calculate_accuracy, calculate_top5_accuracy

def load_model():
    if not os.path.exists(SAVE_MODEL_PATH):
        print("❌ Model file not found. Please train the model first.")
        exit()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GaitRecognitionModel().to(device)
    state_dict = torch.load(SAVE_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device

def evaluate_model():
    model, device = load_model()
    _, test_loader = get_dataloaders(
        TRAIN_DIR, TEST_DIR,
        batch_size=BATCH_SIZE,
        sequence_len=SEQUENCE_LEN
    )

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device).float(), labels.to(device).long()
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = [label + 1 for label in all_labels]
    all_preds = [pred + 1 for pred in all_preds]

    accuracy = accuracy_score(all_labels, all_preds)
    top5_acc = calculate_top5_accuracy(model, test_loader, device)
    print(f"✅ Test Accuracy: {accuracy * 100:.2f}%, Top-5 Accuracy: {top5_acc:.2f}%")

    print("\n🔹 Classification Report:\n")
    print(classification_report(all_labels, all_preds, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm)

def plot_confusion_matrix(cm):
    labels = [str(i).zfill(3) for i in range(1, 125)]
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=False, cmap="Blues", fmt="d", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    evaluate_model()
