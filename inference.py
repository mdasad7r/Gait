import os
os.environ["KERAS_BACKEND"] = "torch"  # Ensures TKAN works with PyTorch backend

import torch
from model import GaitRecognitionModel
from torchvision import transforms
from PIL import Image
from glob import glob

SEQUENCE_LEN = 10  # Must match training/evaluation
DEFAULT_MODEL_PATH = "/content/drive/MyDrive/gait_recognition_model.pth"

def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GaitRecognitionModel().to(device)

    # ✅ Load with strict=False to avoid TKAN key mismatch errors
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, device

def preprocess_sequence(sequence_dir):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    image_paths = sorted(glob(os.path.join(sequence_dir, "*.png")))
    frames = []

    for path in image_paths[:SEQUENCE_LEN]:
        img = Image.open(path).convert('L')
        img = transform(img)
        frames.append(img)

    sequence = torch.stack(frames)  # (T, C, H, W)

    # Pad if sequence is too short
    if len(frames) < SEQUENCE_LEN:
        pad_len = SEQUENCE_LEN - len(frames)
        pad = torch.zeros((pad_len, *frames[0].shape))
        sequence = torch.cat([sequence, pad], dim=0)

    sequence = sequence.unsqueeze(0)  # → (1, T, C, H, W)
    return sequence

def predict(sequence_dir, model_path=DEFAULT_MODEL_PATH):
    model, device = load_model(model_path)
    sequence = preprocess_sequence(sequence_dir)
    sequence = sequence.to(device)

    with torch.no_grad():
        output = model(sequence)
        prediction = torch.argmax(output, dim=1).item()
    return prediction

if __name__ == "__main__":
    test_sequence = "/content/casia-b/test/output/001/nm-05/000"  # Example path
    prediction = predict(test_sequence)
    print(f"Predicted Subject ID: {prediction + 1:03d}")
