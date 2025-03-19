import torch
from model import GaitRecognitionModel
from torchvision import transforms
from PIL import Image

# Load model
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GaitRecognitionModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def predict(image_path, model_path="gait_recognition_model.pth"):
    model = load_model(model_path)
    image = preprocess_image(image_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)
    
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()
    return prediction

if __name__ == "__main__":
    test_image = "path/to/test/image.png"
    prediction = predict(test_image)
    print(f"Predicted Class: {prediction}")
