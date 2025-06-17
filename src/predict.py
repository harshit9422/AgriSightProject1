import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
import os

# Load the pretrained model
model_path = 'model/leaf_disease_detector.pt'
if not os.path.exists(model_path):
    print("Model file not found. Please run download_model.py first.")
    sys.exit(1)

model = torch.load(model_path)
model.eval()

# Transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_t)
        predicted_class = torch.argmax(output, dim=1).item()

    print(f"✅ Predicted class index: {predicted_class} (from model’s ImageNet classes)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py <image_path>")
    else:
        predict(sys.argv[1])
