import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2
import os
from pathlib import Path

# Load the model
model = torch.load('model/leaf_disease_detector.pt')
model.eval()

# Global holders for gradients and activations
gradients = None
activations = None

# Hook functions
def forward_hook(module, input, output):
    global activations
    activations = output.detach()

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0].detach()

# Attach hooks to last conv layer
target_layer = model.features[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# Preprocess image
def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert("RGB")
    return img, transform(img).unsqueeze(0)

# Generate Grad-CAM and save it
def generate_heatmap(image_path):
    img, input_tensor = preprocess(image_path)

    # Forward pass
    output = model(input_tensor)
    class_idx = torch.argmax(output)
    print(f"🔎 Predicted class index: {class_idx.item()}")

    # Backward pass
    model.zero_grad()
    output[0, class_idx].backward()

    # Weight activations by gradients
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    # Generate heatmap
    cam = torch.mean(activations, dim=1).squeeze().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (img.width, img.height))
    cam = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

    # Overlay heatmap on original image
    original_img = np.array(img)
    overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    # Final output path (absolute)
    project_root = Path(__file__).resolve().parent.parent
    gradcam_path = project_root / "output" / "gradcam_overlay.jpg"
    os.makedirs(gradcam_path.parent, exist_ok=True)

    # Save and confirm
    success = cv2.imwrite(str(gradcam_path), overlay)
    if success and gradcam_path.exists():
        print(f"✅ Grad-CAM saved successfully at:\n{gradcam_path}")
    else:
        print("❌ Grad-CAM FAILED to save.")

# Run from terminal
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python src/explainability.py <image_path>")
    else:
        generate_heatmap(sys.argv[1])
