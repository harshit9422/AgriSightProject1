import torch
import torchvision.models as models

def save_model():
    
    model = models.mobilenet_v2(pretrained=True)
    model.eval()  
    
    torch.save(model, 'model/leaf_disease_detector.pt')
    print("Model saved to model/leaf_disease_detector.pt")

if __name__ == "__main__":
    save_model()
