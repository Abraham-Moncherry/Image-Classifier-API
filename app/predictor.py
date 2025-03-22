import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F  # Add this to use softmax

# Define class labels for CIFAR-10
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Load the model
def load_model(model_path='model/trained_model.pt'):
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Transform incoming image to match training format
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(image_bytes).convert("RGB")
    return transform(image).unsqueeze(0)

# Predict the class and confidence of the image
def get_prediction(image_bytes, model):
    tensor = transform_image(image_bytes)
    outputs = model(tensor)

    # Convert outputs to probabilities using softmax
    probabilities = F.softmax(outputs, dim=1)
    confidence, predicted_idx = torch.max(probabilities, 1)

    label = CLASS_NAMES[predicted_idx.item()]
    confidence_score = confidence.item()  # Value between 0 and 1

    return {
        "label": label,
        "confidence": round(confidence_score, 4)  # e.g., 0.9231
    }
