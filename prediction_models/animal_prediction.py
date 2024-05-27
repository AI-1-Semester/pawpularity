import torch
import torchvision.transforms as transforms
from PIL import Image

from pred_models.models import ConvolutionalNeuralNetworkModel

# Preprocess the Image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path)
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

# Load the Model
model = ConvolutionalNeuralNetworkModel()

# Load the trained model weights
model.load_state_dict(torch.load('./data/train/cnn_train_images/model/cats_vs_dogs.pth'))  # Replace 'model_weights.pth' with your model file path

# Perform Inference
def predict_image(image_path, model):
    image = preprocess_image(image_path)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

# Post-process the Predictions
def get_class_label(class_index):
    if class_index == 0:
        return 'Dog'
    elif class_index == 1:
        return 'Cat'
    else:
        return 'Unknown'
    
# Predict the Animal and return the label
def predict_animal(image_path):
    predicted_class = predict_image(image_path, model)
    predicted_label = get_class_label(predicted_class)
    return predicted_label