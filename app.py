from flask import Flask, render_template, request, jsonify
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
from torchvision import models
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet50 model as feature extractor
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove last layer
resnet.to(device)

class ResNetClassifier(nn.Module):
    def __init__(self, feature_extractor, num_classes=5):
        super(ResNetClassifier, self).__init__()
        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)  # Extract features
        x = torch.flatten(x, 1)  # Flatten features properly
        return self.fc(x)

model = ResNetClassifier(resnet, num_classes=5).to(device)

# Load fine-tuned model weights
model_weights_path = "C:/Users/Ijaz Ul Hassan/Documents/semester 5/Application of Data Science/project/resnet_finetuned.pth"
if os.path.exists(model_weights_path):
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    print("✅ Model weights loaded successfully!")
else:
    print("❌ Model weights file not found!")

model.eval()

# Define class labels
class_labels = ["Water", "Desert", "Industrial", "Barren", "Crop"]

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

def classify_image(image_path):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)  # Convert to probabilities
        print("🔍 Probabilities:", probabilities.cpu().numpy())  # Debugging step
        _, predicted_class = torch.max(probabilities, 1)
    return class_labels[predicted_class.item()]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

# @app.route('/dataset')
# def about():
#     return render_template('dataset.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        result = classify_image(filepath)
        return jsonify({'filename': filename, 'result': result})

if __name__ == '__main__':
    app.run(debug=True)
