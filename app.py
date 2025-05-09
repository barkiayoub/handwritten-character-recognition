from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import io
from model_loader import load_model, transform
import torch


app = Flask(__name__)
model, class_mapping = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "Empty filename", 400

    try:
        # Process image
        image = Image.open(io.BytesIO(file.read())).convert('L')  # Convert to grayscale
        image = image.resize((28, 28))
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            character = class_mapping[predicted.item()]
        
        return character  # Return plain text

    except Exception as e:
        return str(e), 500  # Return error as plain text

if __name__ == '__main__':
    app.run(debug=True)