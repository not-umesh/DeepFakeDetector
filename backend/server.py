"""
DeepFake Detector - Pro Analysis Backend
Flask API server using UniversalFakeDetect (CLIP ViT-L/14)
"""

import os
import io
import ssl
import base64

# Disable SSL verification for model downloads (workaround for macOS)
ssl._create_default_https_context = ssl._create_unverified_context

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for browser requests

# Global model variables
model = None
device = None
transform = None
MODEL_LOADED = False
MODEL_NAME = "Unknown"

def get_transform():
    """Get image preprocessing transform"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

def load_clip_model():
    """Load CLIP-based detector model"""
    global model, device, transform, MODEL_LOADED, MODEL_NAME
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Try to load CLIP
        try:
            import clip
            print("Loading CLIP ViT-L/14 model...")
            clip_model, _ = clip.load("ViT-L/14", device=device)
            
            # Create classifier head
            class CLIPDetector(nn.Module):
                def __init__(self, clip_model):
                    super().__init__()
                    self.clip = clip_model
                    self.fc = nn.Linear(768, 1)
                    
                def forward(self, x):
                    with torch.no_grad():
                        features = self.clip.encode_image(x)
                    features = features.float()
                    return torch.sigmoid(self.fc(features))
            
            model = CLIPDetector(clip_model)
            MODEL_NAME = "CLIP-ViT-L/14"
            
            # Load pretrained weights if available
            weights_path = os.path.join(os.path.dirname(__file__), 'pretrained_weights', 'fc_weights.pth')
            if os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location=device)
                model.fc.load_state_dict(state_dict)
                print("Loaded pretrained weights")
            else:
                print("Warning: No pretrained weights found. Using random initialization.")
            
            model = model.to(device)
            model.eval()
            print(f"✓ CLIP model loaded successfully!")
            
        except Exception as clip_error:
            print(f"CLIP loading failed: {clip_error}")
            print("Using fallback ResNet50 model...")
            
            from torchvision import models
            from torchvision.models import ResNet50_Weights
            
            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            model.fc = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 1),
                nn.Sigmoid()
            )
            model = model.to(device)
            model.eval()
            MODEL_NAME = "ResNet50-Fallback"
            print(f"✓ Fallback ResNet50 model loaded successfully!")
        
        transform = get_transform()
        MODEL_LOADED = True
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        MODEL_LOADED = False
        return False

def analyze_image(image):
    """Analyze a single image for deepfake detection"""
    global model, device, transform, MODEL_NAME
    
    if not MODEL_LOADED:
        return None, "Model not loaded"
    
    try:
        # Preprocess image
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
            fake_prob = output.item()
        
        # Determine result
        if fake_prob > 0.6:
            verdict = 'fake'
            confidence = int(fake_prob * 100)
        elif fake_prob < 0.4:
            verdict = 'real'
            confidence = int((1 - fake_prob) * 100)
        else:
            verdict = 'uncertain'
            confidence = int(50 + abs(fake_prob - 0.5) * 100)
        
        return {
            'verdict': verdict,
            'confidence': confidence,
            'fake_probability': round(fake_prob, 4),
            'model': MODEL_NAME,
            'device': str(device)
        }, None
        
    except Exception as e:
        return None, str(e)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'model_loaded': MODEL_LOADED,
        'model_name': MODEL_NAME if MODEL_LOADED else 'none',
        'device': str(device) if device else 'not initialized'
    })

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Analyze image for deepfake detection"""
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Analyze
        result, error = analyze_image(image)
        
        if error:
            return jsonify({'error': error}), 500
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-file', methods=['POST'])
def analyze_file():
    """Analyze uploaded file"""
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        image = Image.open(file.stream)
        
        result, error = analyze_image(image)
        
        if error:
            return jsonify({'error': error}), 500
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 50)
    print("DeepFake Detector - Pro Analysis Server")
    print("=" * 50)
    
    # Load model on startup
    load_clip_model()
    
    # Run server
    print(f"\n🚀 Starting server on http://localhost:5002")
    print("API Endpoints:")
    print("  GET  /api/health   - Check server status")
    print("  POST /api/analyze  - Analyze base64 image")
    print("  POST /api/analyze-file - Analyze uploaded file")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5002, debug=False)
