"""
DeepFake Detector - Pro Analysis Backend
Flask API server with CLIP ViT-L/14 model

Security Features:
- Rate limiting (IP-based with graceful 429 responses)
- Input validation & sanitization (schema-based)
- Environment-based configuration (no hardcoded secrets)
- OWASP security headers

// Built with < /> by Umesh
"""

import os
import io
import ssl
import re
import base64
import time
from functools import wraps
from collections import defaultdict
from datetime import datetime, timedelta

# Disable SSL verification for model downloads (workaround for macOS)
ssl._create_default_https_context = ssl._create_unverified_context

from flask import Flask, request, jsonify, g
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

# =============================================================================
# CONFIGURATION - Load from environment variables
# =============================================================================

class Config:
    """
    Configuration loaded from environment variables.
    All secrets should be in .env file, never hardcoded.
    """
    # Server settings
    PORT = int(os.environ.get('DEEPFAKE_PORT', 5002))
    HOST = os.environ.get('DEEPFAKE_HOST', '0.0.0.0')
    DEBUG = os.environ.get('DEEPFAKE_DEBUG', 'false').lower() == 'true'
    
    # Rate limiting settings (requests per minute)
    RATE_LIMIT_PER_MINUTE = int(os.environ.get('RATE_LIMIT_PER_MINUTE', 30))
    RATE_LIMIT_BURST = int(os.environ.get('RATE_LIMIT_BURST', 5))
    
    # Input validation limits
    MAX_IMAGE_SIZE_MB = int(os.environ.get('MAX_IMAGE_SIZE_MB', 10))
    MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024
    ALLOWED_MIME_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/gif']
    
    # API Key (optional, for future use)
    API_KEY = os.environ.get('DEEPFAKE_API_KEY', None)
    
    # Model settings
    USE_GPU = os.environ.get('USE_GPU', 'true').lower() == 'true'


# =============================================================================
# RATE LIMITING - IP-based with sliding window
# =============================================================================

class RateLimiter:
    """
    Rate limiter using sliding window algorithm.
    Tracks requests per IP address with configurable limits.
    """
    
    def __init__(self, requests_per_minute=30, burst_limit=5):
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.requests = defaultdict(list)  # IP -> list of timestamps
        
    def _cleanup_old_requests(self, ip):
        """Remove requests older than 1 minute"""
        cutoff = time.time() - 60
        self.requests[ip] = [t for t in self.requests[ip] if t > cutoff]
    
    def is_allowed(self, ip):
        """
        Check if request from IP is allowed.
        Returns (allowed: bool, retry_after: int or None)
        """
        self._cleanup_old_requests(ip)
        
        current_requests = len(self.requests[ip])
        
        # Check if over limit
        if current_requests >= self.requests_per_minute:
            # Calculate retry-after time
            oldest = min(self.requests[ip]) if self.requests[ip] else time.time()
            retry_after = int(60 - (time.time() - oldest)) + 1
            return False, max(retry_after, 1)
        
        # Check burst limit (requests in last 5 seconds)
        recent_cutoff = time.time() - 5
        recent_requests = len([t for t in self.requests[ip] if t > recent_cutoff])
        if recent_requests >= self.burst_limit:
            return False, 5
        
        # Allow and record request
        self.requests[ip].append(time.time())
        return True, None
    
    def get_remaining(self, ip):
        """Get remaining requests for this IP"""
        self._cleanup_old_requests(ip)
        return max(0, self.requests_per_minute - len(self.requests[ip]))


# Initialize rate limiter
rate_limiter = RateLimiter(
    requests_per_minute=Config.RATE_LIMIT_PER_MINUTE,
    burst_limit=Config.RATE_LIMIT_BURST
)


# =============================================================================
# INPUT VALIDATION - Schema-based with strict type checking
# =============================================================================

class ValidationError(Exception):
    """Custom exception for validation errors"""
    def __init__(self, message, field=None):
        self.message = message
        self.field = field
        super().__init__(self.message)


class InputValidator:
    """
    Schema-based input validation with strict type checking.
    Follows OWASP input validation guidelines.
    """
    
    @staticmethod
    def validate_base64_image(data, max_size_bytes=Config.MAX_IMAGE_SIZE_BYTES):
        """
        Validate base64-encoded image data.
        
        Checks:
        - Is a string
        - Valid base64 encoding
        - Decoded size within limits
        - Valid image format (JPEG, PNG, WebP, GIF)
        """
        if not isinstance(data, str):
            raise ValidationError("Image data must be a string", "image")
        
        # Remove data URL prefix if present
        if ',' in data:
            header, data = data.split(',', 1)
            # Validate header format
            if not re.match(r'^data:image/(jpeg|png|webp|gif);base64$', header):
                raise ValidationError("Invalid image data URL format", "image")
        
        # Check base64 validity
        try:
            # Remove whitespace
            data = re.sub(r'\s', '', data)
            
            # Validate base64 characters
            if not re.match(r'^[A-Za-z0-9+/]*={0,2}$', data):
                raise ValidationError("Invalid base64 encoding", "image")
            
            decoded = base64.b64decode(data)
        except Exception:
            raise ValidationError("Failed to decode base64 image", "image")
        
        # Check size
        if len(decoded) > max_size_bytes:
            raise ValidationError(
                f"Image too large. Maximum size is {Config.MAX_IMAGE_SIZE_MB}MB",
                "image"
            )
        
        # Validate image format
        try:
            img = Image.open(io.BytesIO(decoded))
            img.verify()  # Verify it's a valid image
            
            # Re-open for actual use (verify() makes it unusable)
            img = Image.open(io.BytesIO(decoded))
            
            # Check format
            if img.format and img.format.lower() not in ['jpeg', 'png', 'webp', 'gif']:
                raise ValidationError(
                    f"Unsupported image format: {img.format}",
                    "image"
                )
                
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Invalid image data: {str(e)}", "image")
        
        return decoded, img
    
    @staticmethod
    def validate_json_request(data, allowed_fields, required_fields=None):
        """
        Validate JSON request body.
        
        Checks:
        - Is a dict
        - Only contains allowed fields (reject unexpected)
        - Contains all required fields
        """
        if not isinstance(data, dict):
            raise ValidationError("Request body must be a JSON object")
        
        # Reject unexpected fields
        unexpected = set(data.keys()) - set(allowed_fields)
        if unexpected:
            raise ValidationError(
                f"Unexpected fields: {', '.join(unexpected)}",
                field="body"
            )
        
        # Check required fields
        if required_fields:
            missing = set(required_fields) - set(data.keys())
            if missing:
                raise ValidationError(
                    f"Missing required fields: {', '.join(missing)}",
                    field="body"
                )
        
        return True
    
    @staticmethod
    def sanitize_string(value, max_length=1000, allow_html=False):
        """Sanitize string input"""
        if not isinstance(value, str):
            raise ValidationError("Value must be a string")
        
        # Trim and limit length
        value = value.strip()[:max_length]
        
        # Remove HTML if not allowed
        if not allow_html:
            value = re.sub(r'<[^>]+>', '', value)
        
        return value


# =============================================================================
# DECORATORS - Rate limiting and validation
# =============================================================================

def rate_limit(f):
    """
    Decorator to apply rate limiting to endpoints.
    Returns 429 Too Many Requests with Retry-After header if limit exceeded.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get client IP (handle proxies)
        ip = request.headers.get('X-Forwarded-For', request.remote_addr)
        if ip:
            ip = ip.split(',')[0].strip()
        
        allowed, retry_after = rate_limiter.is_allowed(ip)
        
        if not allowed:
            response = jsonify({
                'error': 'Too many requests. Please slow down.',
                'retry_after': retry_after,
                'message': 'Rate limit exceeded. Please wait before making more requests.'
            })
            response.status_code = 429
            response.headers['Retry-After'] = str(retry_after)
            response.headers['X-RateLimit-Limit'] = str(Config.RATE_LIMIT_PER_MINUTE)
            response.headers['X-RateLimit-Remaining'] = '0'
            return response
        
        # Add rate limit headers to response
        g.rate_limit_remaining = rate_limiter.get_remaining(ip)
        
        return f(*args, **kwargs)
    return decorated_function


def validate_api_key(f):
    """
    Decorator to validate API key if configured.
    Skip if no API key is set in environment.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if Config.API_KEY:
            provided_key = request.headers.get('X-API-Key')
            if not provided_key or provided_key != Config.API_KEY:
                return jsonify({
                    'error': 'Invalid or missing API key',
                    'message': 'Please provide a valid API key in the X-API-Key header'
                }), 401
        return f(*args, **kwargs)
    return decorated_function


# =============================================================================
# FLASK APP SETUP
# =============================================================================

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:*", "http://127.0.0.1:*", "file://*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "X-API-Key"]
    }
})


@app.after_request
def add_security_headers(response):
    """
    Add OWASP recommended security headers to all responses.
    """
    # Prevent clickjacking
    response.headers['X-Frame-Options'] = 'DENY'
    
    # Prevent MIME type sniffing
    response.headers['X-Content-Type-Options'] = 'nosniff'
    
    # XSS protection
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    # Content Security Policy
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    
    # Add rate limit info if available
    if hasattr(g, 'rate_limit_remaining'):
        response.headers['X-RateLimit-Limit'] = str(Config.RATE_LIMIT_PER_MINUTE)
        response.headers['X-RateLimit-Remaining'] = str(g.rate_limit_remaining)
    
    return response


@app.errorhandler(ValidationError)
def handle_validation_error(error):
    """Handle validation errors with proper JSON response"""
    return jsonify({
        'error': 'Validation failed',
        'message': error.message,
        'field': error.field
    }), 400


# =============================================================================
# MODEL LOADING
# =============================================================================

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
        # Select device based on config
        if Config.USE_GPU and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
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


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/api/health', methods=['GET'])
@rate_limit
def health_check():
    """
    Health check endpoint.
    Returns server status and model information.
    """
    return jsonify({
        'status': 'online',
        'model_loaded': MODEL_LOADED,
        'model_name': MODEL_NAME if MODEL_LOADED else 'none',
        'device': str(device) if device else 'not initialized',
        'rate_limit': {
            'requests_per_minute': Config.RATE_LIMIT_PER_MINUTE,
            'max_image_size_mb': Config.MAX_IMAGE_SIZE_MB
        }
    })


@app.route('/api/analyze', methods=['POST'])
@rate_limit
@validate_api_key
def analyze():
    """
    Analyze image for deepfake detection.
    
    Request body:
        {
            "image": "base64-encoded-image-data"
        }
    
    Returns:
        {
            "success": true,
            "result": {
                "verdict": "real|fake|uncertain",
                "confidence": 0-100,
                "fake_probability": 0.0-1.0,
                "model": "model-name",
                "device": "cpu|cuda"
            }
        }
    """
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded. Server is starting up.'}), 503
    
    try:
        data = request.get_json()
        
        # Validate request structure
        InputValidator.validate_json_request(
            data,
            allowed_fields=['image'],
            required_fields=['image']
        )
        
        # Validate and decode image
        image_bytes, image = InputValidator.validate_base64_image(data['image'])
        
        # Analyze
        result, error = analyze_image(image)
        
        if error:
            return jsonify({'error': error}), 500
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except ValidationError as e:
        raise e
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/api/analyze-file', methods=['POST'])
@rate_limit
@validate_api_key
def analyze_file():
    """
    Analyze uploaded file for deepfake detection.
    
    Request:
        multipart/form-data with 'file' field
    
    Returns:
        Same as /api/analyze
    """
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded. Server is starting up.'}), 503
    
    try:
        if 'file' not in request.files:
            raise ValidationError('No file provided', 'file')
        
        file = request.files['file']
        
        # Check file size
        file.seek(0, 2)  # Seek to end
        size = file.tell()
        file.seek(0)  # Seek back to start
        
        if size > Config.MAX_IMAGE_SIZE_BYTES:
            raise ValidationError(
                f'File too large. Maximum size is {Config.MAX_IMAGE_SIZE_MB}MB',
                'file'
            )
        
        # Validate MIME type
        if file.content_type and file.content_type not in Config.ALLOWED_MIME_TYPES:
            raise ValidationError(
                f'Unsupported file type: {file.content_type}',
                'file'
            )
        
        # Open and validate image
        try:
            image = Image.open(file.stream)
            image.verify()
            file.stream.seek(0)
            image = Image.open(file.stream)
        except Exception as e:
            raise ValidationError(f'Invalid image file: {str(e)}', 'file')
        
        result, error = analyze_image(image)
        
        if error:
            return jsonify({'error': error}), 500
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except ValidationError as e:
        raise e
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    print("=" * 50)
    print("DeepFake Detector - Pro Analysis Server")
    print("// Built with </> by Umesh")
    print("=" * 50)
    
    # Load model on startup
    load_clip_model()
    
    # Print config
    print(f"\n📋 Configuration:")
    print(f"   Rate Limit: {Config.RATE_LIMIT_PER_MINUTE} req/min")
    print(f"   Max Image Size: {Config.MAX_IMAGE_SIZE_MB} MB")
    print(f"   API Key Required: {'Yes' if Config.API_KEY else 'No'}")
    
    # Run server
    print(f"\n🚀 Starting server on http://localhost:{Config.PORT}")
    print("API Endpoints:")
    print("  GET  /api/health       - Check server status")
    print("  POST /api/analyze      - Analyze base64 image")
    print("  POST /api/analyze-file - Analyze uploaded file")
    print("=" * 50)
    
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
