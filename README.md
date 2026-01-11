# 🔍 DeepFake Detector

An AI-powered web application that detects whether a face in an image is **real** or **AI-generated (fake)**. Built with a modern dark glassmorphism UI and featuring two detection modes.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow.js-Enabled-orange.svg)
![Security](https://img.shields.io/badge/OWASP-Hardened-green.svg)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📤 **Image Upload** | Drag & drop or click to upload images |
| 📷 **Webcam Capture** | Take photos directly from your webcam |
| 🔍 **Quick Scan** | Browser-based detection using face-api.js |
| ⚡ **Pro Analysis** | CLIP ViT-L/14 AI model for higher accuracy |
| 🎨 **Modern UI** | Dark glassmorphism theme with animations |
| 🔒 **Security First** | Rate limiting, input validation, OWASP headers |

---

## 🚀 Quick Start

### Option 1: Basic Mode (No Setup Required!)

Just open the app in your browser - it works instantly!

```bash
# macOS
open index.html

# Windows
start index.html

# Linux
xdg-open index.html
```

The **Quick Scan** mode uses browser-based AI (TensorFlow.js + face-api.js) and requires no installation.

---

### Option 2: Pro Analysis Mode (Higher Accuracy)

Pro mode uses the **CLIP ViT-L/14** model for more accurate deepfake detection. First-time setup will download ~890MB for the AI model.

#### Step 1: Navigate to the backend folder
```bash
cd backend
```

#### Step 2: Run the setup script (one-time only)
```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create a Python virtual environment
- Install all dependencies (PyTorch, Flask, CLIP)
- Set up the required directories

#### Step 3: Configure environment (optional)
```bash
cp .env.example .env
# Edit .env to customize settings
```

#### Step 4: Start the Pro Analysis server
```bash
source venv/bin/activate
python server.py
```

You should see:
```
✓ CLIP model loaded successfully!
🚀 Starting server on http://localhost:5002
```

#### Step 5: Open the web app
Open `index.html` in your browser. The **Pro Analysis** button should now show as active with a green status indicator!

---

## 🔒 Security Features

This project follows **OWASP best practices** for web application security:

### Rate Limiting
- IP-based sliding window algorithm
- 30 requests/minute default (configurable)
- Burst protection (5 requests/5 seconds)
- Graceful 429 responses with `Retry-After` header

### Input Validation
- Schema-based validation on all endpoints
- Strict type checking
- File size limits (10MB default)
- MIME type validation
- Rejects unexpected fields

### Security Headers
```
X-Frame-Options: DENY
X-Content-Type-Options: nosniff
X-XSS-Protection: 1; mode=block
Content-Security-Policy: default-src 'self'
```

### Environment Configuration
All sensitive settings loaded from environment variables - no hardcoded secrets!

```bash
# Copy example config
cp backend/.env.example backend/.env

# Edit as needed
nano backend/.env
```

---

## 📁 Project Structure

```
DeepFakeDetector/
├── index.html          # Main web interface
├── styles.css          # Dark glassmorphism theme
├── app.js              # Frontend controller + backend connection
├── detector.js         # Browser-based face analysis
├── README.md           # You're reading this!
└── backend/
    ├── server.py       # Flask API with CLIP model (security hardened)
    ├── requirements.txt # Python dependencies
    ├── setup.sh        # Auto-setup script
    ├── .env.example    # Example environment config
    └── pretrained_weights/  # (Optional) Custom weights
```

---

## 🛠️ Technologies Used

### Frontend
- **HTML5 / CSS3** - Modern semantic markup
- **JavaScript (ES6+)** - No framework needed
- **TensorFlow.js** - ML in the browser
- **face-api.js** - Face detection & landmark analysis
- **Google Fonts (Inter)** - Clean typography

### Backend (Pro Mode)
- **Python 3.10+** - Backend language
- **Flask** - Lightweight web server
- **PyTorch** - Deep learning framework
- **OpenAI CLIP** - Vision-language AI model
- **Pillow** - Image processing

---

## ⚙️ Configuration Options

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `DEEPFAKE_PORT` | 5002 | Server port |
| `RATE_LIMIT_PER_MINUTE` | 30 | Max requests per minute |
| `RATE_LIMIT_BURST` | 5 | Max burst requests |
| `MAX_IMAGE_SIZE_MB` | 10 | Max image upload size |
| `DEEPFAKE_API_KEY` | (none) | Optional API key for auth |
| `USE_GPU` | true | Use GPU if available |

---

## 🔧 Troubleshooting

### "Port already in use" error
```bash
# Kill any existing processes on port 5002
lsof -ti:5002 | xargs kill -9
```

### SSL Certificate errors on macOS
The setup script handles this, but if you still have issues:
```bash
pip install certifi
```

### Pro Analysis button stays disabled
Make sure the backend server is running and check the terminal for errors.

### Rate limit exceeded (429 error)
Wait for the `Retry-After` time indicated in the response, or adjust `RATE_LIMIT_PER_MINUTE` in your `.env` file.

---

## 📊 API Endpoints (Pro Mode)

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/health` | GET | No | Check server status |
| `/api/analyze` | POST | Optional | Analyze base64 image |
| `/api/analyze-file` | POST | Optional | Analyze uploaded file |

### Example Request
```bash
curl http://localhost:5002/api/health
# {"model_loaded": true, "model_name": "CLIP-ViT-L/14", "status": "online"}
```

### Rate Limit Headers
All responses include:
- `X-RateLimit-Limit`: Maximum requests per minute
- `X-RateLimit-Remaining`: Remaining requests

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

---

## ⚠️ Disclaimer

This is an **educational MVP** project. Detection results may not be 100% accurate and should not be used as the sole basis for important decisions. The technology is meant to demonstrate AI capabilities and raise awareness about deepfakes.

---

## 📄 License

MIT License - feel free to use this project for learning and personal projects!

---

## 🙏 Acknowledgments

- [OpenAI CLIP](https://github.com/openai/CLIP) - Vision-language model
- [face-api.js](https://github.com/justadudewhohacks/face-api.js) - Browser face detection
- [TensorFlow.js](https://www.tensorflow.org/js) - ML in the browser
- [UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect) - Research inspiration

---

<p align="center">
  <code>&lt;/&gt;</code> by <b>Umesh</b> 
  <br>
  <sub>// console.log("Built with ☕ and curiosity")</sub>
</p>
