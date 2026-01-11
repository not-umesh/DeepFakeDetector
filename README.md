# 🔍 DeepFake Detector

An AI-powered web application that detects whether a face in an image is **real** or **AI-generated (fake)**. Built with a modern dark glassmorphism UI and featuring two detection modes.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow.js-Enabled-orange.svg)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📤 **Image Upload** | Drag & drop or click to upload images |
| 📷 **Webcam Capture** | Take photos directly from your webcam |
| 🔍 **Quick Scan** | Browser-based detection using face-api.js |
| ⚡ **Pro Analysis** | CLIP ViT-L/14 AI model for higher accuracy |
| 🎨 **Modern UI** | Dark glassmorphism theme with animations |
| 🔒 **Privacy First** | Quick Scan runs 100% in your browser |

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

#### Step 3: Start the Pro Analysis server
```bash
source venv/bin/activate
python server.py
```

You should see:
```
✓ CLIP model loaded successfully!
🚀 Starting server on http://localhost:5002
```

#### Step 4: Open the web app
Open `index.html` in your browser. The **Pro Analysis** button should now show as active with a green status indicator!

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
    ├── server.py       # Flask API with CLIP model
    ├── requirements.txt # Python dependencies
    ├── setup.sh        # Auto-setup script
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

---

## 📊 How Detection Works

### Quick Scan (Browser-based)
Analyzes facial features for signs of AI generation:
- Facial symmetry (AI faces are often too perfect)
- Eye pattern analysis (identical eyes are suspicious)
- Facial proportions (golden ratio detection)
- Expression naturalness

### Pro Analysis (CLIP Model)
Uses OpenAI's CLIP vision model to analyze the entire image for AI-generated artifacts that are invisible to simpler methods.

---

## 📝 API Endpoints (Pro Mode)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Check server status |
| `/api/analyze` | POST | Analyze base64 image |
| `/api/analyze-file` | POST | Analyze uploaded file |

Example:
```bash
curl http://localhost:5002/api/health
# {"model_loaded": true, "model_name": "CLIP-ViT-L/14", "status": "online"}
```

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

Made with ❤️ for AI education and awareness
