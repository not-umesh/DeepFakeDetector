/**
 * DeepFake Detector - Main Application
 * Handles UI interactions, mode switching, and backend connection
 */

class DeepFakeApp {
    constructor() {
        this.detector = new DeepFakeDetector();
        this.webcamStream = null;
        this.currentImage = null;
        this.currentImageData = null;

        // Analysis mode: 'basic' (browser-based) or 'pro' (backend)
        this.analysisMode = 'basic';
        this.backendOnline = false;
        this.backendUrl = 'http://localhost:5002';

        // DOM elements
        this.elements = {
            uploadSection: document.getElementById('uploadSection'),
            webcamSection: document.getElementById('webcamSection'),
            analysisSection: document.getElementById('analysisSection'),
            dropzone: document.getElementById('dropzone'),
            fileInput: document.getElementById('fileInput'),
            webcamBtn: document.getElementById('webcamBtn'),
            webcamVideo: document.getElementById('webcamVideo'),
            webcamCanvas: document.getElementById('webcamCanvas'),
            captureBtn: document.getElementById('captureBtn'),
            backBtn: document.getElementById('backBtn'),
            previewImage: document.getElementById('previewImage'),
            faceOverlay: document.getElementById('faceOverlay'),
            loadingState: document.getElementById('loadingState'),
            loadingText: document.getElementById('loadingText'),
            results: document.getElementById('results'),
            resultCard: document.getElementById('resultCard'),
            resultIcon: document.getElementById('resultIcon'),
            resultTitle: document.getElementById('resultTitle'),
            resultSubtitle: document.getElementById('resultSubtitle'),
            confidenceValue: document.getElementById('confidenceValue'),
            confidenceFill: document.getElementById('confidenceFill'),
            analysisDetails: document.getElementById('analysisDetails'),
            analyzeNewBtn: document.getElementById('analyzeNewBtn'),
            loadingProgress: document.querySelector('.loading-progress'),
            // New Pro mode elements
            backendStatus: document.getElementById('backendStatus'),
            statusDot: document.getElementById('statusDot'),
            statusText: document.getElementById('statusText'),
            basicModeBtn: document.getElementById('basicModeBtn'),
            proModeBtn: document.getElementById('proModeBtn'),
            analysisBadge: document.getElementById('analysisBadge'),
            analysisModeText: document.getElementById('analysisModeText'),
            setupInstructions: document.getElementById('setupInstructions')
        };

        this.init();
    }

    async init() {
        this.bindEvents();
        await this.loadModels();
        this.checkBackendStatus();

        // Periodically check backend status
        setInterval(() => this.checkBackendStatus(), 30000);
    }

    async loadModels() {
        console.log('Loading AI models...');
        const loaded = await this.detector.loadModels();
        if (loaded) {
            console.log('Models ready!');
        } else {
            console.error('Failed to load models');
        }
    }

    async checkBackendStatus() {
        this.elements.statusDot.className = 'status-dot checking';
        this.elements.statusText.textContent = 'Pro Server: Checking...';

        try {
            const response = await fetch(`${this.backendUrl}/api/health`, {
                method: 'GET',
                signal: AbortSignal.timeout(5000)
            });

            if (response.ok) {
                const data = await response.json();
                this.backendOnline = data.model_loaded;

                if (this.backendOnline) {
                    this.elements.statusDot.className = 'status-dot online';
                    this.elements.statusText.textContent = `Pro Server: Online (${data.device})`;
                    this.elements.proModeBtn.classList.remove('disabled');
                    this.elements.setupInstructions.classList.add('hidden');
                } else {
                    this.elements.statusDot.className = 'status-dot offline';
                    this.elements.statusText.textContent = 'Pro Server: Model not loaded';
                    this.handleBackendOffline();
                }
            } else {
                throw new Error('Server error');
            }
        } catch (error) {
            console.log('Backend offline:', error.message);
            this.backendOnline = false;
            this.elements.statusDot.className = 'status-dot offline';
            this.elements.statusText.textContent = 'Pro Server: Offline';
            this.handleBackendOffline();
        }
    }

    handleBackendOffline() {
        // If Pro mode was selected but backend is offline, switch to basic
        if (this.analysisMode === 'pro') {
            this.setMode('basic');
        }
        this.elements.proModeBtn.classList.add('disabled');
        this.elements.setupInstructions.classList.remove('hidden');
    }

    setMode(mode) {
        if (mode === 'pro' && !this.backendOnline) {
            return; // Can't switch to pro if offline
        }

        this.analysisMode = mode;

        // Update button states
        this.elements.basicModeBtn.classList.toggle('active', mode === 'basic');
        this.elements.proModeBtn.classList.toggle('active', mode === 'pro');

        console.log(`Analysis mode set to: ${mode}`);
    }

    bindEvents() {
        // Mode toggle
        this.elements.basicModeBtn.addEventListener('click', () => this.setMode('basic'));
        this.elements.proModeBtn.addEventListener('click', () => {
            if (this.backendOnline) {
                this.setMode('pro');
            } else {
                this.checkBackendStatus();
            }
        });

        // Dropzone events
        this.elements.dropzone.addEventListener('click', () => this.elements.fileInput.click());
        this.elements.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));

        // Drag and drop
        this.elements.dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.elements.dropzone.classList.add('dragover');
        });

        this.elements.dropzone.addEventListener('dragleave', () => {
            this.elements.dropzone.classList.remove('dragover');
        });

        this.elements.dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.elements.dropzone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('image/')) {
                this.processFile(files[0]);
            }
        });

        // Webcam events
        this.elements.webcamBtn.addEventListener('click', () => this.startWebcam());
        this.elements.captureBtn.addEventListener('click', () => this.captureWebcam());
        this.elements.backBtn.addEventListener('click', () => this.stopWebcam());

        // Analyze new
        this.elements.analyzeNewBtn.addEventListener('click', () => this.reset());
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file && file.type.startsWith('image/')) {
            this.processFile(file);
        }
    }

    processFile(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            this.currentImageData = e.target.result;
            this.loadImage(e.target.result);
        };
        reader.readAsDataURL(file);
    }

    loadImage(src) {
        this.currentImageData = src;
        this.showSection('analysis');
        this.elements.previewImage.src = src;
        this.elements.previewImage.onload = () => {
            this.analyzeImage(this.elements.previewImage);
        };
    }

    async startWebcam() {
        try {
            this.webcamStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'user', width: 640, height: 480 }
            });
            this.elements.webcamVideo.srcObject = this.webcamStream;
            this.showSection('webcam');
        } catch (error) {
            console.error('Webcam error:', error);
            alert('Could not access webcam. Please ensure you have granted permission.');
        }
    }

    captureWebcam() {
        const video = this.elements.webcamVideo;
        const canvas = this.elements.webcamCanvas;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const ctx = canvas.getContext('2d');
        // Mirror the image
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0);

        const imageData = canvas.toDataURL('image/jpeg', 0.9);
        this.stopWebcam();
        this.loadImage(imageData);
    }

    stopWebcam() {
        if (this.webcamStream) {
            this.webcamStream.getTracks().forEach(track => track.stop());
            this.webcamStream = null;
        }
        this.showSection('upload');
    }

    showSection(section) {
        this.elements.uploadSection.classList.add('hidden');
        this.elements.webcamSection.classList.add('hidden');
        this.elements.analysisSection.classList.add('hidden');
        this.elements.loadingState.classList.remove('hidden');
        this.elements.results.classList.add('hidden');

        switch (section) {
            case 'upload':
                this.elements.uploadSection.classList.remove('hidden');
                break;
            case 'webcam':
                this.elements.webcamSection.classList.remove('hidden');
                break;
            case 'analysis':
                this.elements.analysisSection.classList.remove('hidden');
                break;
        }
    }

    async analyzeImage(imageElement) {
        // Show loading state
        this.elements.loadingState.classList.remove('hidden');
        this.elements.results.classList.add('hidden');
        this.elements.faceOverlay.innerHTML = '';

        // Update loading text based on mode
        if (this.analysisMode === 'pro') {
            this.elements.loadingText.textContent = 'Running CLIP AI analysis...';
        } else {
            this.elements.loadingText.textContent = 'Analyzing facial features...';
        }

        // Animate progress bar
        this.animateProgress();

        try {
            let results;

            if (this.analysisMode === 'pro' && this.backendOnline) {
                // Pro mode: use backend
                results = await this.analyzeWithBackend();
            } else {
                // Basic mode: use browser-based detection
                results = await this.detector.analyzeImage(imageElement);
                // Draw face boxes for basic mode
                this.drawFaceBoxes(results.detections, imageElement);
            }

            // Display results
            setTimeout(() => {
                this.displayResults(results, this.analysisMode);
            }, 1500);
        } catch (error) {
            console.error('Analysis failed:', error);
            this.displayError('Analysis failed. Please try another image.');
        }
    }

    async analyzeWithBackend() {
        try {
            const response = await fetch(`${this.backendUrl}/api/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image: this.currentImageData
                })
            });

            if (!response.ok) {
                throw new Error('Backend analysis failed');
            }

            const data = await response.json();

            if (data.success) {
                return {
                    hasFace: true,
                    faceCount: 1,
                    prediction: data.result.verdict,
                    confidence: data.result.confidence,
                    analysis: {
                        model: data.result.model,
                        device: data.result.device,
                        fakeProbability: data.result.fake_probability
                    },
                    factors: [],
                    detections: []
                };
            } else {
                throw new Error(data.error || 'Unknown error');
            }
        } catch (error) {
            console.error('Backend error:', error);
            // Fallback to basic analysis
            this.analysisMode = 'basic';
            return await this.detector.analyzeImage(this.elements.previewImage);
        }
    }

    animateProgress() {
        let progress = 0;
        this.elements.loadingProgress.style.width = '0%';

        const interval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress >= 100) {
                progress = 100;
                clearInterval(interval);
            }
            this.elements.loadingProgress.style.width = `${progress}%`;
        }, 200);
    }

    drawFaceBoxes(detections, imageElement) {
        const overlay = this.elements.faceOverlay;
        overlay.innerHTML = '';

        if (!detections || detections.length === 0) return;

        // Calculate scale
        const scaleX = imageElement.offsetWidth / imageElement.naturalWidth;
        const scaleY = imageElement.offsetHeight / imageElement.naturalHeight;

        detections.forEach((detection, index) => {
            const box = detection.detection.box;
            const faceBox = document.createElement('div');
            faceBox.className = 'face-box';
            faceBox.style.left = `${box.x * scaleX}px`;
            faceBox.style.top = `${box.y * scaleY}px`;
            faceBox.style.width = `${box.width * scaleX}px`;
            faceBox.style.height = `${box.height * scaleY}px`;
            faceBox.style.animationDelay = `${index * 0.1}s`;
            overlay.appendChild(faceBox);
        });
    }

    displayResults(results, mode) {
        this.elements.loadingState.classList.add('hidden');
        this.elements.results.classList.remove('hidden');

        const card = this.elements.resultCard;

        // Reset classes
        card.classList.remove('real', 'fake', 'uncertain');

        if (results.prediction === 'no_face') {
            this.displayNoFace();
            return;
        }

        // Set result type class
        card.classList.add(results.prediction);

        // Update analysis badge
        if (mode === 'pro') {
            this.elements.analysisBadge.className = 'analysis-badge pro';
            this.elements.analysisModeText.textContent = '⚡ Pro Analysis (CLIP AI)';
        } else {
            this.elements.analysisBadge.className = 'analysis-badge';
            this.elements.analysisModeText.textContent = '🔍 Quick Scan';
        }

        // Set icon
        const icons = {
            real: '✓',
            fake: '⚠',
            uncertain: '?'
        };
        this.elements.resultIcon.textContent = icons[results.prediction];

        // Set title
        const titles = {
            real: 'Likely Real',
            fake: 'Likely AI Generated',
            uncertain: 'Uncertain'
        };
        this.elements.resultTitle.textContent = titles[results.prediction];

        // Set subtitle
        const subtitles = {
            real: 'This face appears to be from a real photograph',
            fake: 'This face shows signs of AI generation',
            uncertain: 'Unable to make a confident determination'
        };
        this.elements.resultSubtitle.textContent = subtitles[results.prediction];

        // Set confidence
        this.elements.confidenceValue.textContent = `${results.confidence}%`;
        setTimeout(() => {
            this.elements.confidenceFill.style.width = `${results.confidence}%`;
        }, 100);

        // Set analysis details
        this.displayAnalysisDetails(results, mode);
    }

    displayAnalysisDetails(results, mode) {
        const container = this.elements.analysisDetails;
        container.innerHTML = '';

        let details;

        if (mode === 'pro') {
            details = [
                { label: 'Model', value: results.analysis.model || 'CLIP ViT-L/14' },
                { label: 'Confidence', value: `${results.confidence}%` },
                { label: 'Fake Probability', value: `${((results.analysis.fakeProbability || 0) * 100).toFixed(1)}%` },
                { label: 'Device', value: results.analysis.device || 'CPU' }
            ];
        } else {
            details = [
                { label: 'Faces Detected', value: results.faceCount || 0 },
                { label: 'Confidence', value: `${results.confidence}%` },
                { label: 'Symmetry', value: results.analysis?.symmetry?.isPerfect ? 'Too Perfect' : 'Natural' },
                { label: 'Eye Pattern', value: results.analysis?.eyeAnalysis?.identicalEyes ? 'Suspicious' : 'Normal' }
            ];
        }

        details.forEach(detail => {
            const item = document.createElement('div');
            item.className = 'detail-item';
            item.innerHTML = `
                <div class="label">${detail.label}</div>
                <div class="value">${detail.value}</div>
            `;
            container.appendChild(item);
        });

        // Add factors if any (basic mode)
        if (mode === 'basic' && results.factors && results.factors.length > 0) {
            const factorsTitle = document.createElement('div');
            factorsTitle.className = 'detail-item';
            factorsTitle.style.gridColumn = '1 / -1';
            factorsTitle.innerHTML = `
                <div class="label">Detection Factors</div>
                <div class="value">${results.factors.map(f => f.name).join(', ')}</div>
            `;
            container.appendChild(factorsTitle);
        }
    }

    displayNoFace() {
        const card = this.elements.resultCard;
        card.classList.add('uncertain');

        this.elements.analysisBadge.className = 'analysis-badge';
        this.elements.analysisModeText.textContent = this.analysisMode === 'pro' ? '⚡ Pro Analysis' : '🔍 Quick Scan';

        this.elements.resultIcon.textContent = '👤';
        this.elements.resultTitle.textContent = 'No Face Detected';
        this.elements.resultSubtitle.textContent = 'Please upload an image with a clearly visible face';

        this.elements.confidenceValue.textContent = 'N/A';
        this.elements.confidenceFill.style.width = '0%';

        this.elements.analysisDetails.innerHTML = `
            <div class="detail-item" style="grid-column: 1 / -1;">
                <div class="label">Tip</div>
                <div class="value">Try an image with better lighting and a front-facing view</div>
            </div>
        `;
    }

    displayError(message) {
        this.elements.loadingState.classList.add('hidden');
        this.elements.results.classList.remove('hidden');

        const card = this.elements.resultCard;
        card.classList.add('uncertain');

        this.elements.resultIcon.textContent = '❌';
        this.elements.resultTitle.textContent = 'Error';
        this.elements.resultSubtitle.textContent = message;

        this.elements.confidenceValue.textContent = 'N/A';
        this.elements.confidenceFill.style.width = '0%';
        this.elements.analysisDetails.innerHTML = '';
    }

    reset() {
        this.elements.fileInput.value = '';
        this.elements.previewImage.src = '';
        this.elements.faceOverlay.innerHTML = '';
        this.elements.loadingProgress.style.width = '0%';
        this.elements.confidenceFill.style.width = '0%';
        this.currentImageData = null;
        this.showSection('upload');
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new DeepFakeApp();
});
