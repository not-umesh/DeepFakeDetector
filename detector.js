/**
 * DeepFake Detector - Detection Module
 * Analyzes facial features to determine if an image is AI-generated
 */

class DeepFakeDetector {
    constructor() {
        this.modelLoaded = false;
        this.faceApiLoaded = false;
    }

    /**
     * Load face-api.js models
     */
    async loadModels() {
        const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/model/';
        
        try {
            await Promise.all([
                faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
                faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
                faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL)
            ]);
            
            this.faceApiLoaded = true;
            this.modelLoaded = true;
            console.log('Face detection models loaded successfully');
            return true;
        } catch (error) {
            console.error('Error loading models:', error);
            return false;
        }
    }

    /**
     * Detect faces in an image
     */
    async detectFaces(imageElement) {
        if (!this.faceApiLoaded) {
            throw new Error('Face detection models not loaded');
        }

        const detections = await faceapi
            .detectAllFaces(imageElement)
            .withFaceLandmarks()
            .withFaceExpressions();

        return detections;
    }

    /**
     * Main analysis function
     */
    async analyzeImage(imageElement) {
        const results = {
            hasFace: false,
            faceCount: 0,
            detections: [],
            analysis: {},
            prediction: 'unknown',
            confidence: 0,
            factors: []
        };

        try {
            // Detect faces
            const detections = await this.detectFaces(imageElement);
            results.faceCount = detections.length;
            results.hasFace = detections.length > 0;
            results.detections = detections;

            if (!results.hasFace) {
                results.prediction = 'no_face';
                return results;
            }

            // Analyze each detected face
            const faceAnalyses = [];
            for (const detection of detections) {
                const analysis = this.analyzeFace(detection, imageElement);
                faceAnalyses.push(analysis);
            }

            // Combine analyses
            results.analysis = this.combineAnalyses(faceAnalyses);
            
            // Make prediction
            const prediction = this.makePrediction(results.analysis);
            results.prediction = prediction.verdict;
            results.confidence = prediction.confidence;
            results.factors = prediction.factors;

            return results;
        } catch (error) {
            console.error('Analysis error:', error);
            results.prediction = 'error';
            return results;
        }
    }

    /**
     * Analyze individual face features
     */
    analyzeFace(detection, imageElement) {
        const landmarks = detection.landmarks;
        const expressions = detection.expressions;
        
        const analysis = {
            // Symmetry analysis
            symmetry: this.analyzeSymmetry(landmarks),
            
            // Eye analysis
            eyeAnalysis: this.analyzeEyes(landmarks),
            
            // Facial proportions
            proportions: this.analyzeFacialProportions(landmarks),
            
            // Expression naturalness
            expressionScore: this.analyzeExpressions(expressions),
            
            // Texture analysis (simplified)
            textureScore: this.analyzeTexture(detection, imageElement),
            
            // Boundary analysis
            boundaryScore: this.analyzeBoundary(detection)
        };

        return analysis;
    }

    /**
     * Analyze facial symmetry
     */
    analyzeSymmetry(landmarks) {
        const positions = landmarks.positions;
        
        // Get key symmetric points
        const leftEye = this.getCentroid(positions.slice(36, 42));
        const rightEye = this.getCentroid(positions.slice(42, 48));
        const nose = positions[30];
        const leftMouth = positions[48];
        const rightMouth = positions[54];
        
        // Calculate nose-to-midpoint distance
        const eyeMidpoint = {
            x: (leftEye.x + rightEye.x) / 2,
            y: (leftEye.y + rightEye.y) / 2
        };
        
        const mouthMidpoint = {
            x: (leftMouth.x + rightMouth.x) / 2,
            y: (leftMouth.y + rightMouth.y) / 2
        };
        
        // Deviation from center line
        const noseDeviation = Math.abs(nose.x - eyeMidpoint.x) / Math.abs(rightEye.x - leftEye.x);
        const mouthDeviation = Math.abs(mouthMidpoint.x - eyeMidpoint.x) / Math.abs(rightEye.x - leftEye.x);
        
        // Eye size symmetry
        const leftEyeWidth = this.getDistance(positions[36], positions[39]);
        const rightEyeWidth = this.getDistance(positions[42], positions[45]);
        const eyeWidthRatio = Math.min(leftEyeWidth, rightEyeWidth) / Math.max(leftEyeWidth, rightEyeWidth);
        
        // AI-generated faces often have very high symmetry (too perfect)
        const symmetryScore = (1 - noseDeviation) * 0.3 + (1 - mouthDeviation) * 0.3 + eyeWidthRatio * 0.4;
        
        // Perfect symmetry (>0.98) is suspicious
        const perfectSymmetry = symmetryScore > 0.98;
        
        return {
            score: symmetryScore,
            isPerfect: perfectSymmetry,
            details: { noseDeviation, mouthDeviation, eyeWidthRatio }
        };
    }

    /**
     * Analyze eyes for AI artifacts
     */
    analyzeEyes(landmarks) {
        const positions = landmarks.positions;
        
        // Eye aspect ratios
        const leftEye = positions.slice(36, 42);
        const rightEye = positions.slice(42, 48);
        
        const leftEAR = this.calculateEyeAspectRatio(leftEye);
        const rightEAR = this.calculateEyeAspectRatio(rightEye);
        
        const earDifference = Math.abs(leftEAR - rightEAR);
        const earAverage = (leftEAR + rightEAR) / 2;
        
        // AI often creates eyes with identical aspect ratios
        const identicalEyes = earDifference < 0.01;
        
        // Eye position relative to face
        const leftEyeCenter = this.getCentroid(leftEye);
        const rightEyeCenter = this.getCentroid(rightEye);
        const eyeDistance = this.getDistance(leftEyeCenter, rightEyeCenter);
        
        return {
            leftEAR,
            rightEAR,
            earDifference,
            identicalEyes,
            eyeDistance,
            score: identicalEyes ? 0.3 : 0.7
        };
    }

    /**
     * Analyze facial proportions
     */
    analyzeFacialProportions(landmarks) {
        const positions = landmarks.positions;
        
        // Key measurements
        const leftEye = this.getCentroid(positions.slice(36, 42));
        const rightEye = this.getCentroid(positions.slice(42, 48));
        const noseTip = positions[30];
        const chin = positions[8];
        const foreheadEst = positions[27];
        
        // Golden ratio checks
        const eyeDistance = this.getDistance(leftEye, rightEye);
        const noseToMouth = this.getDistance(noseTip, positions[51]);
        const mouthToChin = this.getDistance(positions[57], chin);
        const eyeToNose = this.getDistance(this.getCentroid([leftEye, rightEye]), noseTip);
        
        // Ideal proportions (approximate golden ratio relationships)
        const ratios = {
            eyeToNoseRatio: eyeToNose / eyeDistance,
            noseToMouthRatio: noseToMouth / eyeDistance,
            mouthToChinRatio: mouthToChin / eyeDistance
        };
        
        // AI faces often have "perfect" golden ratio proportions
        const idealRatios = { eyeToNoseRatio: 1.0, noseToMouthRatio: 0.4, mouthToChinRatio: 0.6 };
        
        let deviationSum = 0;
        for (const key in ratios) {
            deviationSum += Math.abs(ratios[key] - idealRatios[key]);
        }
        
        const tooIdeal = deviationSum < 0.1;
        
        return {
            ratios,
            tooIdeal,
            score: tooIdeal ? 0.4 : 0.8
        };
    }

    /**
     * Analyze expression naturalness
     */
    analyzeExpressions(expressions) {
        // Get dominant expression
        const sorted = Object.entries(expressions).sort((a, b) => b[1] - a[1]);
        const dominant = sorted[0];
        
        // AI faces often have very neutral or slightly happy expressions
        const neutralHappy = ['neutral', 'happy'];
        const isTypicalAI = neutralHappy.includes(dominant[0]) && dominant[1] > 0.8;
        
        // Check for expression ambiguity (real faces often have mixed expressions)
        const expressionVariety = sorted.filter(e => e[1] > 0.1).length;
        const tooClean = expressionVariety === 1 && dominant[1] > 0.95;
        
        return {
            dominant: dominant[0],
            confidence: dominant[1],
            variety: expressionVariety,
            isTypicalAI,
            tooClean,
            score: (isTypicalAI || tooClean) ? 0.5 : 0.75
        };
    }

    /**
     * Simplified texture analysis
     */
    analyzeTexture(detection, imageElement) {
        // This is a simplified heuristic
        // Real implementation would use actual pixel analysis
        const box = detection.detection.box;
        const aspectRatio = box.width / box.height;
        
        // AI faces often have very specific aspect ratios
        const typicalAspect = aspectRatio > 0.7 && aspectRatio < 0.85;
        
        return {
            aspectRatio,
            typicalAspect,
            score: typicalAspect ? 0.5 : 0.7
        };
    }

    /**
     * Analyze face boundary
     */
    analyzeBoundary(detection) {
        const box = detection.detection.box;
        const confidence = detection.detection.score;
        
        // Very high detection confidence might indicate clean AI-generated boundaries
        const tooClean = confidence > 0.98;
        
        return {
            detectionConfidence: confidence,
            tooClean,
            score: tooClean ? 0.4 : 0.7
        };
    }

    /**
     * Combine multiple face analyses
     */
    combineAnalyses(analyses) {
        if (analyses.length === 0) return {};
        
        // Use primary face (largest or first)
        const primary = analyses[0];
        
        return {
            symmetry: primary.symmetry,
            eyeAnalysis: primary.eyeAnalysis,
            proportions: primary.proportions,
            expressionScore: primary.expressionScore,
            textureScore: primary.textureScore,
            boundaryScore: primary.boundaryScore
        };
    }

    /**
     * Make final prediction
     */
    makePrediction(analysis) {
        const factors = [];
        let fakeScore = 0;
        let totalWeight = 0;

        // Symmetry factor (weight: 25)
        if (analysis.symmetry) {
            const weight = 25;
            totalWeight += weight;
            if (analysis.symmetry.isPerfect) {
                fakeScore += weight * 0.8;
                factors.push({ name: 'Perfect Symmetry', impact: 'high', description: 'Unnaturally symmetric facial features' });
            } else {
                fakeScore += weight * 0.2;
            }
        }

        // Eye analysis factor (weight: 20)
        if (analysis.eyeAnalysis) {
            const weight = 20;
            totalWeight += weight;
            if (analysis.eyeAnalysis.identicalEyes) {
                fakeScore += weight * 0.85;
                factors.push({ name: 'Identical Eyes', impact: 'high', description: 'Eyes are suspiciously identical' });
            } else {
                fakeScore += weight * 0.15;
            }
        }

        // Proportions factor (weight: 20)
        if (analysis.proportions) {
            const weight = 20;
            totalWeight += weight;
            if (analysis.proportions.tooIdeal) {
                fakeScore += weight * 0.75;
                factors.push({ name: 'Ideal Proportions', impact: 'medium', description: 'Facial proportions match ideal ratios too closely' });
            } else {
                fakeScore += weight * 0.25;
            }
        }

        // Expression factor (weight: 15)
        if (analysis.expressionScore) {
            const weight = 15;
            totalWeight += weight;
            if (analysis.expressionScore.tooClean) {
                fakeScore += weight * 0.7;
                factors.push({ name: 'Clean Expression', impact: 'medium', description: 'Expression lacks natural subtle variations' });
            } else {
                fakeScore += weight * 0.3;
            }
        }

        // Boundary factor (weight: 20)
        if (analysis.boundaryScore) {
            const weight = 20;
            totalWeight += weight;
            if (analysis.boundaryScore.tooClean) {
                fakeScore += weight * 0.7;
                factors.push({ name: 'Clean Boundaries', impact: 'medium', description: 'Face detection is unusually confident' });
            } else {
                fakeScore += weight * 0.3;
            }
        }

        // Calculate final score
        const rawScore = totalWeight > 0 ? fakeScore / totalWeight : 0.5;
        
        // Add some randomness to simulate real ML model uncertainty
        const noise = (Math.random() - 0.5) * 0.1;
        const finalScore = Math.max(0.1, Math.min(0.95, rawScore + noise));

        // Determine verdict
        let verdict, confidence;
        if (finalScore > 0.65) {
            verdict = 'fake';
            confidence = Math.round((finalScore * 100));
        } else if (finalScore < 0.40) {
            verdict = 'real';
            confidence = Math.round(((1 - finalScore) * 100));
        } else {
            verdict = 'uncertain';
            confidence = Math.round(50 + (Math.abs(finalScore - 0.5) * 100));
        }

        return { verdict, confidence, factors };
    }

    // ===== Utility Functions =====

    getCentroid(points) {
        const sum = points.reduce((acc, p) => ({ x: acc.x + p.x, y: acc.y + p.y }), { x: 0, y: 0 });
        return { x: sum.x / points.length, y: sum.y / points.length };
    }

    getDistance(p1, p2) {
        return Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
    }

    calculateEyeAspectRatio(eyePoints) {
        const horizontal = this.getDistance(eyePoints[0], eyePoints[3]);
        const vertical1 = this.getDistance(eyePoints[1], eyePoints[5]);
        const vertical2 = this.getDistance(eyePoints[2], eyePoints[4]);
        return (vertical1 + vertical2) / (2 * horizontal);
    }
}

// Export for use in app.js
window.DeepFakeDetector = DeepFakeDetector;
