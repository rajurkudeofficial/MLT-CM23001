# Machine Learning Assignments

This folder contains three assignments focused on transfer learning and pre-trained model usage with TensorFlow.js in a browser environment.

---

## Assignment 1: Load MobileNet and Classify a Static Image

### Aim
To load a pre-trained MobileNet model using TensorFlow.js and classify a static image, displaying the top-3 predicted classes with confidence scores.

### Tools and Technologies Used
- **Programming Language:** JavaScript (ES6+)
- **ML Library:** TensorFlow.js (v4.11.0) + MobileNet v2.1.0 via CDN
- **Development Environment:** VS Code + Web Browser
- **Model:** MobileNet v1 (Pre-trained on ImageNet)
- **Execution:** Client-side browser-based

### Description

#### MobileNet Architecture
MobileNet is a lightweight deep learning model designed for mobile and embedded vision applications. Key features include:

- **Depthwise Separable Convolutions:** Reduces computation and model size
- **Parameters:** Approximately 4.2 million
- **Input Size:** 224×224×3 (RGB images)
- **Output:** 1000 ImageNet classes
- **Trade-off:** Balances accuracy with speed and efficiency

#### Implementation Details
The assignment demonstrates:
1. Loading MobileNet via TensorFlow.js CDN
2. Preprocessing images for model input
3. Running inference to classify images
4. Extracting top-3 predictions with confidence scores
5. Displaying results both on webpage and in console

### Steps to Execute

1. Navigate to `Assignment/Assig.1/`
2. Open `index.html` in a web browser
3. Click **"Load MobileNet Model"** button
4. Wait for model to download (~5-10 seconds)
5. Image loads automatically
6. Click **"Classify Image"** button
7. View top-3 predictions displayed on webpage
8. Press F12 to view detailed console logs

### Output

**Expected Console Output:**
```
=== LOADING MOBILENET MODEL ===
📦 Downloading pre-trained MobileNet...
✅ Model loaded successfully!

=== CLASSIFYING IMAGE ===
🔍 Running MobileNet inference...
⏱️ Inference time: 24.56ms

📊 TOP-3 PREDICTIONS:
──────────────────────────────────────────────
1. tabby cat
   Confidence: 45.23%
   Probability: 0.452300
──────────────────────────────────────────────
2. Egyptian cat
   Confidence: 32.15%
   Probability: 0.321500
──────────────────────────────────────────────
3. tiger cat
   Confidence: 18.67%
   Probability: 0.186700
──────────────────────────────────────────────
```

**Visual Output:**
- Image displayed on webpage
- Top-3 predictions shown as colored cards with:
  - Rank number
  - Class name
  - Confidence percentage

### Conclusion

This assignment successfully demonstrates transfer learning using a pre-trained MobileNet model in a browser environment. Key learnings include:

1. **Transfer Learning Benefits:** Pre-trained models eliminate the need for training from scratch
2. **Browser-based ML:** TensorFlow.js enables ML inference entirely in the browser
3. **Real-time Classification:** Fast inference (~20-30ms) suitable for interactive applications
4. **Accessibility:** No Python or backend setup required

**Practical Applications:**
- Image recognition in web apps
- Content moderation
- Visual search
- Accessibility tools (image descriptions)

---

## Assignment 2: Test Classification Accuracy on Multiple Images

### Aim
To evaluate the classification accuracy of MobileNet across multiple diverse images and analyze prediction performance.

### Tools and Technologies Used
- **Programming Language:** JavaScript (ES6+)
- **ML Library:** TensorFlow.js + MobileNet via CDN
- **Development Environment:** VS Code + Web Browser
- **Test Set:** 5 diverse images from different object categories
- **Evaluation Metrics:** Accuracy, confidence scores, inference time

### Description

#### Testing Methodology
This assignment extends Assignment 1 by:
1. Classifying multiple images (5 different objects)
2. Tracking prediction accuracy
3. Calculating average confidence scores
4. Measuring inference time consistency
5. Analyzing model performance across categories

#### Test Image Categories
The test set includes:
1. **Animals:** Cat, Dog
2. **Objects:** Coffee cup, Wristwatch, Sunglasses

Each image tests the model's ability to recognize distinct object categories from the ImageNet dataset.

### Steps to Execute

1. Navigate to `Assignment/Assig.2/`
2. Open `index.html` in a web browser
3. Click **"Load MobileNet Model"**
4. Click **"Classify All Images"**
5. Watch as each image is classified sequentially
6. View results in the comparison table
7. Read the automated accuracy analysis
8. Check console (F12) for detailed logs

### Output

**Expected Console Output:**
```
=== CLASSIFYING MULTIPLE IMAGES ===
📷 Total images to classify: 5
──────────────────────────────────────────────────

🖼️ Image 1/5: Cat Image
   ⏱️ Inference time: 23.45ms
   🎯 Top Prediction: tabby cat
   📊 Confidence: 48.56%
   ✅ Expected: cat | Correct: YES
   📋 Top-3 Predictions:
      1. tabby cat (48.56%)
      2. Egyptian cat (28.34%)
      3. tiger cat (15.23%)
──────────────────────────────────────────────────

[... Similar output for images 2-5 ...]

=== ACCURACY ANALYSIS ===
📊 Accuracy: 5/5 (100.00%)
📈 Average Confidence: 67.45%
⏱️ Average Inference Time: 25.67ms
```

**Visual Output:**
- Comprehensive table showing:
  - Image thumbnails
  - Predicted classes
  - Expected classes (with ✅/❌ indicators)
  - Confidence bars (visual representation)
  - Inference times
- Accuracy Analysis section with:
  - Overall accuracy percentage
  - Average confidence score
  - Performance observations
  - Model behavior insights

### Conclusion

This assignment provides quantitative evaluation of MobileNet's performance across diverse object categories. Key findings:

1. **Accuracy:** MobileNet typically achieves 80-100% accuracy on clear, well-framed ImageNet classes
2. **Consistency:** Inference time remains stable across images (~20-30ms)
3. **Confidence Patterns:** Higher confidence for distinctive objects, lower for ambiguous cases
4. **Limitations:** Performance depends on image quality, object visibility, and dataset representation

**Insights:**
- Model excels at common objects in ImageNet training set
- Confidence scores provide reliability indicators
- Real-world deployment requires consideration of edge cases
- Browser-based inference is viable for production applications

---

## Assignment 3: Compare MobileNet with Another Pre-trained Model

### Aim
To compare MobileNet with ResNet50 by classifying the same image(s) and analyzing differences in predictions, confidence scores, and performance characteristics.

### Tools and Technologies Used
- **Programming Language:** JavaScript (ES6+)
- **ML Libraries:** 
  - TensorFlow.js (v4.11.0)
  - MobileNet v2.1.0
  - ResNet50 (via TensorFlow Hub)
- **Development Environment:** VS Code + Web Browser
- **Comparison Metrics:** Accuracy, confidence, inference time, model agreement

### Description

#### Model Comparison Overview

**MobileNet v1:**
- **Architecture:** Depthwise Separable Convolutions
- **Parameters:** ~4.2 million
- **Optimization:** Speed and efficiency
- **Best For:** Mobile/web applications, real-time inference
- **Trade-off:** Slightly lower accuracy for faster speed

**ResNet50:**
- **Architecture:** Residual Network with 50 layers
- **Parameters:** ~25.6 million
- **Optimization:** Maximum accuracy
- **Best For:** Server-side processing, accuracy-critical applications
- **Trade-off:** Slower inference for higher accuracy

#### Comparison Criteria
1. **Prediction Agreement:** Do both models agree on the top class?
2. **Confidence Levels:** Which model is more confident?
3. **Inference Speed:** How much faster is MobileNet?
4. **Top-3 Predictions:** Do they identify similar alternative classes?

### Steps to Execute

1. Navigate to `Assignment/Assig.3/`
2. Open `index.html` in a web browser
3. Click **"Load Both Models"**
4. Wait for both models to download (ResNet50 takes longer)
5. Click **"Compare Predictions"**
6. View side-by-side comparison in table
7. Read detailed comparative analysis
8. Check console for technical comparison metrics

### Output

**Expected Console Output:**
```
=== MODEL COMPARISON ===
🖼️ Image URL: [...]

📱 MOBILENET PREDICTION:
──────────────────────────────────────────────────
⏱️ Inference Time: 24.32ms
🎯 Top-3 Predictions:
   1. tabby cat - 45.67%
   2. Egyptian cat - 28.45%
   3. tiger cat - 16.23%

🏗️ RESNET50 PREDICTION:
──────────────────────────────────────────────────
⏱️ Inference Time: 67.89ms
🎯 Top-3 Predictions:
   1. tabby cat - 52.34%
   2. Egyptian cat - 31.22%
   3. Persian cat - 12.11%

📊 COMPARISON ANALYSIS:
──────────────────────────────────────────────────
⚡ Speed Comparison:
   MobileNet: 24.32ms
   ResNet50: 67.89ms
   ResNet50 is 279% of MobileNet's inference time

🎯 Prediction Agreement:
   MobileNet Top-1: tabby cat (45.67%)
   ResNet50 Top-1: tabby cat (52.34%)
   Models AGREE on top prediction
   Confidence Difference: 6.67%
```

**Visual Output:**

1. **Comparison Table:**
   - Model names with badges
   - Top predictions
   - Confidence bars (color-coded)
   - Inference times

2. **Top-3 Predictions Breakdown:**
   - Side-by-side cards for each model
   - Detailed prediction percentages
   - Visual comparison of alternative predictions

3. **Comparative Analysis Text:**
   - **Model Agreement:** Whether predictions match
   - **Performance Comparison:** Speed differences quantified
   - **Confidence Analysis:** Which model is more certain
   - **Key Takeaways:** Practical recommendations

### Conclusion

This assignment provides comprehensive comparison between two state-of-the-art models with different design philosophies. Key conclusions:

#### Performance Trade-offs
1. **Speed vs Accuracy:** MobileNet is 2-3× faster but may have slightly lower accuracy
2. **Resource Requirements:** MobileNet uses 6× fewer parameters
3. **Use Case Alignment:** Choose based on application constraints

#### Model Agreement
- Both models typically agree on obvious objects
- Disagreements occur with:
  - Ambiguous images
  - Multiple objects
  - Poor image quality
  - Uncommon object combinations

#### Practical Recommendations

**Choose MobileNet for:**
- Mobile applications
- Real-time video processing
- Resource-constrained environments
- User-facing interactive features
- Applications where speed > absolute accuracy

**Choose ResNet50 for:**
- Server-side batch processing
- Accuracy-critical applications
- Applications with abundant computational resources
- When processing time is not a constraint
- Medical imaging, security systems

**Hybrid Approaches:**
- Use MobileNet for initial screening
- Route uncertain cases to ResNet50
- Best of both worlds: speed + accuracy when needed

---

## General Notes

### File Structure
```
Assignment/
├── Assig.1/
│   ├── index.html
│   ├── model.js
│   └── output.png (screenshot - optional)
├── Assig.2/
│   ├── index.html
│   ├── model.js
│   └── output.png (screenshot - optional)
├── Assig.3/
│   ├── index.html
│   ├── model.js
│   └── output.png (screenshot - optional)
└── README.md
```

### Running the Assignments

**No Installation Required:**
- All assignments run directly in the browser
- No Python, Node.js, or npm needed
- Internet connection required for CDN resources

**Recommended Browsers:**
- Google Chrome (recommended)
- Microsoft Edge
- Mozilla Firefox
- Safari (with WebGL support)

**Developer Tools:**
- Press F12 to open Developer Console
- View detailed logs, predictions, and metrics
- Monitor network requests and model loading

### Common Issues and Solutions

**Issue:** Model loading fails
- **Solution:** Check internet connection, try different browser, disable ad blockers

**Issue:** Slow inference on first run
- **Solution:** Normal - models are being downloaded and cached

**Issue:** Image not loading
- **Solution:** Check CORS policy, use images from allowed domains

**Issue:** ResNet50 unavailable (Assignment 3)
- **Solution:** Code includes fallback with simulated comparison

### Learning Outcomes

After completing these assignments, students will understand:
1. Transfer learning and pre-trained models
2. Browser-based machine learning with TensorFlow.js
3. Model comparison and performance evaluation
4. Practical trade-offs in model selection
5. Real-world deployment considerations
6. Inference optimization techniques

---

**Submitted by:** [Student Name]  
**Roll Number:** [Roll No]  
**Date:** [Submission Date]  
**Subject:** Machine Learning Laboratory  
