# Practical 2: MNIST Digit Recognition using CNN

## Aim
To train a Convolutional Neural Network (CNN) on the MNIST handwritten digit dataset using TensorFlow.js and evaluate predictions in a browser-based environment.

## Tools and Technologies Used
- **Programming Language:** JavaScript (ES6+)
- **Machine Learning Library:** TensorFlow.js (v4.11.0) loaded via CDN
- **Development Environment:** VS Code + Web Browser (Chrome/Edge/Firefox)
- **Dataset:** MNIST (Modified National Institute of Standards and Technology)
- **Execution Environment:** Client-side (Browser-based, no backend required)

## Description

### Overview
This practical demonstrates the implementation of a Convolutional Neural Network for handwritten digit classification using TensorFlow.js. The entire pipeline—from data loading to model training and prediction—runs entirely in the browser without requiring any server-side processing or Python installation.

### MNIST Dataset
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9):
- **Training Set:** 60,000 images
- **Test Set:** 10,000 images
- **Image Dimensions:** 28×28 pixels
- **Pixel Values:** Normalized to range [0, 1]

For browser performance optimization, this implementation uses:
- **5,000 training samples**
- **1,000 test samples**

### Model Architecture

The CNN model consists of the following layers:

1. **First Convolutional Block:**
   - Conv2D layer: 32 filters, 3×3 kernel, ReLU activation
   - MaxPooling2D: 2×2 pool size

2. **Second Convolutional Block:**
   - Conv2D layer: 64 filters, 3×3 kernel, ReLU activation
   - MaxPooling2D: 2×2 pool size

3. **Fully Connected Layers:**
   - Flatten layer
   - Dense layer: 128 units, ReLU activation
   - Output layer: 10 units (one per digit class), Softmax activation

**Total Parameters:** Approximately 93,000 trainable parameters

### Training Configuration
- **Optimizer:** Adam (learning rate: 0.001)
- **Loss Function:** Categorical Cross-Entropy
- **Metrics:** Accuracy
- **Batch Size:** 128
- **Epochs:** 5
- **Validation Split:** 15%

### Features
1. **Data Loading:** Automatic download and preprocessing of MNIST dataset from TensorFlow.js hosted sprites
2. **Model Training:** Real-time training with epoch-by-epoch metrics display
3. **Visualization:** Canvas-based rendering of predicted digit images
4. **Evaluation:** Test set accuracy and per-class probability distribution
5. **Interactive Prediction:** Random sample selection for prediction demonstration

## Steps to Execute

### Prerequisites
- Modern web browser (Chrome, Edge, Firefox recommended)
- Text editor (VS Code recommended)
- Active internet connection (for loading TensorFlow.js CDN)

### Execution Steps

1. **Setup Files:**
   ```
   Practical-2/
   ├── index.html
   ├── model.js
   └── README.md
   ```

2. **Open in Browser:**
   - Navigate to the `Practical-2` folder
   - Double-click `index.html` to open in browser
   - Alternatively, right-click → Open with → [Your Browser]

3. **Load Data:**
   - Click the **"Load MNIST Data"** button
   - Wait for the dataset to download (approximately 10-15 seconds)
   - Console will display data loading progress

4. **Train Model:**
   - Click the **"Train Model"** button
   - Training will begin and run for 5 epochs
   - Monitor progress in:
     - Status bar (epoch, accuracy)
     - Training metrics cards (accuracy, loss)
     - Browser console (F12) for detailed logs

5. **Make Predictions:**
   - After training completes, click **"Predict Random Digit"**
   - A random test image will be displayed
   - Predicted digit and confidence score appear below the image
   - Check console for detailed probability distribution across all 10 classes

6. **View Console Logs:**
   - Press **F12** to open Developer Tools
   - Navigate to **Console** tab
   - View detailed metrics including:
     - Training progress per epoch
     - Validation accuracy and loss
     - Test set evaluation results
     - Per-digit prediction probabilities

## Output

### Expected Results

**Training Metrics (after 5 epochs):**
- Training Accuracy: ~95-98%
- Validation Accuracy: ~93-96%
- Test Accuracy: ~93-96%
- Training Loss: ~0.10-0.15

**Prediction Output:**
- Visual display of 28×28 digit image (upscaled to 280×280 for visibility)
- Predicted digit label (0-9)
- Confidence score (typically 90-99% for correct predictions)

**Console Output Example:**
```
=== TRAINING CNN MODEL ===
📈 Epoch 1/5
   Loss: 0.3245
   Accuracy: 90.12%
   Val Loss: 0.1823
   Val Accuracy: 94.56%

📈 Epoch 5/5
   Loss: 0.0876
   Accuracy: 97.34%
   Val Loss: 0.1234
   Val Accuracy: 95.67%

✅ Training completed!

🧪 EVALUATING ON TEST SET:
   Test Loss: 0.1289
   Test Accuracy: 95.20%

=== MAKING PREDICTION ===
🎯 Predicted Label: 7
📊 Confidence: 98.45%
✅ Correct: YES

📊 Class Probabilities:
   Digit 0: 0.12%
   Digit 1: 0.05%
   Digit 2: 0.23%
   Digit 3: 0.08%
   Digit 4: 0.15%
   Digit 5: 0.31%
   Digit 6: 0.09%
   Digit 7: 98.45%
   Digit 8: 0.42%
   Digit 9: 0.10%
```

### Visual Output
1. **Training Metrics Dashboard:** Real-time display of epoch number, training accuracy, training loss, and validation accuracy
2. **Digit Canvas:** 280×280 pixel canvas showing the grayscale handwritten digit
3. **Prediction Card:** Highlighted result card with predicted digit (large font) and confidence percentage

## Conclusion

This practical successfully demonstrates the implementation of a Convolutional Neural Network for image classification using TensorFlow.js in a browser environment. Key takeaways include:

1. **Browser-based Deep Learning:** TensorFlow.js enables complete ML pipelines to run client-side without Python or backend infrastructure.

2. **CNN Effectiveness:** The convolutional architecture effectively learns spatial hierarchies in image data, achieving ~95% accuracy on MNIST digit classification.

3. **Real-time Training:** Despite browser constraints, the model trains efficiently on a subset of MNIST in approximately 2-3 minutes on modern hardware.

4. **Interpretability:** The probability distribution output provides insight into model confidence and potential confusion between similar digits (e.g., 3 vs 8, 4 vs 9).

5. **Practical Applications:** This approach can be extended to:
   - Custom digit recognition systems
   - Real-time handwriting recognition
   - Educational demonstrations of neural networks
   - Privacy-preserving ML (data never leaves the browser)

### Performance Considerations
- Using full MNIST (60,000 images) would require significantly more training time
- GPU acceleration via WebGL improves performance dramatically
- Model can be saved using `model.save()` for reuse without retraining

### Future Enhancements
- Data augmentation (rotation, scaling, noise)
- Deeper architectures (additional convolutional layers)
- Dropout layers for regularization
- Learning rate scheduling
- Real-time drawing canvas for user input
- Model comparison with different architectures

---

**Submitted by:** [Student Name]  
**Roll Number:** [Roll No]  
**Date:** [Submission Date]  
**Subject:** Machine Learning Laboratory  
