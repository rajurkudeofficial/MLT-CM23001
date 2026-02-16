# MNIST Digit Recognition – CNN and Dense Network Comparison

## 📌 Overview
This practical demonstrates handwritten digit recognition using the **MNIST dataset** and compares the performance of a **Convolutional Neural Network (CNN)** with a **simple Dense (ANN) network**.

The CNN model is trained offline using **Kaggle GPU (Tesla T4 ×2)** for fast and accurate training.  
The trained model is then deployed in the browser using **TensorFlow.js** for real-time digit classification and performance comparison.

---

## 🎯 Aim
To train a CNN model on the MNIST dataset, classify handwritten digits using a web-based interface, and compare its performance with a simple dense neural network.

---

## 🎯 Objectives
- To train a CNN on the MNIST dataset for 5 epochs and record accuracy
- To classify user-drawn digits using a trained model
- To compare accuracy and performance of CNN and Dense networks
- To understand why CNN performs better for image classification tasks

---

## 🧠 Task-wise Description

---

## ✅ Task 1: Train a CNN on MNIST and Report Accuracy

### Description
A Convolutional Neural Network (CNN) was trained on the MNIST dataset using Kaggle notebooks with **dual Tesla T4 GPU acceleration**. The model was trained for **5 epochs** to achieve high accuracy.

### Output
- Training Epochs: 5  
- Final Accuracy: ~98–99%  

### Observation
Accuracy improves with each epoch due to better feature learning from handwritten digits.

---

## ✅ Task 2: Draw a Digit on Canvas and Classify It

### Description
The trained CNN model was converted into **TensorFlow.js format** and loaded into the browser using JavaScript and CDN.  
Users can draw digits (0–9) on an HTML canvas, and the model predicts the digit in real time.

### Processing Steps
Canvas Image
→ Grayscale Conversion
→ Resize to 28×28
→ Normalization (0–1)
→ CNN Prediction

### Output
- Predicted digit is displayed in the **browser console**

---

## ✅ Task 3: Compare CNN and Dense Network Performance

### Description
A simple Dense (ANN) network and a CNN were trained separately on the same MNIST dataset for 5 epochs.  
Their accuracies were compared and displayed on a web page.

### Accuracy Comparison

| Model | Epochs | Accuracy |
|------|------|---------|
| Dense Network (ANN) | 5 | ~92–94% |
| CNN | 5 | ~98–99% |

### Comparison Summary

| Feature | Dense Network | CNN |
|------|------|------|
| Spatial Feature Learning | ❌ No | ✅ Yes |
| Image Understanding | Poor | Excellent |
| Accuracy | Lower | Higher |

---

## 📊 Conclusion
The CNN outperforms the dense network because it preserves spatial relationships in image data and automatically extracts meaningful features such as edges and shapes.  
Dense networks flatten the image early, resulting in loss of spatial information and lower accuracy.

---

## 🛠 Technologies Used
- TensorFlow
- TensorFlow.js (CDN)
- Kaggle GPU (Tesla T4 ×2)
- HTML5 Canvas
- JavaScript

---

## ▶️ How to Run
1. Open the project folder in **VS Code**
2. Use **Live Server** to serve `index.html`
3. Draw a digit on the canvas (Task 2)
4. View accuracy comparison on the web page (Task 3)
5. Open **Browser Console (F12)** to view prediction logs

---

## 📌 Notes
- Model training is performed offline for efficiency
- Browser-based implementation uses only JavaScript (no backend)
- This practical demonstrates a real-world ML deployment workflow
