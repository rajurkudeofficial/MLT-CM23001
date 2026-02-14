# MNIST Digit Recognition using TensorFlow.js (CNN)

## 📌 Overview
This project implements **Handwritten Digit Recognition** using a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset**.  
The model is trained **offline on Kaggle using dual Tesla T4 GPUs** for faster and more accurate training, then converted to **TensorFlow.js** format and deployed in the browser using **JavaScript and CDN only** (no backend, no Node.js).

Users can draw digits (0–9) on an HTML canvas, and the pretrained model predicts the digit.  
The prediction result is displayed in the **browser console**.

---

## 🎯 Aim
To design and implement a handwritten digit recognition system using a CNN model trained on the MNIST dataset and deploy it for browser-based inference using TensorFlow.js.

---

## 🎯 Objectives
- To train a CNN model on the full MNIST dataset with high accuracy
- To utilize GPU acceleration for fast training
- To convert a trained TensorFlow model into TensorFlow.js format
- To perform real-time digit prediction in the browser using JavaScript
- To avoid backend technologies such as Node.js and Python during deployment

---

## 🧠 Approach Used

### 1. Model Training (Offline)
- Platform: **Kaggle Notebook**
- Accelerator: **2× Tesla T4 GPUs**
- Dataset: **MNIST (60,000 training images, 10,000 test images)**
- Model: **Convolutional Neural Network (CNN)**
- Result: High accuracy (~98–99%)

Training is done offline to avoid browser performance limitations.

---

### 2. Model Export & Conversion
- The trained CNN model is exported in **TensorFlow SavedModel** format
- The SavedModel is converted to **TensorFlow.js format** using `tensorflowjs_converter`
- Output files:
  - `model.json` (model architecture)
  - `.bin` file (trained weights)

---

### 3. Browser Deployment
- TensorFlow.js loaded using **CDN**
- Pretrained model loaded using `tf.loadGraphModel()`
- User draws digit on HTML canvas
- Canvas image is:
  - Converted to grayscale
  - Resized to 28×28
  - Normalized (0–1)
- Prediction is performed in the browser
- Output is shown in the **browser console**

---

## 🧩 Project Structure

PRACTICAL-2/
│
├── index.html # User interface (canvas, buttons)
├── model.js # Model loading, preprocessing, prediction
│
└── tfjs_model/
├── model.json
└── group1-shard1of1.bin


---

## 🛠 Technologies Used
- TensorFlow
- TensorFlow.js (CDN)
- Kaggle GPU (Tesla T4 × 2)
- HTML5 Canvas
- JavaScript

---

## 📘 Important Terminologies

### 🔹 What is GPU (Tesla T4 × 2)?
A **GPU (Graphics Processing Unit)** is specialized hardware for parallel computation.  
**Tesla T4 × 2** means two NVIDIA Tesla T4 GPUs were used simultaneously, enabling:
- Faster training
- Efficient handling of large datasets
- Reduced training time

---

### 🔹 What are Epochs?
An **epoch** represents one complete pass of the entire training dataset through the neural network.

Example:
- 5 epochs → the model sees the entire dataset 5 times

More epochs generally improve learning, but too many can cause overfitting.

---

### 🔹 What is Batch Size?
**Batch size** defines how many training samples are processed before updating the model weights.

Example:
- Batch size = 128
- Model updates weights after every 128 images

Using batches improves performance and stability during training.

---

### 🔹 What is Grayscale?
Grayscale images contain **only intensity values** (no RGB colors).  
MNIST digits are grayscale, which:
- Reduces computational complexity
- Matches the original dataset format
- Improves model compatibility

---

### 🔹 Why Image Size is 28×28?
MNIST dataset images are originally **28×28 pixels**.

Reasons:
- Maintains consistency with training data
- Reduces input dimensionality
- Faster processing
- Standard benchmark size for digit recognition

---

### 🔹 What is CNN?
A **Convolutional Neural Network (CNN)** is a deep learning model designed for image data.  
It automatically learns features such as:
- Edges
- Curves
- Shapes

CNNs are highly effective for handwritten digit recognition.

---

## ▶️ How to Run the Project

1. Open the project folder in **VS Code**
2. Install **Live Server** extension
3. Right-click `index.html` → **Open with Live Server**
4. Draw a digit on the canvas
5. Click **Predict**
6. Open **Browser Console (F12)** to view prediction result

---

## 📤 Output
Example console output:
Predicted Digit: 7

---

## ✅ Conclusion
This project demonstrates a complete real-world machine learning pipeline:
- GPU-accelerated training
- Model export and conversion
- Lightweight browser-based deployment

By separating training and inference, the system achieves high accuracy while remaining efficient and browser-compatible.

---

## 📌 Notes
- No Node.js is used
- No backend server is required
- Entire prediction runs on the client-side browser
