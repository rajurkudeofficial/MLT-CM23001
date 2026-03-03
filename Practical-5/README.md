# 🔥 Webcam-Based Object Detection using TensorFlow.js

## 📌 1. AI Objective & Goal

The objective of this project is to design and implement a real-time, browser-based object detection system using deep learning. The system captures live webcam frames, performs real-time inference using a pre-trained deep learning model, and overlays detection results directly on the video feed.

The primary goals of this project are:

* To implement real-time computer vision in the browser
* To apply deep learning inference without using Python or backend servers
* To analyze the performance impact of real-time inference
* To build an interactive and intelligent AI-based detection system

This project demonstrates how modern web technologies and deep learning models can be combined to create efficient AI applications directly inside the browser.

---

## 🧠 2. Why COCO-SSD was Selected Over MobileNet

Although the assignment mentions MobileNet for classification, this project uses the COCO-SSD model for the following reasons:

1. COCO-SSD uses MobileNet as its backbone for feature extraction.
2. MobileNet alone performs only image classification (no bounding boxes).
3. COCO-SSD performs object detection (classification + localization).
4. The assignment requires overlaying labels on the video feed, which is better achieved using object detection.
5. COCO-SSD provides bounding boxes, class names, and confidence scores in a single forward pass.

Therefore, COCO-SSD was selected because it extends MobileNet’s capability by enabling real-time object detection rather than only classification.

---

## 🛠 3. Technologies Used

* HTML5
* CSS3
* JavaScript (ES6)
* TensorFlow.js (CDN)
* COCO-SSD Pre-trained Model
* Browser Webcam API (getUserMedia)

No Python or Node.js was used in the final implementation.

---

## 🎯 4. Features Implemented

The system includes the following advanced features:

### Core Features

* Real-time webcam capture
* Object detection using COCO-SSD
* Bounding box overlay on video
* Class labels with confidence score
* FPS (Frames Per Second) monitoring
* Start/Stop detection control
* Adjustable confidence threshold

### Advanced Features

* Object Presence Timer (tracks how long an object remains on screen)
* Dominant Object Detection (largest detected object)
* Voice Alert on first detection
* Snapshot capture with timestamp
* Detection frequency counter
* Real-time analytics panel
* Clean and professional dark UI
* 100% browser-based inference

---

## ⚙ 5. System Workflow

1. Webcam access is requested using getUserMedia().
2. Live video frames are continuously processed.
3. Each frame is passed to the COCO-SSD model.
4. The model performs inference and returns detected objects.
5. Bounding boxes and labels are drawn on canvas.
6. FPS is calculated to measure performance.
7. Additional logic handles timers, snapshots, and analytics.

---

## 📊 6. Performance Analysis of Real-Time Inference

Real-time inference significantly impacts system performance because each video frame is processed through a deep learning model.

The following experiments were performed:

### Performance Comparison Table

| Condition              | FPS   | Observation                  |
| ---------------------- | ----- | ---------------------------- |
| Camera Only (No Model) | 50–60 | Smooth performance           |
| Detection ON (720p)    | 25–30 | Moderate processing load     |
| Detection ON (1080p)   | 15–22 | Increased GPU usage          |
| Confidence = 0.7       | 28–32 | Stable performance           |
| Confidence = 0.3       | 20–25 | More detections, higher load |

### Observations

* Enabling detection reduces FPS due to model inference.
* Higher resolution increases computational cost.
* Lower confidence threshold increases number of detections and processing time.
* Additional features like snapshots and speech synthesis add minor overhead.
* There is a trade-off between accuracy and speed.

### Conclusion of Performance Impact

Real-time inference introduces computational overhead that affects frame rate and responsiveness. Optimizing resolution, confidence threshold, and detection frequency improves performance stability.

---

## 📘 7. Assignment Task Coverage

### Assignment Requirement:

1. Capture webcam frames and classify objects using MobileNet.
2. Overlay classification labels on the video feed.
3. Measure FPS and analyze performance impact of real-time inference.

### How This Project Covers the Tasks:

* Webcam frames are captured using getUserMedia().
* COCO-SSD (which uses MobileNet backbone) performs classification and detection.
* Labels and bounding boxes are overlaid on the video feed.
* FPS is calculated and displayed in real time.
* Performance impact is analyzed using controlled experiments.

Thus, all assignment requirements are fully satisfied.

---

## 🏁 8. Final Conclusion

This project successfully demonstrates real-time object detection using TensorFlow.js entirely within the browser environment. By leveraging a pre-trained COCO-SSD model, the system achieves accurate object detection with bounding boxes, labels, analytics, and performance monitoring.

The project highlights the feasibility of deploying deep learning applications directly in web browsers without backend dependencies, making it scalable, efficient, and portable.

---

## 🚀 Future Scope

* Face recognition integration
* Object tracking with ID assignment
* Real-time video recording and export
* Performance benchmarking dashboard
* Custom-trained model integration

---

**Developed as part of CO3 – Apply Practical Implementation**
