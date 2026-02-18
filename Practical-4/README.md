# Image Classification using Pre-trained Models (TensorFlow.js)

---

## 1. AIM

To perform image classification in the browser using pre-trained deep learning models with TensorFlow.js, without using Python or Node.js, and to compare predictions of different CNN architectures.

---

## 2. OBJECTIVES

After completing this practical and assignments, the student will be able to:

- Understand browser-based machine learning using TensorFlow.js
- Load and use pre-trained CNN models without training
- Perform image classification directly in the web browser
- Display prediction results on UI and browser console
- Analyze prediction confidence and accuracy
- Compare different CNN architectures
- Understand browser limitations such as CORS policy

---

## 3. TOOLS & TECHNOLOGIES USED

- HTML5
- JavaScript
- TensorFlow.js (CDN)
- Pre-trained CNN models
- Web Browser (Chrome / Edge)

---

## 4. CONSTRAINTS / RULES

- Python is not allowed
- Node.js is not allowed
- Backend / server is not allowed
- Only browser-based execution is permitted
- Only CDN-based models are used

---

## 5. MAIN PRACTICAL  
### Image Classification using Pre-trained MobileNet

### 5.1 Approach

1. TensorFlow.js CDN is loaded in the HTML file.
2. A pre-trained MobileNet model is loaded in the browser.
3. User uploads an image using a file input.
4. The uploaded image is displayed using an HTML image tag.
5. The image is passed to the MobileNet model using the `classify()` method.
6. The model generates top-3 predictions with confidence scores.
7. The prediction results are displayed:
   - In the browser console
   - On the web UI

---

### 5.2 Algorithm Used

- Convolutional Neural Network (CNN)
- MobileNet architecture using Depthwise Separable Convolutions

---

### 5.3 Result

- MobileNet successfully classifies common objects.
- The execution is fast and suitable for browser environments.

---

## 6. ASSIGNMENT TASKS

---

### Task 1  
**Load MobileNet and classify a static image; display top-3 predictions**

#### Approach:
- A static image is uploaded by the user.
- MobileNet predicts the image class.
- Top-3 predictions with confidence scores are displayed.

**Status:**  
✔️ Completed in the main practical.

---

### Task 2  
**Test classification accuracy on at least 5 different images**

#### Approach:
1. Five different images from different categories are selected.
2. Each image is classified using the MobileNet model.
3. Predictions are manually compared with actual objects in the images.
4. Accuracy is observed based on correct predictions.

**Status:**  
✔️ Completed in the main practical.

---

### Task 3  
**Compare MobileNet predictions with another pre-trained model**

---

## 7. INITIAL PLAN FOR TASK-3 (ResNet)

### Original Idea:
- Use ResNet as a second pre-trained model.
- Load ResNet from TensorFlow Hub using TensorFlow.js.
- Compare predictions of MobileNet and ResNet on the same image.

---

## 8. ERROR ENCOUNTERED WITH RESNET

### Error Observed:

Access to fetch at 'https://tfhub.dev/...' has been blocked by CORS policy  
Failed to load resource: net::ERR_FAILED  
TypeError: Failed to fetch  

---

### Reason for Error:

- ResNet models hosted on TensorFlow Hub / Kaggle do not allow cross-origin access.
- Browser security (CORS policy) blocks loading these models directly.
- Therefore, ResNet cannot be loaded using client-side JavaScript.

---

### Learning Outcome:

TFHub-based ResNet models are not browser-friendly and cannot be directly loaded using TensorFlow.js CDN.

---

## 9. FINAL DECISION & SOLUTION

Since:
- Python and Node.js were not allowed
- Backend hosting was not permitted
- ResNet could not be loaded due to CORS issues

**ResNet was replaced with MobileNet V2.**

---

## 10. WHY MOBILENET V2?

- Fully supported in TensorFlow.js
- Browser-compatible
- No CORS issues
- Deeper and more optimized than MobileNet V1
- Suitable for performance comparison

---

## 11. FINAL TASK-3 IMPLEMENTATION  
### MobileNet V1 vs MobileNet V2 Comparison

### Approach:

1. MobileNet V1 and MobileNet V2 models are loaded simultaneously.
2. A single image is uploaded by the user.
3. The same image is classified by both models.
4. Predictions and confidence scores are displayed on the UI and console.
5. A bar graph is drawn using HTML Canvas to compare confidence values.

---

## 12. COMPARISON TABLE

| Parameter        | MobileNet V1 | MobileNet V2 |
|------------------|--------------|--------------|
| Model Depth      | Shallow      | Deeper       |
| Speed            | Faster       | Moderate     |
| Accuracy         | Good         | Higher       |
| Browser Support  | Excellent    | Excellent    |

---

## 13. FINAL CONCLUSION

In this experiment, image classification was successfully performed using pre-trained CNN models directly in the browser with TensorFlow.js. MobileNet was used in the main practical to classify images and test accuracy. For comparison, MobileNet V1 and MobileNet V2 were implemented due to browser compatibility constraints. The experiment demonstrates efficient client-side machine learning without the use of Python or backend servers.

---

## 14. KEY LEARNINGS

- Browser-based machine learning is possible using TensorFlow.js
- Not all deep learning models are browser-compatible
- Browser security policies such as CORS affect model deployment
- Model selection depends on environment constraints
- MobileNet family models are ideal for web-based applications

---

## 15. VIVA-READY QUESTIONS

**Q1. Why was ResNet not used?**  
ResNet models are blocked by browser CORS policy and cannot be loaded directly.

**Q2. Why MobileNet V2 was selected?**  
MobileNet V2 is optimized, deeper, and fully supported in TensorFlow.js.

**Q3. Was training performed in this experiment?**  
No, only pre-trained models were used for inference.

---

## STATUS: COMPLETED
