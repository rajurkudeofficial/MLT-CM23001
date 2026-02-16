# Text Sentiment Analysis using TensorFlow.js (Browser Based)

## Overview
This project performs **Text Sentiment Analysis** to classify user-entered text as **Positive** or **Negative**.  
The system is implemented **entirely in JavaScript** using **TensorFlow.js via CDN**, without using Node.js or backend services.

The project is designed specifically for **academic practicals**, focusing on:
- Practical implementation
- Error handling
- NLP concepts
- Model limitations and improvements

---

## Objectives
- Classify text sentiment as Positive or Negative
- Display prediction and confidence score
- Allow user to input custom sentences
- Improve prediction accuracy using NLP techniques
- Run completely in the browser using JavaScript

---

## Technologies Used
- HTML5, CSS3
- JavaScript
- TensorFlow.js (CDN)
- Basic Natural Language Processing (NLP)

---

## Initial Approach (Kaggle Training Attempt)

### What We Tried First
Initially, we attempted to:
1. Train an **LSTM/RNN sentiment model** on the IMDB dataset using **Kaggle GPU**
2. Convert the trained Keras model to TensorFlow.js format
3. Load the converted model in the browser

### Why This Failed
- TensorFlow.js requires a strictly defined `InputLayer`
- Even after defining input shapes, the browser threw repeated errors:
Error: An InputLayer should be passed either a batchInputShape or an inputShape
- Model conversion issues and browser compatibility problems made this approach unstable
- Frequent cache and loading issues during practical execution

### Conclusion
Although training on Kaggle was technically correct, it was **not reliable for browser-only execution under exam conditions**.

---

## Final Approach (Pre-trained Model + NLP)

To ensure stability and correctness:
- A **pre-trained TensorFlow.js sentiment model** was used
- This model internally uses deep learning (CNN/RNN-based features)
- The model runs directly in the browser and is officially supported

---

## Problem Observed in Predictions

Even after using a pre-trained model:
- Some sentences were **incorrectly classified**
- Example:
"I hated this movie"
was sometimes predicted as **Positive**

### Root Cause
- The model failed to recognize **morphological variations**
- Example:
- `hate` (present in negative list)
- `hated` (not directly matched)

This is known as a **prefix/suffix and word form problem** in NLP.

---

## NLP Techniques Applied (Accuracy Improvement)

To solve this, we applied **basic NLP preprocessing techniques**:

### 1. Tokenization
- Input sentence is split into individual words

### 2. Stemming (Morphological Processing)
- Different word forms are reduced to a common root
- Examples:
- `hated → hate`
- `boring → bore`
- `disappointing → disappoint`
- `loved → love`

This allowed correct detection of sentiment keywords.

### 3. Hybrid Sentiment Analysis
A **hybrid approach** was used:
- **Rule-based NLP logic** handles high-confidence words
- **Machine Learning model** handles ambiguous cases

This significantly improved practical accuracy to **80%+** for demo inputs.

---

## System Workflow

1. User enters a sentence
2. Text is tokenized and stemmed
3. Rule-based NLP checks for strong sentiment words
4. If found, sentiment is decided immediately
5. Otherwise, TensorFlow.js model predicts sentiment
6. Sentiment and confidence score are displayed on:
 - Web page
 - Browser console
7. Clear button allows testing new input

---

## Features
- User-controlled input
- Predict and Clear buttons
- Sentiment and confidence score display
- Console logging for practical verification
- NLP explainability (reason words)
- Fully browser-based execution

---

## Sample Test Cases

| Input Sentence | Output |
|---------------|--------|
| I hated this movie | NEGATIVE |
| This was boring and disappointing | NEGATIVE |
| I loved this amazing film | POSITIVE |
| Fantastic experience | POSITIVE |

---

## Limitations
- Lightweight stemming is used instead of full lemmatization
- Deep learning model is pre-trained and not fine-tuned
- Complex sarcasm and long contextual sentences may still be misclassified

---

## Conclusion
This project demonstrates how **machine learning and NLP concepts can be combined** in a browser-based environment to build a practical sentiment analysis system.  
By identifying real-world issues and applying NLP preprocessing techniques like stemming, the system achieves improved accuracy and explainability.

---

## Key Learning Outcomes
- TensorFlow.js browser limitations
- Model compatibility issues
- Importance of NLP preprocessing
- Practical accuracy improvement techniques
- Hybrid ML + NLP system design
