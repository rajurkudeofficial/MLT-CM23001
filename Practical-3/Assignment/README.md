# Practical–3: Text Sentiment Analysis using TensorFlow.js

## Aim
To design and implement a **Text Sentiment Analysis system** that classifies text as **Positive or Negative**, displays **confidence scores**, and compares a **self-trained model** with an **old (pre-trained / rule-based) model** using **JavaScript and TensorFlow.js (CDN only)**.

---

## Assignment Idea & Approach (Student Proposed)

This practical follows a **custom approach proposed by the student**, consisting of three major tasks:

### Task-1  
Train a sentiment classifier using a **custom dataset** containing positive and negative sentences.

### Task-2  
Test the trained model on **user-entered custom sentences** and interpret the **confidence score**.

### Task-3  
Compare the performance of:
- A **self-trained deep learning model**
- An **old pre-trained (rule-based) sentiment model**

The comparison is done **live in the browser** using the same input sentence.

---

## Tools & Technologies Used
- Python (for dataset handling and training)
- TensorFlow & Keras (model training)
- TensorFlow.js (CDN based, no Node.js)
- JavaScript (logic & inference)
- HTML & CSS (UI)
- Google Colab (training environment)
- VS Code + Live Server (execution)

---

## Dataset Description (Task-1)

- A **custom dataset** with approximately **10,000 sentences**
- Columns:
  - `sentence`
  - `type` (positive / negative)
- Neutral sentences were **removed** to focus on **binary classification**
- Labels used:
  - Positive → 1
  - Negative → 0

### Why Neutral Data Was Removed?
The assignment focuses on **binary sentiment classification**, so removing neutral data reduces ambiguity and improves learning clarity.

---

## Model Training (Task-1)

- The model was trained in **Google Colab**
- Model architecture:
  - Embedding Layer
  - Global Average Pooling
  - Dense Hidden Layer
  - Sigmoid Output Layer
- After training:
  - Model was exported in **SavedModel format**
  - Converted into **TensorFlow.js GraphModel**
  - Tokenizer saved separately as `tokenizer.json`

---

## Browser Deployment using TensorFlow.js

- The trained model is loaded in the browser using:
  ```js
  tf.loadGraphModel()
TensorFlow.js is included using CDN

No Node.js or backend is used (as per assignment rules)

Input text is:

Tokenized using the same tokenizer

Converted to tensors

Explicitly cast to float32 to match model requirements
TensorFlow.js is included using CDN

No Node.js or backend is used (as per assignment rules)

Input text is:

Tokenized using the same tokenizer

Converted to tensors

Explicitly cast to float32 to match model requirements
Folder Structure
Practical-3/
└── Assignment/
    ├── index.html
    ├── model.js
    └── Training/
        ├── model.json
        ├── group1-shard1of1.bin
        └── tokenizer.json
Observations & Limitations

Some strong words (e.g., “hated”) may occasionally be misclassified

This happens due to:

Dataset imbalance

Limited linguistic diversity

Such behavior is common in NLP models trained on synthetic data

This highlights the importance of data quality and preprocessing.

Conclusion

Successfully implemented a complete sentiment analysis pipeline

Covered:

Dataset preparation

Model training

TFJS conversion

Browser deployment

Live comparison of models

The practical demonstrates strong understanding of:

NLP concepts

Deep learning

Real-world deployment challenges

Viva-Ready Key Points

TensorFlow.js enables ML models to run directly in the browser

Confidence score represents prediction probability

Binary classification simplifies sentiment analysis

Rule-based models are interpretable but less flexible

Deep learning models generalize better with large datasets

Developed By

Name: Raj Urkude
Subject: Machine Learning Techniques
Practical No: 3