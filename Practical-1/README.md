# Practical 1: Linear Regression with Synthetic Data (TensorFlow.js)

## Aim
To implement a simple Linear Regression model using **TensorFlow.js** with synthetic data and analyze the relationship between input and output values using graphical visualization.

---

## Software & Tools Used
- Language: JavaScript
- Library: TensorFlow.js (via CDN)
- IDE: VS Code
- Platform: Web Browser (Chrome / Edge)

---

## Description
In this practical, a linear regression model is trained on synthetic data using
`tf.sequential()` and `tf.layers.dense()`.  
The trained model predicts output values and visualizes **actual vs predicted**
data on a graph.

The complete execution is done **in the browser** without using Node.js.

---

## Files Included
- `index.html` – Creates the webpage and visualization layout
- `model.js` – Handles model creation, training, prediction, and console output
- `output.png` – Screenshot of final output
- `README.md` – Practical documentation

---

## Output
- Graph showing **Actual Data vs Predicted Line**
- Model Equation (y = mx + c)
- Mean Squared Error (MSE)
- Console output for training and prediction

---

## Conclusion
The linear regression model successfully learned the linear relationship from
synthetic data. The predicted values closely match the actual values, proving
correct implementation of TensorFlow.js in a browser environment.
