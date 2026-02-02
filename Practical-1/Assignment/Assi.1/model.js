// Global variables
let model;
let trainingData = { x: [], y: [] };
let lossHistory = [];
let dataChart, lossChart, predictionChart;

// Generate synthetic data: y = 2x + 1 + noise
function generateSyntheticData(numSamples = 100) {
    console.log("=".repeat(60));
    console.log("GENERATING SYNTHETIC DATA");
    console.log("=".repeat(60));
    
    const xData = [];
    const yData = [];
    
    for (let i = 0; i < numSamples; i++) {
        const x = Math.random() * 10; // Random x between 0 and 10
        const noise = (Math.random() - 0.5) * 2; // Random noise between -1 and 1
        const y = 2 * x + 1 + noise; // Linear relationship: y = 2x + 1 + noise
        
        xData.push(x);
        yData.push(y);
    }
    
    trainingData = { x: xData, y: yData };
    
    console.log(`Generated ${numSamples} samples`);
    console.log("True relationship: y = 2x + 1 + noise");
    console.log("\nFirst 5 samples:");
    for (let i = 0; i < 5; i++) {
        console.log(`  x: ${xData[i].toFixed(4)}, y: ${yData[i].toFixed(4)}`);
    }
    console.log("=".repeat(60));
    
    return { xData, yData };
}

// Create the model using tf.sequential() and tf.layers.dense()
function createModel() {
    console.log("\n" + "=".repeat(60));
    console.log("CREATING MODEL");
    console.log("=".repeat(60));
    
    // Create a sequential model
    const model = tf.sequential();
    
    // Add a dense layer with 1 unit (for linear regression)
    model.add(tf.layers.dense({
        units: 1,
        inputShape: [1],
        kernelInitializer: 'randomNormal',
        biasInitializer: 'zeros'
    }));
    
    // Compile the model
    model.compile({
        optimizer: tf.train.adam(0.1), // Learning rate = 0.1
        loss: 'meanSquaredError'
    });
    
    console.log("Model Architecture:");
    model.summary();
    console.log("\nOptimizer: Adam (learning rate: 0.1)");
    console.log("Loss Function: Mean Squared Error");
    console.log("=".repeat(60));
    
    return model;
}

// Train the model
async function trainModel() {
    console.log("\n" + "=".repeat(60));
    console.log("STARTING TRAINING");
    console.log("=".repeat(60));
    
    // Disable buttons during training
    document.getElementById('trainBtn').disabled = true;
    document.getElementById('predictBtn').disabled = true;
    
    // Show status
    const statusDiv = document.getElementById('status');
    statusDiv.style.display = 'block';
    statusDiv.innerHTML = '🔄 Training in progress...';
    
    // Generate data
    const { xData, yData } = generateSyntheticData(100);
    
    // Create model
    model = createModel();
    
    // Convert data to tensors
    const xs = tf.tensor2d(xData, [xData.length, 1]);
    const ys = tf.tensor2d(yData, [yData.length, 1]);
    
    // Reset loss history
    lossHistory = [];
    
    console.log("\nTraining for 100 epochs...");
    console.log("-".repeat(60));
    
    // Train the model
    const history = await model.fit(xs, ys, {
        epochs: 100,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                lossHistory.push(logs.loss);
                
                // Log every 10 epochs
                if ((epoch + 1) % 10 === 0) {
                    console.log(`Epoch ${epoch + 1}/100 - Loss: ${logs.loss.toFixed(6)}`);
                }
                
                // Update loss chart
                updateLossChart();
            }
        }
    });
    
    console.log("-".repeat(60));
    console.log(`Training completed!`);
    console.log(`Final Loss: ${lossHistory[lossHistory.length - 1].toFixed(6)}`);
    
    // Get model weights
    const weights = model.getWeights();
    const weightData = await weights[0].data();
    const biasData = await weights[1].data();
    
    console.log("\nLearned Parameters:");
    console.log(`  Weight (slope): ${weightData[0].toFixed(4)}`);
    console.log(`  Bias (intercept): ${biasData[0].toFixed(4)}`);
    console.log(`  Equation: y = ${weightData[0].toFixed(4)}x + ${biasData[0].toFixed(4)}`);
    console.log("  Expected: y ≈ 2x + 1");
    console.log("=".repeat(60));
    
    // Update UI
    document.getElementById('stats').style.display = 'grid';
    document.getElementById('samplesCount').textContent = xData.length;
    document.getElementById('finalLoss').textContent = lossHistory[lossHistory.length - 1].toFixed(4);
    document.getElementById('epochs').textContent = 100;
    
    // Calculate accuracy (R² score)
    const predictions = model.predict(xs);
    const predData = await predictions.data();
    const r2 = calculateR2(yData, Array.from(predData));
    document.getElementById('accuracy').textContent = (r2 * 100).toFixed(2) + '%';
    
    statusDiv.innerHTML = '✅ Training completed successfully!';
    
    // Update data chart
    updateDataChart(xData, yData);
    
    // Enable predict button
    document.getElementById('predictBtn').disabled = false;
    document.getElementById('trainBtn').disabled = false;
    
    // Cleanup tensors
    xs.dispose();
    ys.dispose();
}

// Make predictions on unseen data
async function makePredictions() {
    if (!model) {
        console.log("Please train the model first!");
        return;
    }
    
    console.log("\n" + "=".repeat(60));
    console.log("MAKING PREDICTIONS ON UNSEEN DATA");
    console.log("=".repeat(60));
    
    // Generate unseen test data
    const testX = [0.5, 2.5, 4.5, 6.5, 8.5, 10.5];
    const expectedY = testX.map(x => 2 * x + 1); // True values without noise
    
    console.log("\nTest Data:");
    console.log("-".repeat(60));
    
    // Make predictions
    const testXTensor = tf.tensor2d(testX, [testX.length, 1]);
    const predictions = model.predict(testXTensor);
    const predictedY = await predictions.data();
    
    console.log("X Value  | Expected Y | Predicted Y | Difference");
    console.log("-".repeat(60));
    
    const predictionResults = [];
    for (let i = 0; i < testX.length; i++) {
        const diff = Math.abs(predictedY[i] - expectedY[i]);
        console.log(
            `${testX[i].toString().padEnd(8)} | ` +
            `${expectedY[i].toFixed(4).padEnd(10)} | ` +
            `${predictedY[i].toFixed(4).padEnd(11)} | ` +
            `${diff.toFixed(4)}`
        );
        
        predictionResults.push({
            x: testX[i],
            expected: expectedY[i],
            predicted: predictedY[i]
        });
    }
    
    console.log("-".repeat(60));
    
    // Calculate mean absolute error
    const mae = predictionResults.reduce((sum, r) => 
        sum + Math.abs(r.predicted - r.expected), 0) / predictionResults.length;
    
    console.log(`\nMean Absolute Error: ${mae.toFixed(4)}`);
    console.log("=".repeat(60));
    
    // Update prediction chart
    updatePredictionChart(testX, expectedY, Array.from(predictedY));
    
    // Cleanup
    testXTensor.dispose();
    predictions.dispose();
}

// Calculate R² score
function calculateR2(actual, predicted) {
    const mean = actual.reduce((a, b) => a + b) / actual.length;
    const ssTotal = actual.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0);
    const ssResidual = actual.reduce((sum, val, i) => 
        sum + Math.pow(val - predicted[i], 2), 0);
    return 1 - (ssResidual / ssTotal);
}

// Update data chart
function updateDataChart(xData, yData) {
    const ctx = document.getElementById('dataChart').getContext('2d');
    
    if (dataChart) {
        dataChart.destroy();
    }
    
    // Get predictions for plotting the line
    const xRange = Array.from({ length: 50 }, (_, i) => i * 0.2);
    const xTensor = tf.tensor2d(xRange, [xRange.length, 1]);
    const predictions = model.predict(xTensor);
    const predData = predictions.dataSync();
    
    dataChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Training Data',
                    data: xData.map((x, i) => ({ x, y: yData[i] })),
                    backgroundColor: 'rgba(102, 126, 234, 0.6)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    pointRadius: 4
                },
                {
                    label: 'Fitted Line',
                    data: xRange.map((x, i) => ({ x, y: predData[i] })),
                    type: 'line',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    borderWidth: 3,
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                x: {
                    title: { display: true, text: 'X Values' }
                },
                y: {
                    title: { display: true, text: 'Y Values' }
                }
            }
        }
    });
    
    xTensor.dispose();
    predictions.dispose();
}

// Update loss chart
function updateLossChart() {
    const ctx = document.getElementById('lossChart').getContext('2d');
    
    if (lossChart) {
        lossChart.destroy();
    }
    
    lossChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: lossHistory.map((_, i) => i + 1),
            datasets: [{
                label: 'Training Loss',
                data: lossHistory,
                borderColor: 'rgba(118, 75, 162, 1)',
                backgroundColor: 'rgba(118, 75, 162, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                x: {
                    title: { display: true, text: 'Epoch' }
                },
                y: {
                    title: { display: true, text: 'Loss (MSE)' },
                    beginAtZero: false
                }
            }
        }
    });
}

// Update prediction chart
function updatePredictionChart(xVals, expected, predicted) {
    const ctx = document.getElementById('predictionChart').getContext('2d');
    
    if (predictionChart) {
        predictionChart.destroy();
    }
    
    predictionChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Expected Values',
                    data: xVals.map((x, i) => ({ x, y: expected[i] })),
                    backgroundColor: 'rgba(75, 192, 192, 0.6)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    pointRadius: 8,
                    pointStyle: 'circle'
                },
                {
                    label: 'Predicted Values',
                    data: xVals.map((x, i) => ({ x, y: predicted[i] })),
                    backgroundColor: 'rgba(255, 159, 64, 0.6)',
                    borderColor: 'rgba(255, 159, 64, 1)',
                    pointRadius: 8,
                    pointStyle: 'triangle'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                x: {
                    title: { display: true, text: 'X Values' }
                },
                y: {
                    title: { display: true, text: 'Y Values' }
                }
            }
        }
    });
}

// Reset model
function resetModel() {
    console.log("\n" + "=".repeat(60));
    console.log("RESETTING MODEL");
    console.log("=".repeat(60));
    
    if (model) {
        model.dispose();
        model = null;
    }
    
    trainingData = { x: [], y: [] };
    lossHistory = [];
    
    if (dataChart) dataChart.destroy();
    if (lossChart) lossChart.destroy();
    if (predictionChart) predictionChart.destroy();
    
    document.getElementById('stats').style.display = 'none';
    document.getElementById('status').style.display = 'none';
    document.getElementById('predictBtn').disabled = true;
    document.getElementById('trainBtn').disabled = false;
    
    console.log("Model and data reset successfully!");
    console.log("=".repeat(60));
}

// Initialize
console.log("%c Linear Regression - Assignment 1 ", 
    "background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; font-size: 20px; padding: 10px; font-weight: bold;");
console.log("TensorFlow.js version:", tf.version.tfjs);
console.log("Ready! Click 'Train Model' to begin.");
