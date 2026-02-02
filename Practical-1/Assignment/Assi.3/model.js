// Global variables
let model;
let trainingData = { x: [], y: [] };
let testData = { x: [], expected: [], predicted: [] };
let lossHistory = [];
let trainingChart, lossChart, comparisonChart, errorChart;

// Generate synthetic data for training
function generateTrainingData(numSamples = 100) {
    console.log("=".repeat(70));
    console.log("GENERATING TRAINING DATA");
    console.log("=".repeat(70));
    
    const xData = [];
    const yData = [];
    
    for (let i = 0; i < numSamples; i++) {
        const x = Math.random() * 10;
        const noise = (Math.random() - 0.5) * 2;
        const y = 2 * x + 1 + noise;
        
        xData.push(x);
        yData.push(y);
    }
    
    trainingData = { x: xData, y: yData };
    
    console.log(`Generated ${numSamples} training samples`);
    console.log("True relationship: y = 2x + 1 + noise");
    console.log("\nSample training data (first 5):");
    for (let i = 0; i < 5; i++) {
        console.log(`  x: ${xData[i].toFixed(4)}, y: ${yData[i].toFixed(4)}`);
    }
    console.log("=".repeat(70));
    
    return { xData, yData };
}

// Generate unseen test data
function generateTestData(numSamples = 20) {
    console.log("\n" + "=".repeat(70));
    console.log("GENERATING UNSEEN TEST DATA");
    console.log("=".repeat(70));
    
    const xData = [];
    const expectedY = [];
    
    // Generate test points across the range, including some outside training range
    for (let i = 0; i < numSamples; i++) {
        const x = (i * 12) / (numSamples - 1); // Range from 0 to 12
        const y = 2 * x + 1; // True relationship without noise
        
        xData.push(x);
        expectedY.push(y);
    }
    
    console.log(`Generated ${numSamples} test samples`);
    console.log("Range: 0 to 12 (includes extrapolation beyond training range)");
    console.log("\nSample test data (first 5):");
    for (let i = 0; i < 5; i++) {
        console.log(`  x: ${xData[i].toFixed(4)}, expected y: ${expectedY[i].toFixed(4)}`);
    }
    console.log("=".repeat(70));
    
    return { xData, expectedY };
}

// Create the model
function createModel() {
    console.log("\n" + "=".repeat(70));
    console.log("CREATING MODEL");
    console.log("=".repeat(70));
    
    const model = tf.sequential();
    
    model.add(tf.layers.dense({
        units: 1,
        inputShape: [1],
        kernelInitializer: 'randomNormal',
        biasInitializer: 'zeros'
    }));
    
    model.compile({
        optimizer: tf.train.adam(0.1),
        loss: 'meanSquaredError'
    });
    
    console.log("Model Architecture:");
    model.summary();
    console.log("\nOptimizer: Adam (learning rate: 0.1)");
    console.log("Loss Function: Mean Squared Error");
    console.log("=".repeat(70));
    
    return model;
}

// Train the model
async function trainModel() {
    console.log("\n" + "=".repeat(70));
    console.log("TRAINING MODEL");
    console.log("=".repeat(70));
    
    // Disable buttons
    document.getElementById('trainBtn').disabled = true;
    document.getElementById('predictBtn').disabled = true;
    
    // Show status
    const statusDiv = document.getElementById('status');
    statusDiv.style.display = 'block';
    statusDiv.innerHTML = '🔄 Training model on synthetic data...';
    
    // Generate training data
    const { xData, yData } = generateTrainingData(100);
    
    // Create model
    model = createModel();
    
    // Convert to tensors
    const xs = tf.tensor2d(xData, [xData.length, 1]);
    const ys = tf.tensor2d(yData, [yData.length, 1]);
    
    // Reset loss history
    lossHistory = [];
    
    console.log("\nTraining for 100 epochs...");
    console.log("-".repeat(70));
    
    // Train
    await model.fit(xs, ys, {
        epochs: 100,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                lossHistory.push(logs.loss);
                
                if ((epoch + 1) % 10 === 0) {
                    console.log(`Epoch ${epoch + 1}/100 - Loss: ${logs.loss.toFixed(6)}`);
                }
                
                updateLossChart();
            }
        }
    });
    
    console.log("-".repeat(70));
    console.log("Training completed!");
    
    // Get learned parameters
    const weights = model.getWeights();
    const weightData = await weights[0].data();
    const biasData = await weights[1].data();
    
    console.log("\nLearned Parameters:");
    console.log(`  Weight (slope): ${weightData[0].toFixed(4)}`);
    console.log(`  Bias (intercept): ${biasData[0].toFixed(4)}`);
    console.log(`  Learned Equation: y = ${weightData[0].toFixed(4)}x + ${biasData[0].toFixed(4)}`);
    console.log(`  True Equation: y = 2x + 1`);
    console.log(`  Weight Error: ${Math.abs(weightData[0] - 2).toFixed(4)}`);
    console.log(`  Bias Error: ${Math.abs(biasData[0] - 1).toFixed(4)}`);
    console.log("=".repeat(70));
    
    // Update UI
    document.getElementById('stats').style.display = 'grid';
    document.getElementById('trainingSamples').textContent = xData.length;
    document.getElementById('modelLoss').textContent = lossHistory[lossHistory.length - 1].toFixed(4);
    
    statusDiv.innerHTML = '✅ Model trained successfully! Ready for predictions.';
    
    // Update charts
    updateTrainingChart(xData, yData);
    
    // Enable predict button
    document.getElementById('predictBtn').disabled = false;
    document.getElementById('trainBtn').disabled = false;
    
    xs.dispose();
    ys.dispose();
}

// Predict on unseen test data
async function predictUnseen() {
    if (!model) {
        console.log("Please train the model first!");
        return;
    }
    
    console.log("\n" + "=".repeat(70));
    console.log("PREDICTING ON UNSEEN TEST DATA");
    console.log("=".repeat(70));
    
    // Update status
    const statusDiv = document.getElementById('status');
    statusDiv.innerHTML = '🔄 Making predictions on unseen data...';
    
    // Generate test data
    const { xData, expectedY } = generateTestData(20);
    
    // Make predictions
    const xTensor = tf.tensor2d(xData, [xData.length, 1]);
    const predictions = model.predict(xTensor);
    const predictedY = await predictions.data();
    
    // Store test results
    testData = {
        x: xData,
        expected: expectedY,
        predicted: Array.from(predictedY)
    };
    
    console.log("\nPREDICTION RESULTS");
    console.log("=".repeat(70));
    console.log("Test # | X Input | Expected Y | Predicted Y | Abs Error | Error %");
    console.log("-".repeat(70));
    
    let totalError = 0;
    const errors = [];
    
    for (let i = 0; i < xData.length; i++) {
        const absError = Math.abs(predictedY[i] - expectedY[i]);
        const errorPercent = (absError / Math.abs(expectedY[i])) * 100;
        
        totalError += absError;
        errors.push(absError);
        
        console.log(
            `${(i + 1).toString().padStart(6)} | ` +
            `${xData[i].toFixed(3).padStart(7)} | ` +
            `${expectedY[i].toFixed(4).padStart(10)} | ` +
            `${predictedY[i].toFixed(4).padStart(11)} | ` +
            `${absError.toFixed(4).padStart(9)} | ` +
            `${errorPercent.toFixed(2)}%`
        );
    }
    
    const mae = totalError / xData.length;
    const maxError = Math.max(...errors);
    const minError = Math.min(...errors);
    
    console.log("-".repeat(70));
    console.log(`\nPERFORMANCE METRICS:`);
    console.log(`  Mean Absolute Error (MAE): ${mae.toFixed(4)}`);
    console.log(`  Maximum Error: ${maxError.toFixed(4)}`);
    console.log(`  Minimum Error: ${minError.toFixed(4)}`);
    console.log(`  Average Error %: ${((mae / (expectedY.reduce((a, b) => a + b) / expectedY.length)) * 100).toFixed(2)}%`);
    console.log("=".repeat(70));
    
    // Update UI
    document.getElementById('testSamples').textContent = xData.length;
    document.getElementById('predictionMAE').textContent = mae.toFixed(4);
    
    // Update table
    updatePredictionsTable(xData, expectedY, predictedY, errors);
    
    // Update charts
    updateComparisonChart(xData, expectedY, predictedY);
    updateErrorChart(errors);
    
    // Show analysis
    showAnalysis(mae, maxError, minError, errors);
    
    // Show custom input section
    document.getElementById('customInput').style.display = 'block';
    
    statusDiv.innerHTML = '✅ Predictions completed! Scroll down to see detailed results.';
    
    xTensor.dispose();
    predictions.dispose();
}

// Update predictions table
function updatePredictionsTable(xData, expectedY, predictedY, errors) {
    const tbody = document.getElementById('predictionsBody');
    tbody.innerHTML = '';
    
    for (let i = 0; i < xData.length; i++) {
        const row = tbody.insertRow();
        const absError = errors[i];
        const errorPercent = (absError / Math.abs(expectedY[i])) * 100;
        
        let errorClass = 'error-low';
        if (errorPercent > 5) errorClass = 'error-medium';
        if (errorPercent > 10) errorClass = 'error-high';
        
        row.innerHTML = `
            <td>${i + 1}</td>
            <td>${xData[i].toFixed(3)}</td>
            <td>${expectedY[i].toFixed(4)}</td>
            <td>${predictedY[i].toFixed(4)}</td>
            <td class="error-cell ${errorClass}">${absError.toFixed(4)}</td>
            <td class="error-cell ${errorClass}">${errorPercent.toFixed(2)}%</td>
        `;
    }
    
    document.getElementById('predictionsTableContainer').style.display = 'block';
}

// Update training chart
function updateTrainingChart(xData, yData) {
    const ctx = document.getElementById('trainingChart').getContext('2d');
    
    if (trainingChart) {
        trainingChart.destroy();
    }
    
    // Generate line points
    const xRange = Array.from({ length: 50 }, (_, i) => i * 0.25);
    const xTensor = tf.tensor2d(xRange, [xRange.length, 1]);
    const predictions = model.predict(xTensor);
    const predData = predictions.dataSync();
    
    trainingChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Training Data',
                    data: xData.map((x, i) => ({ x, y: yData[i] })),
                    backgroundColor: 'rgba(79, 172, 254, 0.6)',
                    borderColor: 'rgba(79, 172, 254, 1)',
                    pointRadius: 4
                },
                {
                    label: 'Model Prediction',
                    data: xRange.map((x, i) => ({ x, y: predData[i] })),
                    type: 'line',
                    borderColor: 'rgba(0, 242, 254, 1)',
                    backgroundColor: 'rgba(0, 242, 254, 0.1)',
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
                x: { title: { display: true, text: 'X Values' } },
                y: { title: { display: true, text: 'Y Values' } }
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
                borderColor: 'rgba(0, 242, 254, 1)',
                backgroundColor: 'rgba(0, 242, 254, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                x: { title: { display: true, text: 'Epoch' } },
                y: { title: { display: true, text: 'Loss (MSE)' } }
            }
        }
    });
}

// Update comparison chart
function updateComparisonChart(xData, expectedY, predictedY) {
    const ctx = document.getElementById('comparisonChart').getContext('2d');
    
    if (comparisonChart) {
        comparisonChart.destroy();
    }
    
    comparisonChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Expected Values',
                    data: xData.map((x, i) => ({ x, y: expectedY[i] })),
                    backgroundColor: 'rgba(76, 175, 80, 0.6)',
                    borderColor: 'rgba(76, 175, 80, 1)',
                    pointRadius: 6
                },
                {
                    label: 'Predicted Values',
                    data: xData.map((x, i) => ({ x, y: predictedY[i] })),
                    backgroundColor: 'rgba(255, 152, 0, 0.6)',
                    borderColor: 'rgba(255, 152, 0, 1)',
                    pointRadius: 6,
                    pointStyle: 'triangle'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                x: { title: { display: true, text: 'X Input' } },
                y: { title: { display: true, text: 'Y Value' } }
            }
        }
    });
    
    document.getElementById('comparisonCharts').style.display = 'grid';
}

// Update error chart
function updateErrorChart(errors) {
    const ctx = document.getElementById('errorChart').getContext('2d');
    
    if (errorChart) {
        errorChart.destroy();
    }
    
    errorChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: errors.map((_, i) => `Test ${i + 1}`),
            datasets: [{
                label: 'Absolute Error',
                data: errors,
                backgroundColor: errors.map(e => {
                    if (e < 0.5) return 'rgba(76, 175, 80, 0.6)';
                    if (e < 1.0) return 'rgba(255, 152, 0, 0.6)';
                    return 'rgba(244, 67, 54, 0.6)';
                }),
                borderColor: errors.map(e => {
                    if (e < 0.5) return 'rgba(76, 175, 80, 1)';
                    if (e < 1.0) return 'rgba(255, 152, 0, 1)';
                    return 'rgba(244, 67, 54, 1)';
                }),
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: { 
                    title: { display: true, text: 'Absolute Error' },
                    beginAtZero: true
                }
            }
        }
    });
}

// Show analysis
function showAnalysis(mae, maxError, minError, errors) {
    const analysisDiv = document.getElementById('comparisonAnalysis');
    const contentDiv = document.getElementById('analysisContent');
    
    const avgExpected = testData.expected.reduce((a, b) => a + b) / testData.expected.length;
    const avgErrorPercent = (mae / avgExpected) * 100;
    
    const goodPredictions = errors.filter(e => e < 0.5).length;
    const fairPredictions = errors.filter(e => e >= 0.5 && e < 1.0).length;
    const poorPredictions = errors.filter(e => e >= 1.0).length;
    
    contentDiv.innerHTML = `
        <p><strong>Overall Accuracy:</strong> The model achieved a Mean Absolute Error of ${mae.toFixed(4)}, which is ${avgErrorPercent.toFixed(2)}% of the average expected value.</p>
        
        <p><strong>Error Range:</strong> Errors ranged from ${minError.toFixed(4)} (best) to ${maxError.toFixed(4)} (worst).</p>
        
        <p><strong>Prediction Quality Distribution:</strong></p>
        <ul>
            <li>🟢 Excellent (error < 0.5): ${goodPredictions} predictions (${((goodPredictions/errors.length)*100).toFixed(1)}%)</li>
            <li>🟡 Fair (error 0.5-1.0): ${fairPredictions} predictions (${((fairPredictions/errors.length)*100).toFixed(1)}%)</li>
            <li>🔴 Needs Improvement (error > 1.0): ${poorPredictions} predictions (${((poorPredictions/errors.length)*100).toFixed(1)}%)</li>
        </ul>
        
        <p><strong>Interpolation vs Extrapolation:</strong> The model was trained on data from 0-10. Test points beyond this range (10-12) show ${poorPredictions > 0 ? 'some' : 'minimal'} extrapolation error.</p>
        
        <p><strong>Conclusion:</strong> ${avgErrorPercent < 5 ? 'The model performs excellently with high accuracy across the test range.' : avgErrorPercent < 10 ? 'The model shows good predictive performance with acceptable error rates.' : 'The model predictions are reasonable but could benefit from more training data or tuning.'}</p>
    `;
    
    analysisDiv.style.display = 'block';
}

// Predict custom input
async function predictCustom() {
    if (!model) {
        alert('Please train the model first!');
        return;
    }
    
    const input = parseFloat(document.getElementById('customX').value);
    
    if (isNaN(input)) {
        alert('Please enter a valid number!');
        return;
    }
    
    console.log("\n" + "=".repeat(70));
    console.log("CUSTOM PREDICTION");
    console.log("=".repeat(70));
    
    const xTensor = tf.tensor2d([input], [1, 1]);
    const prediction = model.predict(xTensor);
    const predictedY = (await prediction.data())[0];
    const expectedY = 2 * input + 1;
    const error = Math.abs(predictedY - expectedY);
    
    console.log(`Input X: ${input}`);
    console.log(`Expected Y (true): ${expectedY.toFixed(4)}`);
    console.log(`Predicted Y: ${predictedY.toFixed(4)}`);
    console.log(`Absolute Error: ${error.toFixed(4)}`);
    console.log(`Error %: ${((error/Math.abs(expectedY))*100).toFixed(2)}%`);
    console.log("=".repeat(70));
    
    document.getElementById('customResult').innerHTML = 
        `Predicted: <span style="color: #00f2fe;">${predictedY.toFixed(4)}</span> ` +
        `(Expected: ${expectedY.toFixed(4)}, Error: ${error.toFixed(4)})`;
    
    xTensor.dispose();
    prediction.dispose();
}

// Reset all
function resetAll() {
    console.log("\n" + "=".repeat(70));
    console.log("RESETTING APPLICATION");
    console.log("=".repeat(70));
    
    if (model) model.dispose();
    model = null;
    
    trainingData = { x: [], y: [] };
    testData = { x: [], expected: [], predicted: [] };
    lossHistory = [];
    
    if (trainingChart) trainingChart.destroy();
    if (lossChart) lossChart.destroy();
    if (comparisonChart) comparisonChart.destroy();
    if (errorChart) errorChart.destroy();
    
    document.getElementById('stats').style.display = 'none';
    document.getElementById('status').style.display = 'none';
    document.getElementById('predictionsTableContainer').style.display = 'none';
    document.getElementById('comparisonCharts').style.display = 'none';
    document.getElementById('comparisonAnalysis').style.display = 'none';
    document.getElementById('customInput').style.display = 'none';
    document.getElementById('customX').value = '';
    document.getElementById('customResult').innerHTML = '';
    
    document.getElementById('predictBtn').disabled = true;
    document.getElementById('trainBtn').disabled = false;
    
    console.log("Reset complete!");
    console.log("=".repeat(70));
}

// Initialize
console.log("%c Linear Regression - Assignment 3 ", 
    "background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; font-size: 20px; padding: 10px; font-weight: bold;");
console.log("TensorFlow.js version:", tf.version.tfjs);
console.log("Ready! Follow the workflow steps to train, predict, and analyze.");
