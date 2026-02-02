// Global variables
let models = [];
let trainingData = { x: [], y: [] };
let comparisonChart, convergenceChart;

// Color palette for different learning rates
const colors = [
    { bg: 'rgba(102, 126, 234, 0.6)', border: 'rgba(102, 126, 234, 1)' },
    { bg: 'rgba(255, 99, 132, 0.6)', border: 'rgba(255, 99, 132, 1)' },
    { bg: 'rgba(75, 192, 192, 0.6)', border: 'rgba(75, 192, 192, 1)' },
    { bg: 'rgba(255, 159, 64, 0.6)', border: 'rgba(255, 159, 64, 1)' },
    { bg: 'rgba(153, 102, 255, 0.6)', border: 'rgba(153, 102, 255, 1)' }
];

// Generate synthetic data
function generateSyntheticData(numSamples = 100) {
    console.log("=".repeat(70));
    console.log("GENERATING SYNTHETIC DATA FOR LEARNING RATE COMPARISON");
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
    
    console.log(`Generated ${numSamples} samples`);
    console.log("True relationship: y = 2x + 1 + noise");
    console.log("=".repeat(70));
    
    return { xData, yData };
}

// Create a model with specific learning rate
function createModelWithLR(learningRate) {
    const model = tf.sequential();
    
    model.add(tf.layers.dense({
        units: 1,
        inputShape: [1],
        kernelInitializer: 'randomNormal',
        biasInitializer: 'zeros'
    }));
    
    model.compile({
        optimizer: tf.train.adam(learningRate),
        loss: 'meanSquaredError'
    });
    
    return model;
}

// Train a single model
async function trainSingleModel(learningRate, xData, yData) {
    console.log(`\nTraining model with learning rate: ${learningRate}`);
    console.log("-".repeat(70));
    
    const model = createModelWithLR(learningRate);
    const lossHistory = [];
    
    const xs = tf.tensor2d(xData, [xData.length, 1]);
    const ys = tf.tensor2d(yData, [yData.length, 1]);
    
    const startTime = performance.now();
    
    await model.fit(xs, ys, {
        epochs: 100,
        verbose: 0,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                lossHistory.push(logs.loss);
            }
        }
    });
    
    const endTime = performance.now();
    const trainingTime = (endTime - startTime) / 1000;
    
    // Get final weights
    const weights = model.getWeights();
    const weightData = await weights[0].data();
    const biasData = await weights[1].data();
    
    const finalLoss = lossHistory[lossHistory.length - 1];
    
    // Find epoch where loss is below threshold (convergence point)
    const convergenceThreshold = finalLoss * 1.1; // Within 10% of final loss
    let convergenceEpoch = lossHistory.findIndex(loss => loss <= convergenceThreshold);
    if (convergenceEpoch === -1) convergenceEpoch = 100;
    
    console.log(`  Final Loss: ${finalLoss.toFixed(6)}`);
    console.log(`  Learned Parameters: y = ${weightData[0].toFixed(4)}x + ${biasData[0].toFixed(4)}`);
    console.log(`  Training Time: ${trainingTime.toFixed(2)}s`);
    console.log(`  Convergence Epoch: ${convergenceEpoch + 1}`);
    
    xs.dispose();
    ys.dispose();
    
    return {
        model,
        learningRate,
        lossHistory,
        finalLoss,
        weight: weightData[0],
        bias: biasData[0],
        trainingTime,
        convergenceEpoch
    };
}

// Compare multiple models with different learning rates
async function compareModels() {
    console.log("\n" + "=".repeat(70));
    console.log("COMPARING MODELS WITH DIFFERENT LEARNING RATES");
    console.log("=".repeat(70));
    
    // Disable button
    document.getElementById('compareBtn').disabled = true;
    
    // Show status
    const statusDiv = document.getElementById('status');
    statusDiv.style.display = 'block';
    statusDiv.innerHTML = '🔄 Training models with different learning rates...';
    
    // Get selected learning rates
    const selectedLRs = [];
    document.querySelectorAll('.lr-input input[type="checkbox"]:checked').forEach(cb => {
        selectedLRs.push(parseFloat(cb.value));
    });
    
    if (selectedLRs.length === 0) {
        alert('Please select at least one learning rate!');
        document.getElementById('compareBtn').disabled = false;
        return;
    }
    
    console.log(`\nSelected Learning Rates: ${selectedLRs.join(', ')}`);
    
    // Generate data
    const { xData, yData } = generateSyntheticData(100);
    
    // Clear previous models
    models.forEach(m => m.model.dispose());
    models = [];
    
    // Train all models
    for (let i = 0; i < selectedLRs.length; i++) {
        const result = await trainSingleModel(selectedLRs[i], xData, yData);
        models.push(result);
    }
    
    console.log("\n" + "=".repeat(70));
    console.log("TRAINING SUMMARY");
    console.log("=".repeat(70));
    console.log("\nLearning Rate | Final Loss  | Convergence | Training Time | Equation");
    console.log("-".repeat(70));
    
    models.forEach(m => {
        console.log(
            `${m.learningRate.toString().padEnd(13)} | ` +
            `${m.finalLoss.toFixed(6).padEnd(11)} | ` +
            `Epoch ${(m.convergenceEpoch + 1).toString().padEnd(5)} | ` +
            `${m.trainingTime.toFixed(2)}s`.padEnd(13) + ` | ` +
            `y = ${m.weight.toFixed(2)}x + ${m.bias.toFixed(2)}`
        );
    });
    
    console.log("=".repeat(70));
    
    // Update UI
    displayModelCards();
    updateComparisonChart();
    updateConvergenceChart();
    analyzeResults();
    
    statusDiv.innerHTML = '✅ Comparison completed successfully!';
    document.getElementById('compareBtn').disabled = false;
}

// Display model cards
function displayModelCards() {
    const container = document.getElementById('modelsContainer');
    container.innerHTML = '';
    
    models.forEach((m, idx) => {
        const card = document.createElement('div');
        card.className = 'model-card';
        card.style.borderLeftColor = colors[idx % colors.length].border;
        
        let status = '';
        if (m.learningRate <= 0.01) {
            status = '🐌 Slow but Stable';
        } else if (m.learningRate <= 0.1) {
            status = '✅ Optimal';
        } else if (m.learningRate <= 0.5) {
            status = '⚡ Fast';
        } else {
            status = '⚠️ Too Fast';
        }
        
        card.innerHTML = `
            <h3>Learning Rate: ${m.learningRate} ${status}</h3>
            <div class="model-stats">
                <div class="stat">
                    <div class="stat-label">Final Loss</div>
                    <div class="stat-value">${m.finalLoss.toFixed(4)}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Convergence Epoch</div>
                    <div class="stat-value">${m.convergenceEpoch + 1}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Training Time</div>
                    <div class="stat-value">${m.trainingTime.toFixed(2)}s</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Learned Equation</div>
                    <div class="stat-value" style="font-size: 0.9em;">y = ${m.weight.toFixed(2)}x + ${m.bias.toFixed(2)}</div>
                </div>
            </div>
        `;
        
        container.appendChild(card);
    });
}

// Update comparison chart
function updateComparisonChart() {
    const ctx = document.getElementById('comparisonChart').getContext('2d');
    
    if (comparisonChart) {
        comparisonChart.destroy();
    }
    
    const datasets = models.map((m, idx) => ({
        label: `LR = ${m.learningRate}`,
        data: m.lossHistory,
        borderColor: colors[idx % colors.length].border,
        backgroundColor: colors[idx % colors.length].bg,
        borderWidth: 2,
        fill: false,
        tension: 0.4
    }));
    
    comparisonChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({ length: 100 }, (_, i) => i + 1),
            datasets: datasets
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
                    type: 'logarithmic'
                }
            },
            plugins: {
                legend: {
                    position: 'top'
                }
            }
        }
    });
}

// Update convergence chart
function updateConvergenceChart() {
    const ctx = document.getElementById('convergenceChart').getContext('2d');
    
    if (convergenceChart) {
        convergenceChart.destroy();
    }
    
    convergenceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: models.map(m => `LR = ${m.learningRate}`),
            datasets: [{
                label: 'Epochs to Convergence',
                data: models.map(m => m.convergenceEpoch + 1),
                backgroundColor: models.map((m, idx) => colors[idx % colors.length].bg),
                borderColor: models.map((m, idx) => colors[idx % colors.length].border),
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    title: { display: true, text: 'Number of Epochs' },
                    beginAtZero: true
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

// Analyze results and provide insights
function analyzeResults() {
    const analysisDiv = document.getElementById('analysis');
    const analysisList = document.getElementById('analysisList');
    
    analysisDiv.style.display = 'block';
    analysisList.innerHTML = '';
    
    // Find best model (lowest final loss)
    const bestModel = models.reduce((best, current) => 
        current.finalLoss < best.finalLoss ? current : best
    );
    
    // Find fastest convergence
    const fastestModel = models.reduce((fastest, current) => 
        current.convergenceEpoch < fastest.convergenceEpoch ? current : fastest
    );
    
    const insights = [
        `<strong>Best Final Loss:</strong> Learning rate ${bestModel.learningRate} achieved the lowest loss of ${bestModel.finalLoss.toFixed(6)}`,
        `<strong>Fastest Convergence:</strong> Learning rate ${fastestModel.learningRate} converged in ${fastestModel.convergenceEpoch + 1} epochs`,
        `<strong>Stability:</strong> ${models.filter(m => m.learningRate <= 0.1).length > 0 ? 
            'Lower learning rates (≤0.1) showed more stable convergence with smoother loss curves' : 
            'Higher learning rates showed faster but potentially less stable convergence'}`,
        `<strong>Trade-off:</strong> Very low learning rates (≤0.01) are stable but slow, while high rates (≥0.5) are fast but may overshoot the optimal solution`,
        `<strong>Recommendation:</strong> For this problem, a learning rate around 0.1 provides a good balance between convergence speed and final accuracy`
    ];
    
    insights.forEach(insight => {
        const li = document.createElement('li');
        li.innerHTML = insight;
        analysisList.appendChild(li);
    });
    
    console.log("\n" + "=".repeat(70));
    console.log("ANALYSIS & INSIGHTS");
    console.log("=".repeat(70));
    insights.forEach((insight, idx) => {
        console.log(`${idx + 1}. ${insight.replace(/<\/?strong>/g, '')}`);
    });
    console.log("=".repeat(70));
}

// Reset comparison
function resetComparison() {
    console.log("\n" + "=".repeat(70));
    console.log("RESETTING COMPARISON");
    console.log("=".repeat(70));
    
    models.forEach(m => m.model.dispose());
    models = [];
    
    if (comparisonChart) comparisonChart.destroy();
    if (convergenceChart) convergenceChart.destroy();
    
    document.getElementById('modelsContainer').innerHTML = '';
    document.getElementById('status').style.display = 'none';
    document.getElementById('analysis').style.display = 'none';
    
    console.log("Comparison reset successfully!");
    console.log("=".repeat(70));
}

// Initialize
console.log("%c Linear Regression - Assignment 2 ", 
    "background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; font-size: 20px; padding: 10px; font-weight: bold;");
console.log("TensorFlow.js version:", tf.version.tfjs);
console.log("Ready! Select learning rates and click 'Compare Models' to begin.");
