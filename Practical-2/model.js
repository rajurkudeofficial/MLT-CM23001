// ============================================
// MNIST Digit Recognition using CNN
// TensorFlow.js Browser Implementation
// ============================================

console.log("🚀 TensorFlow.js Version:", tf.version.tfjs);
console.log("📱 Backend:", tf.getBackend());

// Global variables
let model;
let trainData, testData;
let trainImages, trainLabels, testImages, testLabels;

// ============================================
// STEP 1: LOAD AND PREPROCESS MNIST DATA
// ============================================

async function loadData() {
    updateStatus("Loading MNIST dataset...");
    console.log("\n=== LOADING MNIST DATASET ===");
    
    try {
        // Load MNIST data from TensorFlow.js hosted sprites
        const IMAGE_SIZE = 784; // 28x28
        const NUM_CLASSES = 10;
        const TRAIN_SIZE = 5000; // Reduced for browser performance
        const TEST_SIZE = 1000;
        
        console.log(`📦 Loading ${TRAIN_SIZE} training images and ${TEST_SIZE} test images...`);
        
        // Load data using tf.data (MNIST from TensorFlow.js)
        const data = await loadMNISTData(TRAIN_SIZE, TEST_SIZE);
        
        trainImages = data.trainImages;
        trainLabels = data.trainLabels;
        testImages = data.testImages;
        testLabels = data.testLabels;
        
        console.log("✅ Data loaded successfully!");
        console.log("📊 Training Images Shape:", trainImages.shape);
        console.log("📊 Training Labels Shape:", trainLabels.shape);
        console.log("📊 Test Images Shape:", testImages.shape);
        console.log("📊 Test Labels Shape:", testLabels.shape);
        
        // Verify data range
        const minVal = trainImages.min().dataSync()[0];
        const maxVal = trainImages.max().dataSync()[0];
        console.log(`📈 Pixel value range: [${minVal.toFixed(4)}, ${maxVal.toFixed(4)}]`);
        
        updateStatus("✅ Data loaded successfully! Ready to train.");
        document.getElementById("loadDataBtn").disabled = true;
        document.getElementById("trainBtn").disabled = false;
        
    } catch (error) {
        console.error("❌ Error loading data:", error);
        updateStatus("❌ Error loading data. Check console for details.");
    }
}

// Helper function to load MNIST data
async function loadMNISTData(trainSize, testSize) {
    // MNIST dataset URLs
    const MNIST_IMAGES_SPRITE_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
    const MNIST_LABELS_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';
    
    // Load the images sprite
    const img = new Image();
    const imgPromise = new Promise((resolve, reject) => {
        img.crossOrigin = '';
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = MNIST_IMAGES_SPRITE_PATH;
    });
    
    // Load the labels
    const labelsRequest = fetch(MNIST_LABELS_PATH);
    const [imgData, labelsResponse] = await Promise.all([imgPromise, labelsRequest]);
    const labelsData = new Uint8Array(await labelsResponse.arrayBuffer());
    
    // Process images from sprite
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const imageSize = 28;
    const totalImages = 65000;
    
    canvas.width = imageSize;
    canvas.height = imageSize;
    
    // Extract training images
    const trainImagesData = new Float32Array(trainSize * imageSize * imageSize);
    const trainLabelsData = new Uint8Array(trainSize);
    
    for (let i = 0; i < trainSize; i++) {
        const col = i % 1000;
        const row = Math.floor(i / 1000);
        
        ctx.drawImage(
            imgData,
            col * imageSize, row * imageSize, imageSize, imageSize,
            0, 0, imageSize, imageSize
        );
        
        const imageData = ctx.getImageData(0, 0, imageSize, imageSize);
        for (let j = 0; j < imageSize * imageSize; j++) {
            trainImagesData[i * imageSize * imageSize + j] = imageData.data[j * 4] / 255.0;
        }
        trainLabelsData[i] = labelsData[i];
    }
    
    // Extract test images
    const testImagesData = new Float32Array(testSize * imageSize * imageSize);
    const testLabelsData = new Uint8Array(testSize);
    
    for (let i = 0; i < testSize; i++) {
        const idx = 60000 + i; // Test set starts at index 60000
        const col = idx % 1000;
        const row = Math.floor(idx / 1000);
        
        ctx.drawImage(
            imgData,
            col * imageSize, row * imageSize, imageSize, imageSize,
            0, 0, imageSize, imageSize
        );
        
        const imageData = ctx.getImageData(0, 0, imageSize, imageSize);
        for (let j = 0; j < imageSize * imageSize; j++) {
            testImagesData[i * imageSize * imageSize + j] = imageData.data[j * 4] / 255.0;
        }
        testLabelsData[i] = labelsData[60000 + i];
    }
    
    // Convert to tensors and reshape
    const trainImages = tf.tensor4d(trainImagesData, [trainSize, 28, 28, 1]);
    const testImages = tf.tensor4d(testImagesData, [testSize, 28, 28, 1]);
    
    // One-hot encode labels
    const trainLabels = tf.oneHot(tf.tensor1d(trainLabelsData, 'int32'), 10);
    const testLabels = tf.oneHot(tf.tensor1d(testLabelsData, 'int32'), 10);
    
    return { trainImages, trainLabels, testImages, testLabels };
}

// ============================================
// STEP 2: BUILD CNN MODEL
// ============================================

function buildModel() {
    console.log("\n=== BUILDING CNN MODEL ===");
    
    const model = tf.sequential();
    
    // First Convolutional Block
    console.log("➕ Adding Conv2D layer (32 filters, 3x3 kernel)");
    model.add(tf.layers.conv2d({
        inputShape: [28, 28, 1],
        kernelSize: 3,
        filters: 32,
        activation: 'relu',
        kernelInitializer: 'heNormal'
    }));
    
    console.log("➕ Adding MaxPooling2D layer (2x2 pool)");
    model.add(tf.layers.maxPooling2d({
        poolSize: 2,
        strides: 2
    }));
    
    // Second Convolutional Block
    console.log("➕ Adding Conv2D layer (64 filters, 3x3 kernel)");
    model.add(tf.layers.conv2d({
        kernelSize: 3,
        filters: 64,
        activation: 'relu',
        kernelInitializer: 'heNormal'
    }));
    
    console.log("➕ Adding MaxPooling2D layer (2x2 pool)");
    model.add(tf.layers.maxPooling2d({
        poolSize: 2,
        strides: 2
    }));
    
    // Flatten and Dense Layers
    console.log("➕ Adding Flatten layer");
    model.add(tf.layers.flatten());
    
    console.log("➕ Adding Dense layer (128 units, ReLU)");
    model.add(tf.layers.dense({
        units: 128,
        activation: 'relu',
        kernelInitializer: 'heNormal'
    }));
    
    console.log("➕ Adding Output Dense layer (10 units, Softmax)");
    model.add(tf.layers.dense({
        units: 10,
        activation: 'softmax'
    }));
    
    console.log("\n📊 MODEL ARCHITECTURE:");
    model.summary();
    
    return model;
}

// ============================================
// STEP 3: COMPILE AND TRAIN MODEL
// ============================================

async function trainModel() {
    updateStatus("Building and compiling model...");
    console.log("\n=== TRAINING CNN MODEL ===");
    
    try {
        // Build model
        model = buildModel();
        
        // Compile model
        console.log("\n⚙️ Compiling model...");
        console.log("   Optimizer: Adam (learning rate: 0.001)");
        console.log("   Loss: Categorical Crossentropy");
        console.log("   Metrics: Accuracy");
        
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
        
        // Training parameters
        const EPOCHS = 5;
        const BATCH_SIZE = 128;
        
        console.log(`\n🏋️ Training for ${EPOCHS} epochs with batch size ${BATCH_SIZE}`);
        updateStatus(`Training model (${EPOCHS} epochs)...`);
        
        // Train the model
        const history = await model.fit(trainImages, trainLabels, {
            epochs: EPOCHS,
            batchSize: BATCH_SIZE,
            validationSplit: 0.15,
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    console.log(`\n📈 Epoch ${epoch + 1}/${EPOCHS}`);
                    console.log(`   Loss: ${logs.loss.toFixed(4)}`);
                    console.log(`   Accuracy: ${(logs.acc * 100).toFixed(2)}%`);
                    console.log(`   Val Loss: ${logs.val_loss.toFixed(4)}`);
                    console.log(`   Val Accuracy: ${(logs.val_acc * 100).toFixed(2)}%`);
                    
                    // Update UI
                    document.getElementById("epochValue").textContent = `${epoch + 1}/${EPOCHS}`;
                    document.getElementById("trainAccValue").textContent = `${(logs.acc * 100).toFixed(2)}%`;
                    document.getElementById("trainLossValue").textContent = logs.loss.toFixed(4);
                    document.getElementById("valAccValue").textContent = `${(logs.val_acc * 100).toFixed(2)}%`;
                    
                    updateStatus(`Training... Epoch ${epoch + 1}/${EPOCHS} - Acc: ${(logs.acc * 100).toFixed(2)}%`);
                    
                    await tf.nextFrame();
                }
            }
        });
        
        console.log("\n✅ Training completed!");
        
        // Evaluate on test set
        console.log("\n🧪 EVALUATING ON TEST SET:");
        const evalResult = model.evaluate(testImages, testLabels);
        const testLoss = evalResult[0].dataSync()[0];
        const testAcc = evalResult[1].dataSync()[0];
        
        console.log(`   Test Loss: ${testLoss.toFixed(4)}`);
        console.log(`   Test Accuracy: ${(testAcc * 100).toFixed(2)}%`);
        
        updateStatus(`✅ Training complete! Test Accuracy: ${(testAcc * 100).toFixed(2)}%`);
        
        // Enable prediction button
        document.getElementById("trainBtn").disabled = true;
        document.getElementById("predictBtn").disabled = false;
        
        // Cleanup
        evalResult.forEach(t => t.dispose());
        
    } catch (error) {
        console.error("❌ Training error:", error);
        updateStatus("❌ Training failed. Check console for details.");
    }
}

// ============================================
// STEP 4: MAKE PREDICTIONS
// ============================================

async function predictDigit() {
    console.log("\n=== MAKING PREDICTION ===");
    
    // Get random test sample
    const randomIndex = Math.floor(Math.random() * testImages.shape[0]);
    console.log(`🎲 Selected random test image at index: ${randomIndex}`);
    
    // Get single image
    const testImage = testImages.slice([randomIndex, 0, 0, 0], [1, 28, 28, 1]);
    const trueLabel = testLabels.slice([randomIndex, 0], [1, 10]);
    
    // Get true label
    const trueLabelIndex = trueLabel.argMax(-1).dataSync()[0];
    console.log(`📌 True Label: ${trueLabelIndex}`);
    
    // Make prediction
    const prediction = model.predict(testImage);
    const predictedClass = prediction.argMax(-1).dataSync()[0];
    const confidence = prediction.max().dataSync()[0];
    
    console.log(`🎯 Predicted Label: ${predictedClass}`);
    console.log(`📊 Confidence: ${(confidence * 100).toFixed(2)}%`);
    console.log(`✅ Correct: ${predictedClass === trueLabelIndex ? 'YES' : 'NO'}`);
    
    // Display all class probabilities
    const probabilities = prediction.dataSync();
    console.log("\n📊 Class Probabilities:");
    for (let i = 0; i < 10; i++) {
        console.log(`   Digit ${i}: ${(probabilities[i] * 100).toFixed(2)}%`);
    }
    
    // Draw digit on canvas
    drawDigit(testImage, randomIndex);
    
    // Update UI
    document.getElementById("predictedDigit").textContent = predictedClass;
    document.getElementById("confidenceValue").textContent = (confidence * 100).toFixed(2);
    document.getElementById("predictionResult").classList.add("active");
    
    // Update status
    const isCorrect = predictedClass === trueLabelIndex;
    const statusIcon = isCorrect ? "✅" : "❌";
    updateStatus(`${statusIcon} Prediction: ${predictedClass} | True: ${trueLabelIndex} | Confidence: ${(confidence * 100).toFixed(2)}%`);
    
    // Cleanup
    testImage.dispose();
    trueLabel.dispose();
    prediction.dispose();
}

// ============================================
// HELPER FUNCTIONS
// ============================================

function drawDigit(imageTensor, index) {
    const canvas = document.getElementById("digitCanvas");
    const ctx = canvas.getContext("2d");
    
    // Get image data
    const imageData = imageTensor.reshape([28, 28]).arraySync();
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Scale factor for better visibility
    const scale = 10;
    
    // Draw each pixel
    for (let i = 0; i < 28; i++) {
        for (let j = 0; j < 28; j++) {
            const value = Math.floor(imageData[i][j] * 255);
            ctx.fillStyle = `rgb(${value}, ${value}, ${value})`;
            ctx.fillRect(j * scale, i * scale, scale, scale);
        }
    }
    
    console.log(`🖼️ Digit displayed on canvas (Image #${index})`);
}

function updateStatus(message) {
    document.getElementById("status").textContent = message;
    console.log(`📢 Status: ${message}`);
}

// ============================================
// INITIALIZATION
// ============================================

console.log("\n" + "=".repeat(50));
console.log("🧠 MNIST DIGIT RECOGNITION - CNN");
console.log("   TensorFlow.js Browser Implementation");
console.log("=".repeat(50));
console.log("\n📋 Instructions:");
console.log("   1. Click 'Load MNIST Data' to download dataset");
console.log("   2. Click 'Train Model' to train CNN (5 epochs)");
console.log("   3. Click 'Predict Random Digit' to test model");
console.log("   4. Check console for detailed metrics");
console.log("\n✅ System ready. Waiting for user interaction...\n");
