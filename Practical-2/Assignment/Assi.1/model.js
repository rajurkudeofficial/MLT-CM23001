// ============================================
// Assignment 1: MobileNet Image Classification
// TensorFlow.js Browser Implementation
// ============================================

console.log("🚀 TensorFlow.js Version:", tf.version.tfjs);
console.log("📱 Backend:", tf.getBackend());

// Global variables
let mobilenetModel;

// Sample image URL - feel free to change this
const IMAGE_URL = 'https://images.unsplash.com/photo-1574158622682-e40e69881006?w=500';

// ============================================
// LOAD MOBILENET MODEL
// ============================================

async function loadModel() {
    updateStatus("Loading MobileNet model...");
    console.log("\n=== LOADING MOBILENET MODEL ===");
    
    try {
        console.log("📦 Downloading pre-trained MobileNet...");
        console.log("   Model: MobileNet v1");
        console.log("   Input size: 224x224x3");
        console.log("   Classes: 1000 ImageNet categories");
        
        mobilenetModel = await mobilenet.load();
        
        console.log("✅ Model loaded successfully!");
        console.log("📊 Model Details:");
        console.log("   - Architecture: MobileNet v1");
        console.log("   - Pre-trained on: ImageNet dataset");
        console.log("   - Total classes: 1000");
        
        updateStatus("✅ Model loaded! Ready to classify.");
        document.getElementById("classifyBtn").disabled = false;
        
        // Load and display the image
        loadImage();
        
    } catch (error) {
        console.error("❌ Error loading model:", error);
        updateStatus("❌ Error loading model. Check console.");
    }
}

// ============================================
// LOAD IMAGE
// ============================================

function loadImage() {
    console.log("\n=== LOADING IMAGE ===");
    console.log("📷 Image URL:", IMAGE_URL);
    
    const imgElement = document.getElementById("imageDisplay");
    imgElement.src = IMAGE_URL;
    imgElement.style.display = "block";
    
    imgElement.onload = () => {
        console.log("✅ Image loaded successfully");
        console.log(`   Dimensions: ${imgElement.naturalWidth}x${imgElement.naturalHeight}`);
    };
    
    imgElement.onerror = () => {
        console.error("❌ Error loading image");
        updateStatus("❌ Error loading image. Check URL or connection.");
    };
}

// ============================================
// CLASSIFY IMAGE
// ============================================

async function classifyImage() {
    updateStatus("Classifying image...");
    console.log("\n=== CLASSIFYING IMAGE ===");
    
    try {
        const imgElement = document.getElementById("imageDisplay");
        
        console.log("🔍 Running MobileNet inference...");
        const startTime = performance.now();
        
        // Get top-3 predictions
        const predictions = await mobilenetModel.classify(imgElement, 3);
        
        const endTime = performance.now();
        const inferenceTime = (endTime - startTime).toFixed(2);
        
        console.log(`⏱️ Inference time: ${inferenceTime}ms`);
        console.log("\n📊 TOP-3 PREDICTIONS:");
        console.log("─".repeat(60));
        
        predictions.forEach((pred, index) => {
            console.log(`${index + 1}. ${pred.className}`);
            console.log(`   Confidence: ${(pred.probability * 100).toFixed(2)}%`);
            console.log(`   Probability: ${pred.probability.toFixed(6)}`);
            console.log("─".repeat(60));
        });
        
        // Display predictions on webpage
        displayPredictions(predictions);
        
        updateStatus(`✅ Classification complete! (${inferenceTime}ms)`);
        
    } catch (error) {
        console.error("❌ Classification error:", error);
        updateStatus("❌ Classification failed. Check console.");
    }
}

// ============================================
// DISPLAY PREDICTIONS
// ============================================

function displayPredictions(predictions) {
    const container = document.getElementById("predictions");
    container.innerHTML = "";
    
    predictions.forEach((pred, index) => {
        const item = document.createElement("div");
        item.className = "prediction-item";
        
        item.innerHTML = `
            <div class="rank">#${index + 1}</div>
            <div class="class-name">${pred.className}</div>
            <div class="confidence">${(pred.probability * 100).toFixed(2)}%</div>
        `;
        
        container.appendChild(item);
    });
    
    console.log("✅ Predictions displayed on webpage");
}

// ============================================
// HELPER FUNCTIONS
// ============================================

function updateStatus(message) {
    document.getElementById("status").textContent = message;
    console.log(`📢 Status: ${message}`);
}

// ============================================
// INITIALIZATION
// ============================================

console.log("\n" + "=".repeat(60));
console.log("🖼️ ASSIGNMENT 1: MOBILENET IMAGE CLASSIFICATION");
console.log("   Pre-trained Model using TensorFlow.js");
console.log("=".repeat(60));
console.log("\n📋 Instructions:");
console.log("   1. Click 'Load MobileNet Model' to download model");
console.log("   2. Image will load automatically");
console.log("   3. Click 'Classify Image' to get predictions");
console.log("   4. View top-3 classes with confidence scores");
console.log("\n✅ System ready. Waiting for user interaction...\n");
