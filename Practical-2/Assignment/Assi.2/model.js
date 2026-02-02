// ============================================
// Assignment 2: Multiple Image Classification
// TensorFlow.js Browser Implementation
// ============================================

console.log("🚀 TensorFlow.js Version:", tf.version.tfjs);
console.log("📱 Backend:", tf.getBackend());

// Global variables
let mobilenetModel;

// Array of 5 different test images with expected labels for accuracy analysis
const TEST_IMAGES = [
    {
        url: 'https://images.unsplash.com/photo-1574158622682-e40e69881006?w=400',
        description: 'Cat Image',
        expected: 'cat'
    },
    {
        url: 'https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=400',
        description: 'Dog Image',
        expected: 'dog'
    },
    {
        url: 'https://images.unsplash.com/photo-1568572933382-74d440642117?w=400',
        description: 'Coffee Cup',
        expected: 'coffee'
    },
    {
        url: 'https://images.unsplash.com/photo-1511593358241-7eea1f3c84e5?w=400',
        description: 'Wristwatch',
        expected: 'watch'
    },
    {
        url: 'https://images.unsplash.com/photo-1568605114967-8130f3a36994?w=400',
        description: 'Sunglasses',
        expected: 'sunglasses'
    }
];

// Store classification results
let classificationResults = [];

// ============================================
// LOAD MOBILENET MODEL
// ============================================

async function loadModel() {
    updateStatus("Loading MobileNet model...");
    console.log("\n=== LOADING MOBILENET MODEL ===");
    
    try {
        console.log("📦 Downloading pre-trained MobileNet...");
        mobilenetModel = await mobilenet.load();
        
        console.log("✅ Model loaded successfully!");
        console.log(`📊 Ready to classify ${TEST_IMAGES.length} test images`);
        
        updateStatus("✅ Model loaded! Ready to classify images.");
        document.getElementById("classifyBtn").disabled = false;
        
    } catch (error) {
        console.error("❌ Error loading model:", error);
        updateStatus("❌ Error loading model. Check console.");
    }
}

// ============================================
// CLASSIFY ALL IMAGES
// ============================================

async function classifyAllImages() {
    updateStatus("Classifying multiple images...");
    console.log("\n=== CLASSIFYING MULTIPLE IMAGES ===");
    console.log(`📷 Total images to classify: ${TEST_IMAGES.length}`);
    console.log("─".repeat(70));
    
    classificationResults = [];
    const resultsBody = document.getElementById("resultsBody");
    resultsBody.innerHTML = "";
    
    try {
        for (let i = 0; i < TEST_IMAGES.length; i++) {
            const imageData = TEST_IMAGES[i];
            
            console.log(`\n🖼️ Image ${i + 1}/${TEST_IMAGES.length}: ${imageData.description}`);
            console.log(`   URL: ${imageData.url}`);
            
            updateStatus(`Classifying image ${i + 1}/${TEST_IMAGES.length}...`);
            
            // Load image
            const img = await loadImageElement(imageData.url);
            
            // Classify
            const startTime = performance.now();
            const predictions = await mobilenetModel.classify(img, 3);
            const endTime = performance.now();
            
            const topPrediction = predictions[0];
            const inferenceTime = (endTime - startTime).toFixed(2);
            
            console.log(`   ⏱️ Inference time: ${inferenceTime}ms`);
            console.log(`   🎯 Top Prediction: ${topPrediction.className}`);
            console.log(`   📊 Confidence: ${(topPrediction.probability * 100).toFixed(2)}%`);
            
            // Check if prediction is correct
            const isCorrect = topPrediction.className.toLowerCase().includes(imageData.expected.toLowerCase());
            console.log(`   ${isCorrect ? '✅' : '❌'} Expected: ${imageData.expected} | Correct: ${isCorrect ? 'YES' : 'NO'}`);
            
            // Display all top-3 predictions
            console.log("   📋 Top-3 Predictions:");
            predictions.forEach((pred, idx) => {
                console.log(`      ${idx + 1}. ${pred.className} (${(pred.probability * 100).toFixed(2)}%)`);
            });
            
            // Store result
            classificationResults.push({
                index: i + 1,
                imageUrl: imageData.url,
                description: imageData.description,
                expected: imageData.expected,
                predicted: topPrediction.className,
                confidence: topPrediction.probability,
                inferenceTime: inferenceTime,
                isCorrect: isCorrect,
                allPredictions: predictions
            });
            
            // Add to table
            addResultRow(classificationResults[i]);
            
            console.log("─".repeat(70));
        }
        
        // Show analysis
        displayAnalysis();
        
        updateStatus(`✅ Classification complete! Processed ${TEST_IMAGES.length} images.`);
        
    } catch (error) {
        console.error("❌ Classification error:", error);
        updateStatus("❌ Classification failed. Check console.");
    }
}

// ============================================
// LOAD IMAGE ELEMENT
// ============================================

function loadImageElement(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = url;
    });
}

// ============================================
// ADD RESULT TO TABLE
// ============================================

function addResultRow(result) {
    const tbody = document.getElementById("resultsBody");
    const row = document.createElement("tr");
    
    const confidencePercent = (result.confidence * 100).toFixed(2);
    
    row.innerHTML = `
        <td><strong>${result.index}</strong></td>
        <td>
            <img src="${result.imageUrl}" class="image-thumb" alt="${result.description}">
            <div style="margin-top: 5px; font-size: 0.9em; color: #666;">${result.description}</div>
        </td>
        <td>
            <strong style="color: #333; font-size: 1.1em;">${result.predicted}</strong>
            <div style="margin-top: 5px; font-size: 0.85em; color: ${result.isCorrect ? '#28a745' : '#dc3545'};">
                ${result.isCorrect ? '✅' : '❌'} Expected: ${result.expected}
            </div>
        </td>
        <td>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${confidencePercent}%">
                    ${confidencePercent}%
                </div>
            </div>
            <div style="margin-top: 5px; font-size: 0.85em; color: #666;">
                Inference: ${result.inferenceTime}ms
            </div>
        </td>
    `;
    
    tbody.appendChild(row);
}

// ============================================
// DISPLAY ACCURACY ANALYSIS
// ============================================

function displayAnalysis() {
    console.log("\n=== ACCURACY ANALYSIS ===");
    
    const correctCount = classificationResults.filter(r => r.isCorrect).length;
    const totalCount = classificationResults.length;
    const accuracy = ((correctCount / totalCount) * 100).toFixed(2);
    
    const avgConfidence = (classificationResults.reduce((sum, r) => sum + r.confidence, 0) / totalCount * 100).toFixed(2);
    const avgInferenceTime = (classificationResults.reduce((sum, r) => sum + parseFloat(r.inferenceTime), 0) / totalCount).toFixed(2);
    
    console.log(`📊 Accuracy: ${correctCount}/${totalCount} (${accuracy}%)`);
    console.log(`📈 Average Confidence: ${avgConfidence}%`);
    console.log(`⏱️ Average Inference Time: ${avgInferenceTime}ms`);
    
    // Detailed results
    console.log("\n📋 DETAILED RESULTS:");
    classificationResults.forEach((result, idx) => {
        console.log(`${idx + 1}. ${result.description}`);
        console.log(`   Predicted: ${result.predicted} | Expected: ${result.expected}`);
        console.log(`   ${result.isCorrect ? '✅ CORRECT' : '❌ INCORRECT'} | Confidence: ${(result.confidence * 100).toFixed(2)}%`);
    });
    
    // Display on webpage
    const analysisSection = document.getElementById("analysisSection");
    const analysisContent = document.getElementById("analysisContent");
    
    analysisContent.innerHTML = `
        <h3>📊 Performance Summary</h3>
        <p><strong>Overall Accuracy:</strong> ${correctCount}/${totalCount} correct predictions (${accuracy}%)</p>
        <p><strong>Average Confidence:</strong> ${avgConfidence}%</p>
        <p><strong>Average Inference Time:</strong> ${avgInferenceTime}ms</p>
        
        <h3 style="margin-top: 20px;">💡 Observations</h3>
        <p>
            The MobileNet model achieved ${accuracy}% accuracy on this test set of ${totalCount} images. 
            ${correctCount === totalCount 
                ? 'All predictions were correct, demonstrating excellent performance on these common object categories.' 
                : `${totalCount - correctCount} prediction(s) were incorrect, which may be due to image quality, object positioning, or similarity to other classes in the ImageNet dataset.`}
            The average confidence score of ${avgConfidence}% indicates ${parseFloat(avgConfidence) > 80 ? 'high model certainty' : 'moderate model certainty'} in its predictions.
            Average inference time of ${avgInferenceTime}ms demonstrates efficient real-time classification capability.
        </p>
        
        <h3 style="margin-top: 20px;">🔍 Analysis</h3>
        <p>
            MobileNet is optimized for mobile and web deployment, trading some accuracy for speed and model size. 
            The model performs best on objects that are well-represented in the ImageNet dataset and clearly visible in the image.
            Factors affecting accuracy include image quality, object clarity, background complexity, and similarity between object categories.
        </p>
    `;
    
    analysisSection.style.display = "block";
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

console.log("\n" + "=".repeat(70));
console.log("📊 ASSIGNMENT 2: MULTIPLE IMAGE CLASSIFICATION");
console.log("   Testing Accuracy Across Different Images");
console.log("=".repeat(70));
console.log("\n📋 Test Images:");
TEST_IMAGES.forEach((img, idx) => {
    console.log(`   ${idx + 1}. ${img.description} (Expected: ${img.expected})`);
});
console.log("\n✅ System ready. Click 'Load MobileNet Model' to begin...\n");
