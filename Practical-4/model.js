/*
 AIM:
 Image classification using pre-trained MobileNet model
 using TensorFlow.js in the browser.
*/

// Global variable for model
let model;

// Load MobileNet model when page loads
async function loadModel() {
    console.log("Loading MobileNet model...");
    
    model = await mobilenet.load();   // Load pre-trained MobileNet
    
    console.log("MobileNet model loaded successfully");
}

// Call model load function
loadModel();

/*
 Function to handle image upload
*/
function loadImage(event) {
    const image = document.getElementById("preview");
    image.src = URL.createObjectURL(event.target.files[0]);

    // Wait till image is loaded completely
    image.onload = () => {
        classifyImage(image);
    };
}

/*
 Function to classify image
*/
async function classifyImage(imgElement) {

    if (!model) {
        console.log("Model not loaded yet!");
        return;
    }

    console.log("Classifying image...");

    // Perform prediction
    const predictions = await model.classify(imgElement);

    // Print output in console
    console.log("Predictions:", predictions);

    // Display output in UI
    let output = `<b>Prediction Results:</b><br><br>`;
    predictions.forEach((pred, index) => {
        output += `${index + 1}. ${pred.className} 
                   (${(pred.probability * 100).toFixed(2)}%)<br>`;
    });

    document.getElementById("result").innerHTML = output;
}
