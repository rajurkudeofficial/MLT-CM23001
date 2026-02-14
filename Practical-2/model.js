let model;
const canvas = document.getElementById("drawCanvas");
const ctx = canvas.getContext("2d");

let drawing = false;

// Initialize canvas
ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);

// Load pretrained TFJS model
async function loadModel() {
    console.log("Loading pretrained model...");
    model = await tf.loadGraphModel("tfjs_model/model.json");
    console.log("Model loaded successfully");
}

loadModel();

// Drawing logic
canvas.addEventListener("mousedown", () => drawing = true);
canvas.addEventListener("mouseup", () => drawing = false);
canvas.addEventListener("mouseleave", () => drawing = false);
canvas.addEventListener("mousemove", draw);

function draw(e) {
    if (!drawing) return;

    ctx.fillStyle = "white";
    ctx.beginPath();
    ctx.arc(e.offsetX, e.offsetY, 10, 0, Math.PI * 2);
    ctx.fill();
}

// Clear canvas
function clearCanvas() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

// Predict digit
async function predictDigit() {
    if (!model) {
        console.log("Model not loaded yet");
        return;
    }

    const imageData = ctx.getImageData(0, 0, 280, 280);

    let tensor = tf.browser.fromPixels(imageData, 1)
        .resizeNearestNeighbor([28, 28])
        .toFloat()
        .div(255.0)
        .expandDims(0);

    const prediction = model.predict(tensor);
    const digit = prediction.argMax(1).dataSync()[0];

    console.log("Predicted Digit:", digit);
}