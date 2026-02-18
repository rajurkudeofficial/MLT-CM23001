// ---------- MobileNet V1 ----------
let modelV1;
let confidenceV1 = 0;

// Load MobileNet V1
async function loadMobileNetV1() {
    console.log("Loading MobileNet V1...");
    modelV1 = await mobilenet.load({ version: 1, alpha: 1.0 });
    console.log("MobileNet V1 Loaded");
}
loadMobileNetV1();

// Predict using MobileNet V1
async function predictV1(img) {
    const predictions = await modelV1.classify(img);
    confidenceV1 = predictions[0].probability;

    console.log("MobileNet V1 Prediction:", predictions);

    document.getElementById("mobilenetV1Result").innerHTML =
        `<b>MobileNet V1:</b> ${predictions[0].className}
        (${(confidenceV1 * 100).toFixed(2)}%)`;

    drawChart(); // update graph
}
