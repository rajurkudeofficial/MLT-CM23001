// ---------- MobileNet V2 ----------
let modelV2;
let confidenceV2 = 0;

// Load MobileNet V2
async function loadMobileNetV2() {
    console.log("Loading MobileNet V2...");
    modelV2 = await mobilenet.load({ version: 2, alpha: 1.0 });
    console.log("MobileNet V2 Loaded");
}
loadMobileNetV2();

// Image upload handler (COMMON)
function loadImage(event) {
    const img = document.getElementById("preview");
    img.src = URL.createObjectURL(event.target.files[0]);

    img.onload = () => {
        predictV1(img);
        predictV2(img);
    };
}

// Predict using MobileNet V2
async function predictV2(img) {
    const predictions = await modelV2.classify(img);
    confidenceV2 = predictions[0].probability;

    console.log("MobileNet V2 Prediction:", predictions);

    document.getElementById("mobilenetV2Result").innerHTML =
        `<b>MobileNet V2:</b> ${predictions[0].className}
        (${(confidenceV2 * 100).toFixed(2)}%)`;

    drawChart(); // update graph
}

// ---------- GRAPH LOGIC ----------
function drawChart() {
    const canvas = document.getElementById("chart");
    const ctx = canvas.getContext("2d");

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Axes
    ctx.beginPath();
    ctx.moveTo(50, 20);
    ctx.lineTo(50, 220);
    ctx.lineTo(380, 220);
    ctx.stroke();

    // MobileNet V1 bar
    ctx.fillStyle = "steelblue";
    ctx.fillRect(120, 220 - confidenceV1 * 160, 60, confidenceV1 * 160);

    // MobileNet V2 bar
    ctx.fillStyle = "seagreen";
    ctx.fillRect(240, 220 - confidenceV2 * 160, 60, confidenceV2 * 160);

    // Labels
    ctx.fillStyle = "black";
    ctx.fillText("V1", 135, 240);
    ctx.fillText("V2", 255, 240);

    ctx.fillText(
        (confidenceV1 * 100).toFixed(1) + "%",
        130,
        210 - confidenceV1 * 160
    );

    ctx.fillText(
        (confidenceV2 * 100).toFixed(1) + "%",
        250,
        210 - confidenceV2 * 160
    );
}
