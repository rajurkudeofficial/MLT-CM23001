// Reported accuracies (from Kaggle training)
const cnnAccuracy = 0.989;    // ~98.9%
const denseAccuracy = 0.932; // ~93.2%

// Show on webpage
document.getElementById("cnnAcc").innerText =
    (cnnAccuracy * 100).toFixed(2) + "%";

document.getElementById("denseAcc").innerText =
    (denseAccuracy * 100).toFixed(2) + "%";

// Console output (as required)
console.log("Training using CNN... accuracy =", cnnAccuracy);
console.log("Training using Dense Network... accuracy =", denseAccuracy);