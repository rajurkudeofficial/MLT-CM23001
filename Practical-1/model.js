// Check if TensorFlow.js is loaded
console.log("TensorFlow.js version:", tf.version.tfjs);

// -----------------------------
// Step 1: Create Synthetic Data
// y = 2x + 1 (approx)
// -----------------------------
const xValues = tf.tensor1d([1, 2, 3, 4, 5]);
const yValues = tf.tensor1d([3, 5, 7, 9, 11]);

console.log("Synthetic X Data:", xValues.arraySync());
console.log("Synthetic Y Data:", yValues.arraySync());

// -----------------------------
// Step 2: Create Model
// -----------------------------
const model = tf.sequential();

model.add(
    tf.layers.dense({
        units: 1,
        inputShape: [1]
    })
);

// -----------------------------
// Step 3: Compile Model
// -----------------------------
model.compile({
    optimizer: 'sgd',
    loss: 'meanSquaredError'
});

console.log("Model compiled successfully");

// -----------------------------
// Step 4: Train Model
// -----------------------------
async function trainModel() {
    console.log("Training started...");

    await model.fit(xValues, yValues, {
        epochs: 200,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                if (epoch % 50 === 0) {
                    console.log(`Epoch ${epoch} - Loss: ${logs.loss}`);
                }
            }
        }
    });

    console.log("Training completed");

    // -----------------------------
    // Step 5: Prediction
    // -----------------------------
    const testInput = tf.tensor1d([6]);
    const prediction = model.predict(testInput);

    console.log(
        "Prediction for x = 6:",
        prediction.arraySync()[0]
    );
}

// Run training
trainModel();
