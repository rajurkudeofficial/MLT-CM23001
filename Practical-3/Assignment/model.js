let model;
let tokenizer;

const MAX_LEN = 100;

/* ======================
   LOAD SELF-TRAINED MODEL
   ====================== */
async function loadModels() {
  console.log("Loading trained sentiment model...");

  model = await tf.loadGraphModel("Training/model.json");

  const response = await fetch("Training/tokenizer.json");
  const tokJson = await response.json();

  tokenizer = tokJson.word_index
    ? tokJson
    : { word_index: tokJson.config.word_index };

  console.log("Model & tokenizer loaded successfully!");
}

loadModels();

/* ======================
   TEXT → SEQUENCE
   ====================== */
function textToSequence(text) {
  const words = text.toLowerCase().split(/\s+/);
  const sequence = [];

  for (let w of words) {
    sequence.push(tokenizer.word_index[w] || 1);
  }

  while (sequence.length < MAX_LEN) sequence.push(0);
  return sequence.slice(0, MAX_LEN);
}

/* ======================
   TASK 2: SELF-TRAINED MODEL
   ====================== */
async function predictSelf() {
  if (!model || !tokenizer) {
    alert("Model still loading...");
    return;
  }

  const input = document.getElementById("inputText").value;
  if (!input.trim()) {
    alert("Please enter a sentence!");
    return;
  }

  const seq = textToSequence(input);
  const tensor = tf.tensor2d(seq, [1, MAX_LEN]).toFloat();

  const score = model.predict(tensor).dataSync()[0];
  const sentiment = score > 0.5 ? "POSITIVE" : "NEGATIVE";

  console.log("Input:", input);
  console.log("Self Prediction:", sentiment);
  console.log("Confidence:", score.toFixed(4));

  document.getElementById("selfResult").innerHTML =
    `<b>Sentiment:</b> ${sentiment}<br>
     <b>Confidence Score:</b> ${score.toFixed(4)}`;
}

/* ======================
   OLD MODEL (RULE-BASED)
   ====================== */
function oldModelPredict(text) {
  const negativeWords = [
    "hate","hated","boring","worst","bad","awful","terrible","waste"
  ];
  const positiveWords = [
    "love","loved","amazing","good","great","awesome","fantastic"
  ];

  let score = 0;

  for (let w of negativeWords) {
    if (text.toLowerCase().includes(w)) score -= 1;
  }
  for (let w of positiveWords) {
    if (text.toLowerCase().includes(w)) score += 1;
  }

  // normalize score to confidence
  let confidence = Math.min(Math.abs(score) / 3, 1);
  return confidence;
}

/* ======================
   TASK 3: COMPARISON
   ====================== */
async function compareModels() {
  if (!model || !tokenizer) {
    alert("Model still loading...");
    return;
  }

  const input = document.getElementById("inputText").value;
  if (!input.trim()) {
    alert("Please enter a sentence first!");
    return;
  }

  // Self-trained
  const seq = textToSequence(input);
  const tensor = tf.tensor2d(seq, [1, MAX_LEN]).toFloat();
  const selfScore = model.predict(tensor).dataSync()[0];

  // Old model (rule-based)
  const oldScore = oldModelPredict(input);

  const best =
    selfScore > oldScore ? "Self-Trained Model" : "Pre-trained Model";

  console.log("Self score:", selfScore.toFixed(4));
  console.log("Old score:", oldScore.toFixed(4));
  console.log("Best model:", best);

  document.getElementById("compareResult").innerHTML =
    `<b>Self-Trained Confidence:</b> ${selfScore.toFixed(4)}<br>
     <b>Pre-trained Confidence:</b> ${oldScore.toFixed(4)}<br><br>
     <b>Best Performing Model:</b> ${best}`;
}

/* ======================
   CLEAR
   ====================== */
function clearText() {
  document.getElementById("inputText").value = "";
  document.getElementById("selfResult").innerHTML =
    "Result will appear here...";
  document.getElementById("compareResult").innerHTML =
    "Comparison result will appear here...";
}
