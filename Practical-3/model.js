let model;
let metadata;

// ---------- STRONG WORD LISTS ----------
const strongNegative = [
    "hate", "boring", "worst", "waste", "terrible",
    "awful", "pathetic", "disgust", "poor", "disappoint"
];

const strongPositive = [
    "love", "amazing", "excellent", "fantastic",
    "awesome", "brilliant", "outstanding", "superb"
];

// ---------- BASIC NLP STEMMER (Morphology Handling) ----------
function stemWord(word) {
    if (word.endsWith("ing")) return word.slice(0, -3);
    if (word.endsWith("ed")) return word.slice(0, -2);
    if (word.endsWith("ly")) return word.slice(0, -2);
    if (word.endsWith("s")) return word.slice(0, -1);
    return word;
}

// ---------- LOAD PRE-TRAINED TFJS MODEL ----------
async function loadModel() {
    console.log("Loading trained sentiment model...");

    model = await tf.loadLayersModel(
        "https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json"
    );

    metadata = await fetch(
        "https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/metadata.json"
    ).then(res => res.json());

    console.log("Model loaded successfully!");
}

// ---------- RULE-BASED + NLP LOGIC ----------
function ruleBasedSentiment(text) {
    const rawWords = text.toLowerCase().split(/\W+/);
    const words = rawWords.map(w => stemWord(w));

    for (let w of strongNegative) {
        if (words.includes(w)) {
            return { sentiment: "NEGATIVE", confidence: 0.92, reason: w };
        }
    }

    for (let w of strongPositive) {
        if (words.includes(w)) {
            return { sentiment: "POSITIVE", confidence: 0.92, reason: w };
        }
    }

    return null;
}

// ---------- TEXT PREPROCESSING FOR ML ----------
function padSequence(seq, maxLen) {
    if (seq.length > maxLen) {
        return seq.slice(seq.length - maxLen);
    }
    while (seq.length < maxLen) {
        seq.unshift(0);
    }
    return seq;
}

function textToSequence(text) {
    const tokens = text
        .toLowerCase()
        .replace(/[^a-z ]/g, "")
        .split(" ");

    const sequence = tokens.map(word => metadata.word_index[word] || 0);
    return padSequence(sequence, metadata.max_len);
}

// ---------- ML PREDICTION ----------
async function predictWithModel(sentence) {
    const sequence = textToSequence(sentence);
    const inputTensor = tf.tensor2d([sequence]);

    const prediction = model.predict(inputTensor);
    const score = (await prediction.data())[0];

    const sentiment = score >= 0.6 ? "POSITIVE" : "NEGATIVE";

    document.getElementById("sentiment").innerText = sentiment;
    document.getElementById("confidence").innerText = score.toFixed(4);
    document.getElementById("reason").innerText =
        sentiment === "NEGATIVE" ? "Model-based prediction" : "Not applicable";

    console.log("Input Text:", sentence);
    console.log("Predicted Sentiment (ML):", sentiment);
    console.log("Confidence Score:", score.toFixed(4));
    console.log("------------------------------------");
}

// ---------- BUTTON HANDLER ----------
async function handlePredict() {
    const text = document.getElementById("inputText").value.trim();

    if (!text) {
        alert("Please enter a sentence");
        return;
    }

    // STEP 1: NLP + Rule-based (High confidence)
    const ruleResult = ruleBasedSentiment(text);

    if (ruleResult) {
        document.getElementById("sentiment").innerText = ruleResult.sentiment;
        document.getElementById("confidence").innerText = ruleResult.confidence;
        document.getElementById("reason").innerText = ruleResult.reason;

        console.log("Input Text:", text);
        console.log("Predicted Sentiment (Rule + NLP):", ruleResult.sentiment);
        console.log("Confidence Score:", ruleResult.confidence);
        console.log("Reason Word:", ruleResult.reason);
        console.log("------------------------------------");
        return;
    }

    // STEP 2: ML fallback
    await predictWithModel(text);
}

// ---------- CLEAR ----------
function handleClear() {
    document.getElementById("inputText").value = "";
    document.getElementById("sentiment").innerText = "---";
    document.getElementById("confidence").innerText = "---";
    document.getElementById("reason").innerText = "---";

    console.clear();
    console.log("Cleared. Ready for new input.");
}

// Load model on page load
loadModel();
