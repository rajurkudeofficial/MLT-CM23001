const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const fpsDisplay = document.getElementById("fps");
const statusDisplay = document.getElementById("status");
const primaryDisplay = document.getElementById("primary");
const confidenceSlider = document.getElementById("confidence");
const confValue = document.getElementById("confValue");
const counterList = document.getElementById("counterList");
const snapshotDiv = document.getElementById("snapshots");

let model;
let running = false;
let lastTime = performance.now();

const objectTimers = new Map();
const objectCounts = {};
const spokenObjects = new Set();
const snappedObjects = new Set();

// Hidden canvas for REAL snapshot
const snapshotCanvas = document.createElement("canvas");
const snapshotCtx = snapshotCanvas.getContext("2d");

confidenceSlider.oninput = () => {
  confValue.textContent = confidenceSlider.value;
};

async function startWebcam() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  return new Promise(res => video.onloadedmetadata = res);
}

async function loadModel() {
  statusDisplay.textContent = "Status: Loading model...";
  model = await cocoSsd.load();
  statusDisplay.textContent = "Status: Model loaded";
}

function speak(text) {
  const msg = new SpeechSynthesisUtterance(text);
  speechSynthesis.speak(msg);
}

function takeSnapshot(label) {
  snapshotCanvas.width = video.videoWidth;
  snapshotCanvas.height = video.videoHeight;

  snapshotCtx.drawImage(video, 0, 0,
    snapshotCanvas.width,
    snapshotCanvas.height
  );

  // Optional: add label + timestamp on image
  snapshotCtx.fillStyle = "lime";
  snapshotCtx.font = "20px Arial";
  snapshotCtx.fillText(
    `${label} - ${new Date().toLocaleTimeString()}`,
    10,
    30
  );

  const img = document.createElement("img");
  img.src = snapshotCanvas.toDataURL("image/png");
  img.title = label;
  snapshotDiv.appendChild(img);
}

async function detectFrame() {
  if (!running) return;

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  const predictions = await model.detect(video);

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  let primary = null;
  let maxArea = 0;

  predictions.forEach(pred => {
    if (pred.score < confidenceSlider.value) return;

    const [x, y, w, h] = pred.bbox;
    const area = w * h;

    if (area > maxArea) {
      maxArea = area;
      primary = pred.class;
    }

    // First time detection logic
    if (!objectTimers.has(pred.class)) {
      objectTimers.set(pred.class, performance.now());

      if (!spokenObjects.has(pred.class)) {
        speak(`${pred.class} detected`);
        spokenObjects.add(pred.class);
      }

      if (!snappedObjects.has(pred.class)) {
        takeSnapshot(pred.class);
        snappedObjects.add(pred.class);
      }
    }

    objectCounts[pred.class] =
      (objectCounts[pred.class] || 0) + 1;

    const duration =
      ((performance.now() - objectTimers.get(pred.class)) / 1000)
      .toFixed(1);

    ctx.strokeStyle = "#22c55e";
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);

    ctx.fillStyle = "#22c55e";
    ctx.font = "16px Arial";
    ctx.fillText(
      `${pred.class} | ${duration}s`,
      x,
      y > 10 ? y - 5 : 10
    );

    console.log(
      `${pred.class} | ${pred.score.toFixed(2)} | ${duration}s`
    );
  });

  primaryDisplay.textContent =
    "Primary Object: " + (primary || "None");

  counterList.innerHTML = "";
  Object.keys(objectCounts).forEach(key => {
    const li = document.createElement("li");
    li.textContent =
      `${key} – ${objectCounts[key]} detections`;
    counterList.appendChild(li);
  });

  const now = performance.now();
  fpsDisplay.textContent =
    "FPS: " + Math.round(1000 / (now - lastTime));
  lastTime = now;

  requestAnimationFrame(detectFrame);
}

startBtn.onclick = async () => {
  if (!model) {
    await startWebcam();
    await loadModel();
  }
  running = true;
  statusDisplay.textContent = "Status: Detecting";
  detectFrame();
};

stopBtn.onclick = () => {
  running = false;
  statusDisplay.textContent = "Status: Stopped";
};