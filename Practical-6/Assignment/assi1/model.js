/**
 * Assignment 1 — Skeleton Visualizer
 * Color-coded joints by confidence + toggleable overlays
 */

const video   = document.getElementById('video');
const canvas  = document.getElementById('canvas');
const ctx     = canvas.getContext('2d');
const overlay = document.getElementById('loadingOverlay');

const statusText     = document.getElementById('statusText');
const fpsDisplay     = document.getElementById('fpsDisplay');
const jointsDisplay  = document.getElementById('jointsDisplay');
const minConfSlider  = document.getElementById('minConfidence');
const confidenceVal  = document.getElementById('confidenceVal');
const lineWidthSlider= document.getElementById('lineWidth');
const lineWidthVal   = document.getElementById('lineWidthVal');
const toggleBtn      = document.getElementById('toggleBtn');
const showKpsCb      = document.getElementById('showKeypoints');
const showSkCb       = document.getElementById('showSkeleton');
const showLabelsCb   = document.getElementById('showLabels');
const segmentList    = document.getElementById('segmentList');

let net           = null;
let isDetecting   = true;
let minConf       = 0.4;
let lineThickness = 3;
let frameCount    = 0;
let lastFpsTime   = performance.now();

// Body segments for sidebar breakdown
const SEGMENTS = {
  'Head':         ['nose','leftEye','rightEye','leftEar','rightEar'],
  'Torso':        ['leftShoulder','rightShoulder','leftHip','rightHip'],
  'Left Arm':     ['leftShoulder','leftElbow','leftWrist'],
  'Right Arm':    ['rightShoulder','rightElbow','rightWrist'],
  'Left Leg':     ['leftHip','leftKnee','leftAnkle'],
  'Right Leg':    ['rightHip','rightKnee','rightAnkle'],
};

const SKELETON_PAIRS = [
  ['nose','leftEye'],['nose','rightEye'],
  ['leftEye','leftEar'],['rightEye','rightEar'],
  ['leftShoulder','rightShoulder'],
  ['leftShoulder','leftElbow'],['rightShoulder','rightElbow'],
  ['leftElbow','leftWrist'],['rightElbow','rightWrist'],
  ['leftShoulder','leftHip'],['rightShoulder','rightHip'],
  ['leftHip','rightHip'],
  ['leftHip','leftKnee'],['rightHip','rightKnee'],
  ['leftKnee','leftAnkle'],['rightKnee','rightAnkle'],
];

/** Get color by confidence score */
function scoreColor(score, alpha = 1) {
  if (score > 0.75) return `rgba(22,163,74,${alpha})`;
  if (score > 0.5)  return `rgba(217,119,6,${alpha})`;
  return `rgba(220,38,38,${alpha})`;
}

/** Draw keypoints as layered circles */
function drawKeypoints(keypoints) {
  keypoints.forEach(kp => {
    if (kp.score < minConf) return;
    const { x, y } = kp.position;

    // Glow
    ctx.beginPath();
    ctx.arc(x, y, 12, 0, 2 * Math.PI);
    ctx.fillStyle = scoreColor(kp.score, 0.12);
    ctx.fill();

    // Ring
    ctx.beginPath();
    ctx.arc(x, y, 7, 0, 2 * Math.PI);
    ctx.strokeStyle = scoreColor(kp.score, 0.9);
    ctx.lineWidth = 2;
    ctx.stroke();

    // Fill
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, 2 * Math.PI);
    ctx.fillStyle = scoreColor(kp.score, 1);
    ctx.fill();

    // White dot
    ctx.beginPath();
    ctx.arc(x, y, 2.5, 0, 2 * Math.PI);
    ctx.fillStyle = '#fff';
    ctx.fill();

    // Optional label
    if (showLabelsCb.checked) {
      ctx.font = 'bold 10px DM Mono, monospace';
      ctx.fillStyle = 'rgba(255,255,255,0.85)';
      ctx.fillText(kp.part.replace(/([A-Z])/g,' $1').trim(), x + 9, y - 5);
    }
  });
}

/** Draw skeleton lines */
function drawSkeleton(keypoints) {
  const kpMap = {};
  keypoints.forEach(kp => { kpMap[kp.part] = kp; });

  SKELETON_PAIRS.forEach(([partA, partB]) => {
    const a = kpMap[partA], b = kpMap[partB];
    if (!a || !b || a.score < minConf || b.score < minConf) return;

    const avg = (a.score + b.score) / 2;

    // Shadow
    ctx.beginPath();
    ctx.moveTo(a.position.x, a.position.y);
    ctx.lineTo(b.position.x, b.position.y);
    ctx.strokeStyle = 'rgba(0,0,0,0.15)';
    ctx.lineWidth = lineThickness + 2;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Main line
    ctx.beginPath();
    ctx.moveTo(a.position.x, a.position.y);
    ctx.lineTo(b.position.x, b.position.y);
    ctx.strokeStyle = scoreColor(avg, 0.9);
    ctx.lineWidth = lineThickness;
    ctx.stroke();
  });
}

/** Update segment confidence breakdown in sidebar */
function updateSegments(keypoints) {
  const kpMap = {};
  keypoints.forEach(kp => { kpMap[kp.part] = kp; });

  segmentList.innerHTML = '';
  Object.entries(SEGMENTS).forEach(([segName, parts]) => {
    const scores = parts.map(p => kpMap[p]?.score ?? 0);
    const avg = scores.reduce((a,b) => a+b, 0) / scores.length;

    const div = document.createElement('div');
    div.className = 'seg-item';
    div.innerHTML = `
      <span class="seg-name">${segName}</span>
      <div class="seg-bar-wrap"><div class="seg-bar" style="width:${(avg*100).toFixed(0)}%;background:${scoreColor(avg)}"></div></div>
      <span class="seg-score">${(avg*100).toFixed(0)}%</span>
    `;
    segmentList.appendChild(div);
  });
}

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480, facingMode: 'user' }, audio: false,
  });
  video.srcObject = stream;
  return new Promise(resolve => { video.onloadedmetadata = () => resolve(video); });
}

let lastFps = 0;
function updateFps() {
  frameCount++;
  const now = performance.now();
  if (now - lastFpsTime >= 1000) {
    lastFps = Math.round((frameCount * 1000) / (now - lastFpsTime));
    frameCount = 0;
    lastFpsTime = now;
    fpsDisplay.textContent = lastFps;
  }
}

async function detect() {
  if (!isDetecting) { requestAnimationFrame(detect); return; }

  const pose = await net.estimateSinglePose(video, { flipHorizontal: true });
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (showSkCb.checked) drawSkeleton(pose.keypoints);
  if (showKpsCb.checked) drawKeypoints(pose.keypoints);

  const visible = pose.keypoints.filter(k => k.score >= minConf);
  jointsDisplay.textContent = visible.length + '/17';
  updateSegments(pose.keypoints);
  updateFps();

  requestAnimationFrame(detect);
}

async function init() {
  statusText.textContent = 'Starting camera…';
  try {
    await setupCamera();
    video.play();
    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;
  } catch (e) {
    statusText.textContent = 'Camera error';
    overlay.querySelector('p').textContent = '⚠ Camera access denied.';
    return;
  }

  statusText.textContent = 'Loading model…';
  net = await posenet.load({
    architecture: 'MobileNetV1',
    outputStride: 16,
    inputResolution: { width: 640, height: 480 },
    multiplier: 0.75,
  });

  statusText.textContent = '🟢 Detecting';
  overlay.classList.add('hidden');
  detect();
}

// Event listeners
minConfSlider.addEventListener('input', () => {
  minConf = parseFloat(minConfSlider.value);
  confidenceVal.textContent = minConf.toFixed(2);
});

lineWidthSlider.addEventListener('input', () => {
  lineThickness = parseFloat(lineWidthSlider.value);
  lineWidthVal.textContent = lineThickness;
});

toggleBtn.addEventListener('click', () => {
  isDetecting = !isDetecting;
  toggleBtn.textContent = isDetecting ? 'Pause' : 'Resume';
  statusText.textContent = isDetecting ? '🟢 Detecting' : '⏸ Paused';
});

init();
