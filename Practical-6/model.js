/**
 * Practical 6 — PoseNet Real-Time Pose Detection
 * Main practical: keypoints + skeleton + FPS + confidence
 */

// ── Canvas & Video Setup ─────────────────────────────────────────────────────
const video    = document.getElementById('video');
const canvas   = document.getElementById('canvas');
const ctx      = canvas.getContext('2d');
const overlay  = document.getElementById('loadingOverlay');

// ── UI Elements ──────────────────────────────────────────────────────────────
const statusText        = document.getElementById('statusText');
const fpsDisplay        = document.getElementById('fpsDisplay');
const confidenceDisplay = document.getElementById('confidenceDisplay');
const keypointsDisplay  = document.getElementById('keypointsDisplay');
const poseScoreDisplay  = document.getElementById('poseScoreDisplay');
const minConfidenceSlider = document.getElementById('minConfidence');
const confidenceVal     = document.getElementById('confidenceVal');
const toggleBtn         = document.getElementById('toggleBtn');
const screenshotBtn     = document.getElementById('screenshotBtn');
const kpGrid            = document.getElementById('kpGrid');

// ── State ────────────────────────────────────────────────────────────────────
let net            = null;
let isDetecting    = true;
let minConfidence  = 0.5;
let frameCount     = 0;
let lastFpsTime    = performance.now();
let currentFps     = 0;

// ── Skeleton adjacency list (PoseNet 17 keypoints) ───────────────────────────
const SKELETON_PAIRS = [
  ['nose',        'leftEye'],
  ['nose',        'rightEye'],
  ['leftEye',     'leftEar'],
  ['rightEye',    'rightEar'],
  ['leftShoulder','rightShoulder'],
  ['leftShoulder','leftElbow'],
  ['rightShoulder','rightElbow'],
  ['leftElbow',   'leftWrist'],
  ['rightElbow',  'rightWrist'],
  ['leftShoulder','leftHip'],
  ['rightShoulder','rightHip'],
  ['leftHip',     'rightHip'],
  ['leftHip',     'leftKnee'],
  ['rightHip',    'rightKnee'],
  ['leftKnee',    'leftAnkle'],
  ['rightKnee',   'rightAnkle'],
];

/** Setup webcam stream */
async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 680, height: 510, facingMode: 'user' },
    audio: false,
  });
  video.srcObject = stream;
  return new Promise(resolve => { video.onloadedmetadata = () => resolve(video); });
}

/** Resize canvas to match video */
function resizeCanvas() {
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
}

/** Draw keypoint dots on canvas */
function drawKeypoints(keypoints, minScore) {
  keypoints.forEach(kp => {
    if (kp.score < minScore) return;
    const { x, y } = kp.position;

    // Outer glow ring
    ctx.beginPath();
    ctx.arc(x, y, 9, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(37,99,235,0.18)';
    ctx.fill();

    // Main dot
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, 2 * Math.PI);
    ctx.fillStyle = kp.score > 0.75 ? '#2563eb' : kp.score > 0.5 ? '#f59e0b' : '#ef4444';
    ctx.fill();

    // White center
    ctx.beginPath();
    ctx.arc(x, y, 2, 0, 2 * Math.PI);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
  });
}

/** Draw skeleton lines between connected keypoints */
function drawSkeleton(keypoints, minScore) {
  // Build lookup map: name → keypoint
  const kpMap = {};
  keypoints.forEach(kp => { kpMap[kp.part] = kp; });

  SKELETON_PAIRS.forEach(([partA, partB]) => {
    const a = kpMap[partA];
    const b = kpMap[partB];
    if (!a || !b) return;
    if (a.score < minScore || b.score < minScore) return;

    ctx.beginPath();
    ctx.moveTo(a.position.x, a.position.y);
    ctx.lineTo(b.position.x, b.position.y);

    // Color by average confidence
    const avg = (a.score + b.score) / 2;
    ctx.strokeStyle = avg > 0.75
      ? 'rgba(37,99,235,0.85)'
      : avg > 0.5
      ? 'rgba(245,158,11,0.85)'
      : 'rgba(239,68,68,0.6)';
    ctx.lineWidth  = 2.5;
    ctx.lineCap    = 'round';
    ctx.stroke();
  });
}

/** Update the sidebar keypoints list */
function updateKeypointsList(keypoints, minScore) {
  kpGrid.innerHTML = '';
  keypoints
    .filter(kp => kp.score >= minScore)
    .sort((a, b) => b.score - a.score)
    .forEach(kp => {
      const div = document.createElement('div');
      div.className = 'kp-item';
      const scoreClass = kp.score > 0.75 ? 'high' : kp.score > 0.5 ? 'med' : 'low';
      div.innerHTML = `
        <span class="kp-name">${formatPartName(kp.part)}</span>
        <span class="kp-score ${scoreClass}">${(kp.score * 100).toFixed(0)}%</span>
      `;
      kpGrid.appendChild(div);
    });
}

/** Format camelCase part name → readable */
function formatPartName(part) {
  return part.replace(/([A-Z])/g, ' $1').replace(/^./, s => s.toUpperCase());
}

/** Calculate and update FPS */
function updateFps() {
  frameCount++;
  const now = performance.now();
  const elapsed = now - lastFpsTime;
  if (elapsed >= 1000) {
    currentFps  = Math.round((frameCount * 1000) / elapsed);
    frameCount  = 0;
    lastFpsTime = now;
    fpsDisplay.textContent = currentFps;
  }
}

/** Main detection loop */
async function detect() {
  if (!isDetecting) {
    requestAnimationFrame(detect);
    return;
  }

  const pose = await net.estimateSinglePose(video, {
    flipHorizontal: true,
    decodingMethod:  'single-person',
  });

  // Clear and redraw
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const visibleKps = pose.keypoints.filter(k => k.score >= minConfidence);

  drawSkeleton(pose.keypoints, minConfidence);
  drawKeypoints(pose.keypoints, minConfidence);
  updateKeypointsList(pose.keypoints, minConfidence);

  // Update stats
  updateFps();
  const avgConf = (visibleKps.reduce((s, k) => s + k.score, 0) / (visibleKps.length || 1));
  confidenceDisplay.textContent = (avgConf * 100).toFixed(1) + '%';
  keypointsDisplay.textContent  = visibleKps.length + ' / 17';
  poseScoreDisplay.textContent  = (pose.score * 100).toFixed(1) + '%';

  requestAnimationFrame(detect);
}

/** Entry point */
async function init() {
  statusText.textContent = 'Starting camera…';

  try {
    await setupCamera();
    video.play();
    resizeCanvas();
  } catch (e) {
    statusText.textContent = 'Camera error';
    overlay.querySelector('p').textContent = '⚠ Camera access denied. Please allow camera permission.';
    return;
  }

  statusText.textContent = 'Loading PoseNet…';

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

// ── Event Listeners ──────────────────────────────────────────────────────────

minConfidenceSlider.addEventListener('input', () => {
  minConfidence = parseFloat(minConfidenceSlider.value);
  confidenceVal.textContent = minConfidence.toFixed(2);
});

toggleBtn.addEventListener('click', () => {
  isDetecting = !isDetecting;
  toggleBtn.textContent = isDetecting ? 'Pause Detection' : 'Resume Detection';
  toggleBtn.style.background = isDetecting ? '' : '#16a34a';
  statusText.textContent = isDetecting ? '🟢 Detecting' : '⏸ Paused';
});

screenshotBtn.addEventListener('click', () => {
  const link = document.createElement('a');
  link.download = `posenet-${Date.now()}.png`;

  // Composite video + canvas
  const tmpCanvas = document.createElement('canvas');
  tmpCanvas.width  = canvas.width;
  tmpCanvas.height = canvas.height;
  const tmpCtx = tmpCanvas.getContext('2d');
  tmpCtx.drawImage(video, 0, 0, canvas.width, canvas.height);
  tmpCtx.drawImage(canvas, 0, 0);

  link.href = tmpCanvas.toDataURL('image/png');
  link.click();
});

// Start
init();
