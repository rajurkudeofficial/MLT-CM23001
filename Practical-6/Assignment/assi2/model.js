/**
 * Assignment 2 — Squat Counter
 * Uses knee angle (hip-knee-ankle) to detect squat reps
 */

const video   = document.getElementById('video');
const canvas  = document.getElementById('canvas');
const ctx     = canvas.getContext('2d');
const overlay = document.getElementById('loadingOverlay');

const repCountEl    = document.getElementById('repCount');
const postureText   = document.getElementById('postureText');
const postureBadge  = document.getElementById('postureBadge');
const postureCard   = document.getElementById('postureCard');
const leftAngleEl   = document.getElementById('leftAngle');
const rightAngleEl  = document.getElementById('rightAngle');
const fpsDisplay    = document.getElementById('fpsDisplay');
const statusText    = document.getElementById('statusText');
const downThSlider  = document.getElementById('downThreshold');
const upThSlider    = document.getElementById('upThreshold');
const downValEl     = document.getElementById('downVal');
const upValEl       = document.getElementById('upVal');
const resetBtn      = document.getElementById('resetBtn');
const historyList   = document.getElementById('historyList');

// ── State ─────────────────────────────────────────────────────────────────────
let net          = null;
let repCount     = 0;
let squat        = 'UP';   // current posture phase
let downThreshold = 90;    // angle to trigger DOWN
let upThreshold   = 160;   // angle to trigger UP (complete rep)
let frameCount    = 0;
let lastFpsTime   = performance.now();
let historyData   = [];

// ── Skeleton for display ──────────────────────────────────────────────────────
const SKELETON_PAIRS = [
  ['leftShoulder','rightShoulder'],
  ['leftShoulder','leftElbow'],['rightShoulder','rightElbow'],
  ['leftElbow','leftWrist'],['rightElbow','rightWrist'],
  ['leftShoulder','leftHip'],['rightShoulder','rightHip'],
  ['leftHip','rightHip'],
  ['leftHip','leftKnee'],['rightHip','rightKnee'],
  ['leftKnee','leftAnkle'],['rightKnee','rightAnkle'],
];

/**
 * Calculate the angle (degrees) at joint B, between vectors A→B and B→C
 * @param {{x,y}} A - first point
 * @param {{x,y}} B - vertex (joint)
 * @param {{x,y}} C - end point
 * @returns {number} angle in degrees
 */
function calculateAngle(A, B, C) {
  const radians = Math.atan2(C.y - B.y, C.x - B.x)
                - Math.atan2(A.y - B.y, A.x - B.x);
  let angle = Math.abs(radians * (180 / Math.PI));
  if (angle > 180) angle = 360 - angle;
  return angle;
}

/**
 * Count squat reps based on angle transitions
 * DOWN → angle < downThreshold
 * UP   → angle > upThreshold (1 rep completed)
 */
function countReps(avgAngle) {
  if (avgAngle < downThreshold && squat === 'UP') {
    squat = 'DOWN';
    updatePosture('DOWN');
  } else if (avgAngle > upThreshold && squat === 'DOWN') {
    squat = 'UP';
    repCount++;
    repCountEl.textContent = repCount;
    updatePosture('UP');
    addHistory(repCount, avgAngle);

    // Pulse animation on counter
    repCountEl.style.transform = 'scale(1.2)';
    setTimeout(() => { repCountEl.style.transform = 'scale(1)'; }, 200);
  }
}

/** Update posture UI */
function updatePosture(state) {
  postureText.textContent = state;
  postureText.className   = 'posture-val ' + state.toLowerCase();
  postureBadge.textContent = state;
  postureBadge.className   = 'posture-badge ' + state.toLowerCase();
}

/** Add entry to rep log */
function addHistory(rep, angle) {
  const now = new Date().toLocaleTimeString();
  historyData.unshift({ rep, angle: angle.toFixed(0), time: now });

  // Remove placeholder
  const empty = historyList.querySelector('.history-empty');
  if (empty) empty.remove();

  const item = document.createElement('div');
  item.className = 'history-item';
  item.innerHTML = `
    <span class="h-rep">Rep ${rep}</span>
    <span class="h-angle">~${angle.toFixed(0)}°</span>
    <span class="h-time">${now}</span>
  `;
  historyList.prepend(item);

  // Keep max 20 entries
  const items = historyList.querySelectorAll('.history-item');
  if (items.length > 20) items[items.length - 1].remove();
}

/** Draw skeleton lines */
function drawSkeleton(keypoints, minScore) {
  const kpMap = {};
  keypoints.forEach(kp => { kpMap[kp.part] = kp; });

  SKELETON_PAIRS.forEach(([partA, partB]) => {
    const a = kpMap[partA], b = kpMap[partB];
    if (!a || !b || a.score < minScore || b.score < minScore) return;

    ctx.beginPath();
    ctx.moveTo(a.position.x, a.position.y);
    ctx.lineTo(b.position.x, b.position.y);
    ctx.strokeStyle = 'rgba(22,163,74,0.75)';
    ctx.lineWidth = 2.5;
    ctx.lineCap = 'round';
    ctx.stroke();
  });
}

/** Draw keypoints */
function drawKeypoints(keypoints, minScore) {
  keypoints.forEach(kp => {
    if (kp.score < minScore) return;
    const { x, y } = kp.position;

    ctx.beginPath();
    ctx.arc(x, y, 6, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(22,163,74,0.2)';
    ctx.fill();

    ctx.beginPath();
    ctx.arc(x, y, 4, 0, 2 * Math.PI);
    ctx.fillStyle = '#16a34a';
    ctx.fill();

    ctx.beginPath();
    ctx.arc(x, y, 1.8, 0, 2 * Math.PI);
    ctx.fillStyle = '#fff';
    ctx.fill();
  });
}

/** Draw angle arc at a joint */
function drawAngleArc(B, angle, color) {
  ctx.beginPath();
  ctx.arc(B.x, B.y, 28, 0, (angle / 180) * Math.PI);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.font = 'bold 13px DM Mono, monospace';
  ctx.fillStyle = color;
  ctx.fillText(angle.toFixed(0) + '°', B.x + 32, B.y + 5);
}

/** Highlight knee joints */
function drawKneeHighlight(kp, color) {
  const { x, y } = kp.position;
  ctx.beginPath();
  ctx.arc(x, y, 14, 0, 2 * Math.PI);
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.stroke();
}

function updateFps() {
  frameCount++;
  const now = performance.now();
  if (now - lastFpsTime >= 1000) {
    fpsDisplay.textContent = Math.round((frameCount * 1000) / (now - lastFpsTime));
    frameCount = 0;
    lastFpsTime = now;
  }
}

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480, facingMode: 'user' }, audio: false,
  });
  video.srcObject = stream;
  return new Promise(resolve => { video.onloadedmetadata = () => resolve(video); });
}

async function detect() {
  const MIN_CONF = 0.4;

  const pose = await net.estimateSinglePose(video, { flipHorizontal: true });
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  drawSkeleton(pose.keypoints, MIN_CONF);
  drawKeypoints(pose.keypoints, MIN_CONF);

  const kpMap = {};
  pose.keypoints.forEach(kp => { kpMap[kp.part] = kp; });

  // Left knee angle: leftHip → leftKnee → leftAnkle
  const lh = kpMap['leftHip'], lk = kpMap['leftKnee'], la = kpMap['leftAnkle'];
  // Right knee angle: rightHip → rightKnee → rightAnkle
  const rh = kpMap['rightHip'], rk = kpMap['rightKnee'], ra = kpMap['rightAnkle'];

  let leftAngle  = null;
  let rightAngle = null;

  if (lh && lk && la && lh.score > MIN_CONF && lk.score > MIN_CONF && la.score > MIN_CONF) {
    leftAngle = calculateAngle(lh.position, lk.position, la.position);
    leftAngleEl.textContent = leftAngle.toFixed(1) + '°';
    drawKneeHighlight(lk, squat === 'DOWN' ? '#dc2626' : '#16a34a');
    drawAngleArc(lk.position, leftAngle, squat === 'DOWN' ? '#dc2626' : '#16a34a');
  }

  if (rh && rk && ra && rh.score > MIN_CONF && rk.score > MIN_CONF && ra.score > MIN_CONF) {
    rightAngle = calculateAngle(rh.position, rk.position, ra.position);
    rightAngleEl.textContent = rightAngle.toFixed(1) + '°';
    drawKneeHighlight(rk, squat === 'DOWN' ? '#dc2626' : '#16a34a');
    drawAngleArc(rk.position, rightAngle, squat === 'DOWN' ? '#dc2626' : '#16a34a');
  }

  // Average available angles for rep counting
  const angles = [leftAngle, rightAngle].filter(a => a !== null);
  if (angles.length > 0) {
    const avg = angles.reduce((s, a) => s + a, 0) / angles.length;
    countReps(avg);
  }

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
    return;
  }

  statusText.textContent = 'Loading model…';
  net = await posenet.load({
    architecture: 'MobileNetV1',
    outputStride: 16,
    inputResolution: { width: 640, height: 480 },
    multiplier: 0.75,
  });

  statusText.textContent = '🟢 Active';
  overlay.classList.add('hidden');
  detect();
}

// Sliders
downThSlider.addEventListener('input', () => {
  downThreshold = parseInt(downThSlider.value);
  downValEl.textContent = downThreshold + '°';
});
upThSlider.addEventListener('input', () => {
  upThreshold = parseInt(upThSlider.value);
  upValEl.textContent = upThreshold + '°';
});

resetBtn.addEventListener('click', () => {
  repCount = 0;
  squat = 'UP';
  historyData = [];
  repCountEl.textContent = '0';
  historyList.innerHTML = '<p class="history-empty">No reps recorded yet.</p>';
  updatePosture('UP');
});

init();
