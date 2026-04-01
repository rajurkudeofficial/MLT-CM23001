/**
 * Assignment 3 — Single vs Multi-Pose Detection
 * Toggle between modes, show per-person colors, counts & confidence
 */

const video   = document.getElementById('video');
const canvas  = document.getElementById('canvas');
const ctx     = canvas.getContext('2d');
const overlay = document.getElementById('loadingOverlay');

const modeText         = document.getElementById('modeText');
const modeBadge        = document.getElementById('modeBadge');
const modeToggleBtn    = document.getElementById('modeToggleBtn');
const personCount      = document.getElementById('personCount');
const personCountBadge = document.getElementById('personCountBadge');
const fpsDisplay       = document.getElementById('fpsDisplay');
const avgConfDisplay   = document.getElementById('avgConfDisplay');
const bestPoseDisplay  = document.getElementById('bestPoseDisplay');
const minConfSlider    = document.getElementById('minConfidence');
const confidenceVal    = document.getElementById('confidenceVal');
const maxPosesSlider   = document.getElementById('maxPoses');
const maxPosesVal      = document.getElementById('maxPosesVal');
const nmsRadiusSlider  = document.getElementById('nmsRadius');
const nmsRadiusVal     = document.getElementById('nmsRadiusVal');
const multiSettings    = document.getElementById('multiSettings');
const personsList      = document.getElementById('personsList');

// ── Config ────────────────────────────────────────────────────────────────────
let net         = null;
let isMulti     = false;   // detection mode
let minConf     = 0.4;
let maxPoses    = 3;
let nmsRadius   = 20;
let frameCount  = 0;
let lastFpsTime = performance.now();

// Per-person color palette
const PERSON_COLORS = [
  '#2563eb', '#ea580c', '#16a34a',
  '#7c3aed', '#db2777', '#0891b2',
];

// Skeleton pairs
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

/** Draw a single pose's skeleton with given color */
function drawSkeleton(keypoints, color) {
  const kpMap = {};
  keypoints.forEach(kp => { kpMap[kp.part] = kp; });

  SKELETON_PAIRS.forEach(([partA, partB]) => {
    const a = kpMap[partA], b = kpMap[partB];
    if (!a || !b || a.score < minConf || b.score < minConf) return;

    ctx.beginPath();
    ctx.moveTo(a.position.x, a.position.y);
    ctx.lineTo(b.position.x, b.position.y);
    ctx.strokeStyle = color.replace(')', ', 0.8)').replace('rgb', 'rgba');
    ctx.lineWidth = 2.5;
    ctx.lineCap = 'round';
    ctx.stroke();
  });
}

/** Draw keypoints for a single pose */
function drawKeypoints(keypoints, color) {
  keypoints.forEach(kp => {
    if (kp.score < minConf) return;
    const { x, y } = kp.position;

    ctx.beginPath();
    ctx.arc(x, y, 9, 0, 2 * Math.PI);
    ctx.fillStyle = hexToRgba(color, 0.15);
    ctx.fill();

    ctx.beginPath();
    ctx.arc(x, y, 5, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();

    ctx.beginPath();
    ctx.arc(x, y, 2, 0, 2 * Math.PI);
    ctx.fillStyle = '#fff';
    ctx.fill();
  });
}

/** Draw person label badge */
function drawPersonLabel(keypoints, color, index) {
  // Find highest keypoint for label placement
  const visible = keypoints.filter(k => k.score >= minConf);
  if (!visible.length) return;

  const topY = Math.min(...visible.map(k => k.position.y));
  const midX = (Math.min(...visible.map(k => k.position.x)) +
                Math.max(...visible.map(k => k.position.x))) / 2;

  const label = `P${index + 1}`;
  const tw = ctx.measureText(label).width;

  ctx.fillStyle = color;
  ctx.beginPath();
  roundRect(ctx, midX - tw/2 - 8, topY - 28, tw + 16, 22, 6);
  ctx.fill();

  ctx.font = 'bold 12px DM Mono, monospace';
  ctx.fillStyle = '#fff';
  ctx.textAlign = 'center';
  ctx.fillText(label, midX, topY - 12);
  ctx.textAlign = 'left';
}

/** Utility: round rect path */
function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.arcTo(x + w, y, x + w, y + r, r);
  ctx.lineTo(x + w, y + h - r);
  ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
  ctx.lineTo(x + r, y + h);
  ctx.arcTo(x, y + h, x, y + h - r, r);
  ctx.lineTo(x, y + r);
  ctx.arcTo(x, y, x + r, y, r);
  ctx.closePath();
}

/** Convert hex color to rgba */
function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1,3),16);
  const g = parseInt(hex.slice(3,5),16);
  const b = parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${alpha})`;
}

/** Render detected persons in the sidebar */
function renderPersonsList(poses) {
  personsList.innerHTML = '';
  if (!poses.length) {
    personsList.innerHTML = '<p class="empty-msg">No poses detected.</p>';
    return;
  }

  poses.forEach((pose, i) => {
    const color = PERSON_COLORS[i % PERSON_COLORS.length];
    const visibleKps = pose.keypoints.filter(k => k.score >= minConf);
    const avgConf = visibleKps.length
      ? (visibleKps.reduce((s, k) => s + k.score, 0) / visibleKps.length * 100).toFixed(1)
      : '0';

    const div = document.createElement('div');
    div.className = 'person-item';
    div.innerHTML = `
      <span class="person-color" style="background:${color}"></span>
      <span class="person-name">Person ${i + 1}</span>
      <span class="person-score">${avgConf}%</span>
      <span class="person-kp">${visibleKps.length}/17</span>
    `;
    personsList.appendChild(div);
  });
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

/** Main detection loop — switches between single & multi pose */
async function detect() {
  let poses = [];

  if (isMulti) {
    // Multi-pose estimation
    poses = await net.estimateMultiplePoses(video, {
      flipHorizontal:        true,
      maxDetections:         maxPoses,
      scoreThreshold:        minConf,
      nmsRadius:             nmsRadius,
    });
  } else {
    // Single-pose estimation
    const single = await net.estimateSinglePose(video, { flipHorizontal: true });
    poses = [single];
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw each pose with unique color
  poses.forEach((pose, i) => {
    if (pose.score < minConf && isMulti) return;
    const color = PERSON_COLORS[i % PERSON_COLORS.length];
    drawSkeleton(pose.keypoints, color);
    drawKeypoints(pose.keypoints, color);
    if (isMulti) drawPersonLabel(pose.keypoints, color, i);
  });

  // Filter to visible poses
  const visiblePoses = isMulti
    ? poses.filter(p => p.score >= minConf)
    : poses;

  // Update stats
  const count = visiblePoses.length;
  personCount.textContent = count;
  personCountBadge.textContent = count + (count === 1 ? ' person' : ' persons');

  const allConf = visiblePoses.flatMap(p =>
    p.keypoints.filter(k => k.score >= minConf).map(k => k.score)
  );
  const avgConf = allConf.length
    ? (allConf.reduce((s, v) => s + v, 0) / allConf.length * 100).toFixed(1)
    : '0';
  avgConfDisplay.textContent = avgConf + '%';

  const bestScore = Math.max(...visiblePoses.map(p => p.score), 0);
  bestPoseDisplay.textContent = (bestScore * 100).toFixed(1) + '%';

  renderPersonsList(visiblePoses);
  updateFps();

  requestAnimationFrame(detect);
}

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480, facingMode: 'user' }, audio: false,
  });
  video.srcObject = stream;
  return new Promise(resolve => { video.onloadedmetadata = () => resolve(video); });
}

async function init() {
  try {
    await setupCamera();
    video.play();
    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;
  } catch (e) {
    overlay.querySelector('p').textContent = '⚠ Camera access denied.';
    return;
  }

  overlay.querySelector('p').textContent = 'Loading PoseNet…';
  net = await posenet.load({
    architecture: 'MobileNetV1',
    outputStride: 16,
    inputResolution: { width: 640, height: 480 },
    multiplier: 0.75,
  });

  overlay.classList.add('hidden');
  multiSettings.classList.add('hidden');
  detect();
}

// ── Mode toggle ───────────────────────────────────────────────────────────────
modeToggleBtn.addEventListener('click', () => {
  isMulti = !isMulti;

  if (isMulti) {
    modeText.textContent      = 'Multi-Pose';
    modeBadge.textContent     = 'MULTI';
    modeBadge.classList.add('multi');
    modeToggleBtn.textContent = '⇄ Switch to Single-Pose';
    modeToggleBtn.classList.add('active');
    multiSettings.classList.remove('hidden');
  } else {
    modeText.textContent      = 'Single Pose';
    modeBadge.textContent     = 'SINGLE';
    modeBadge.classList.remove('multi');
    modeToggleBtn.innerHTML   = '<span class="btn-icon">⇄</span> Switch to Multi-Pose';
    modeToggleBtn.classList.remove('active');
    multiSettings.classList.add('hidden');
  }
});

// Sliders
minConfSlider.addEventListener('input', () => {
  minConf = parseFloat(minConfSlider.value);
  confidenceVal.textContent = minConf.toFixed(2);
});
maxPosesSlider.addEventListener('input', () => {
  maxPoses = parseInt(maxPosesSlider.value);
  maxPosesVal.textContent = maxPoses;
});
nmsRadiusSlider.addEventListener('input', () => {
  nmsRadius = parseInt(nmsRadiusSlider.value);
  nmsRadiusVal.textContent = nmsRadius;
});

init();
