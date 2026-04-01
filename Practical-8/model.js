/**
 * NeuralLens — Transfer Learning Studio
 * model.js — Core ML + UI logic
 * Uses MobileNet as feature extractor + custom dense head
 */

// ============================================================
// STATE
// ============================================================
const STATE = {
  mobilenet: null,
  featureModel: null,    // truncated mobilenet for feature extraction
  classifier: null,      // custom head
  classes: [],           // [{id, name, color, images: [{tensor, src}]}]
  isTraining: false,
  stopRequested: false,
  trainedModel: null,
  bestAcc: 0,
  chartData: { loss: [], acc: [] },
  webcamStream: null,
  predWebcamStream: null,
  predLiveInterval: null,
};

const CLASS_COLORS = [
  '#e8652a','#2a9de8','#27ae60','#9b59b6',
  '#e74c3c','#f39c12','#1abc9c','#e91e63',
];

// ============================================================
// UTILS
// ============================================================
function uid() { return Math.random().toString(36).slice(2, 8); }
function toast(msg, type = '') {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  document.getElementById('toast-container').appendChild(el);
  setTimeout(() => el.remove(), 3000);
}
function log(msg, type = '') {
  const box = document.getElementById('training-log');
  if (!box) return;
  const line = document.createElement('div');
  line.className = `log-line ${type}`;
  line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  box.appendChild(line);
  box.scrollTop = box.scrollHeight;
}
function showLoading(btn, text = 'Loading...') {
  btn._origText = btn.textContent;
  btn.textContent = text;
  btn.disabled = true;
}
function hideLoading(btn) {
  btn.textContent = btn._origText;
  btn.disabled = false;
}

// ============================================================
// MODEL LOADING
// ============================================================
async function loadMobileNet() {
  if (STATE.mobilenet) return STATE.mobilenet;
  log('Loading MobileNet from CDN...');
  const btn = document.getElementById('btn-train');
  showLoading(btn, 'Loading MobileNet...');
  try {
    STATE.mobilenet = await mobilenet.load({ version: 2, alpha: 0.5 });
    // Create truncated model for feature extraction
    const layer = STATE.mobilenet.model.getLayer('conv_pw_13_relu');
    STATE.featureModel = tf.model({
      inputs: STATE.mobilenet.model.inputs,
      outputs: layer.output,
    });
    document.getElementById('info-status').textContent = 'MobileNet Ready';
    log('MobileNet loaded successfully ✓', 'success');
    toast('MobileNet loaded!', 'success');
  } catch (e) {
    log('Failed to load MobileNet: ' + e.message, 'warning');
    toast('MobileNet load failed', 'error');
  }
  hideLoading(btn);
  return STATE.mobilenet;
}

// ============================================================
// CLASS MANAGEMENT
// ============================================================
function addClass(name, color) {
  if (!name.trim()) { toast('Class name required', 'error'); return; }
  if (STATE.classes.find(c => c.name.toLowerCase() === name.toLowerCase())) {
    toast('Class already exists', 'error'); return;
  }
  const cls = {
    id: uid(),
    name: name.trim(),
    color: color || CLASS_COLORS[STATE.classes.length % CLASS_COLORS.length],
    images: [],
  };
  STATE.classes.push(cls);
  renderClasses();
  updateClassSelectors();
  updateSummary();
  toast(`Class "${name}" created`, 'success');
}

function removeClass(id) {
  const idx = STATE.classes.findIndex(c => c.id === id);
  if (idx === -1) return;
  // Dispose tensors
  STATE.classes[idx].images.forEach(img => img.tensor && img.tensor.dispose());
  STATE.classes.splice(idx, 1);
  renderClasses();
  updateClassSelectors();
  updateSummary();
  renderPreviewGrid();
}

function addImageToClass(classId, imgSrc) {
  const cls = STATE.classes.find(c => c.id === classId);
  if (!cls) { toast('Select a class first', 'error'); return; }
  cls.images.push({ src: imgSrc, tensor: null });
  renderClasses();
  updateSummary();
  renderPreviewGrid();
}

function renderClasses() {
  const list = document.getElementById('class-list');
  const active = document.getElementById('active-class-select').value;
  list.innerHTML = '';
  STATE.classes.forEach(cls => {
    const card = document.createElement('div');
    card.className = `class-card ${cls.id === active ? 'active' : ''}`;
    card.style.setProperty('--class-color', cls.color);
    card.innerHTML = `
      <div class="class-card-header">
        <div class="class-name">
          <div class="class-dot"></div>
          ${cls.name}
        </div>
        <span class="class-count">${cls.images.length} img</span>
      </div>
      <div class="class-actions">
        <button class="class-btn" onclick="setActiveClass('${cls.id}')">Select</button>
        <button class="class-btn danger" onclick="removeClass('${cls.id}')">Delete</button>
      </div>`;
    list.appendChild(card);
  });
  document.getElementById('info-classes').textContent = STATE.classes.length;
}

function setActiveClass(id) {
  document.getElementById('active-class-select').value = id;
  renderClasses();
  renderPreviewGrid();
}

function updateClassSelectors() {
  const sel = document.getElementById('active-class-select');
  const prev = sel.value;
  sel.innerHTML = '<option value="">— Select Class —</option>';
  STATE.classes.forEach(cls => {
    const opt = document.createElement('option');
    opt.value = cls.id; opt.textContent = cls.name;
    sel.appendChild(opt);
  });
  if (prev && STATE.classes.find(c => c.id === prev)) sel.value = prev;
}

function updateSummary() {
  document.getElementById('total-classes').textContent = STATE.classes.length;
  const total = STATE.classes.reduce((s, c) => s + c.images.length, 0);
  document.getElementById('total-images').textContent = total;
}

// ============================================================
// PREVIEW GRID
// ============================================================
function renderPreviewGrid() {
  const grid = document.getElementById('preview-grid');
  const activeId = document.getElementById('active-class-select').value;
  grid.innerHTML = '';
  const toShow = activeId
    ? STATE.classes.filter(c => c.id === activeId)
    : STATE.classes;
  toShow.forEach(cls => {
    cls.images.forEach((img, idx) => {
      const thumb = document.createElement('div');
      thumb.className = 'preview-thumb';
      thumb.innerHTML = `
        <img src="${img.src}" loading="lazy"/>
        <div class="thumb-class" style="background:${cls.color}88">${cls.name}</div>
        <button class="thumb-del" onclick="removeImage('${cls.id}',${idx})">✕</button>`;
      grid.appendChild(thumb);
    });
  });
}

function removeImage(classId, idx) {
  const cls = STATE.classes.find(c => c.id === classId);
  if (!cls) return;
  if (cls.images[idx].tensor) cls.images[idx].tensor.dispose();
  cls.images.splice(idx, 1);
  updateSummary();
  renderClasses();
  renderPreviewGrid();
}

// ============================================================
// WEBCAM
// ============================================================
async function startWebcam(videoEl) {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 224, height: 224 } });
    videoEl.srcObject = stream;
    await new Promise(r => videoEl.onloadeddata = r);
    return stream;
  } catch (e) {
    toast('Webcam access denied', 'error'); return null;
  }
}

function captureFrame(videoEl) {
  const canvas = document.getElementById('snapshot-canvas') || document.createElement('canvas');
  canvas.width = 224; canvas.height = 224;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(videoEl, 0, 0, 224, 224);
  return canvas.toDataURL('image/jpeg', 0.8);
}

// ============================================================
// IMAGE → TENSOR
// ============================================================
function imageToTensor(imgSrc) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const tensor = tf.tidy(() => {
        return tf.browser.fromPixels(img)
          .resizeBilinear([224, 224])
          .toFloat()
          .div(127.5).sub(1)
          .expandDims(0);
      });
      resolve(tensor);
    };
    img.src = imgSrc;
  });
}

async function extractFeatures(imgSrc) {
  if (!STATE.featureModel) await loadMobileNet();
  const tensor = await imageToTensor(imgSrc);
  const features = tf.tidy(() => STATE.featureModel.predict(tensor).squeeze());
  tensor.dispose();
  return features;
}

// ============================================================
// TRAINING
// ============================================================
async function trainModel() {
  if (STATE.isTraining) return;
  if (STATE.classes.length < 2) { toast('Add at least 2 classes', 'error'); return; }
  for (const cls of STATE.classes) {
    if (cls.images.length < 1) { toast(`No images in class "${cls.name}"`, 'error'); return; }
  }

  await loadMobileNet();
  STATE.isTraining = true;
  STATE.stopRequested = false;

  const epochs = parseInt(document.getElementById('epochs').value) || 20;
  const batchSize = parseInt(document.getElementById('batch-size').value) || 16;
  const lr = parseFloat(document.getElementById('learning-rate').value) || 0.0001;
  const numClasses = STATE.classes.length;

  document.getElementById('training-progress').style.display = 'block';
  document.getElementById('btn-train').disabled = true;
  document.getElementById('btn-stop').disabled = false;
  STATE.chartData = { loss: [], acc: [] };
  document.getElementById('training-log').innerHTML = '';
  log(`Starting training: ${numClasses} classes, ${epochs} epochs, lr=${lr}`);

  // Extract features for all images
  log('Extracting features...');
  const xs = [], ys = [];
  for (let ci = 0; ci < STATE.classes.length; ci++) {
    const cls = STATE.classes[ci];
    log(`Processing class "${cls.name}" (${cls.images.length} images)...`);
    for (const img of cls.images) {
      const feat = await extractFeatures(img.src);
      xs.push(feat);
      ys.push(ci);
    }
  }

  const xTensor = tf.stack(xs);
  const yTensor = tf.oneHot(tf.tensor1d(ys, 'int32'), numClasses);
  xs.forEach(t => t.dispose());

  // Get feature shape
  const featShape = xTensor.shape.slice(1);
  const flatSize = featShape.reduce((a, b) => a * b, 1);
  const xFlat = xTensor.reshape([-1, flatSize]);
  xTensor.dispose();

  // Build classifier head
  if (STATE.classifier) { STATE.classifier.dispose(); }
  STATE.classifier = tf.sequential();
  STATE.classifier.add(tf.layers.dense({
    units: 256, activation: 'relu', inputShape: [flatSize],
    kernelInitializer: 'glorotNormal',
  }));
  STATE.classifier.add(tf.layers.dropout({ rate: 0.3 }));
  STATE.classifier.add(tf.layers.dense({
    units: 128, activation: 'relu',
    kernelInitializer: 'glorotNormal',
  }));
  STATE.classifier.add(tf.layers.dense({
    units: numClasses, activation: 'softmax',
  }));

  STATE.classifier.compile({
    optimizer: tf.train.adam(lr),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  log(`Model built. Training on ${xFlat.shape[0]} samples...`, 'success');

  const chartCanvas = document.getElementById('training-chart');
  const chartCtx = chartCanvas.getContext('2d');

  await STATE.classifier.fit(xFlat, yTensor, {
    epochs,
    batchSize,
    shuffle: true,
    validationSplit: 0.15,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        if (STATE.stopRequested) {
          STATE.classifier.stopTraining = true;
          return;
        }
        const pct = Math.round(((epoch + 1) / epochs) * 100);
        document.getElementById('progress-bar').style.width = pct + '%';
        document.getElementById('progress-label').textContent =
          `Epoch ${epoch + 1}/${epochs}`;
        document.getElementById('live-loss').textContent = logs.loss.toFixed(4);
        document.getElementById('live-acc').textContent =
          (logs.acc * 100).toFixed(1) + '%';

        STATE.chartData.loss.push(logs.loss);
        STATE.chartData.acc.push(logs.acc);
        drawChart(chartCtx, chartCanvas.width, chartCanvas.height);

        if (logs.acc > STATE.bestAcc) {
          STATE.bestAcc = logs.acc;
          document.getElementById('info-acc').textContent =
            (STATE.bestAcc * 100).toFixed(1) + '%';
        }

        log(`Epoch ${epoch+1}/${epochs} — Loss: ${logs.loss.toFixed(4)}, Acc: ${(logs.acc*100).toFixed(1)}%`);
        await tf.nextFrame();
      },
    },
  });

  xFlat.dispose(); yTensor.dispose();

  STATE.isTraining = false;
  STATE.trainedModel = STATE.classifier;
  document.getElementById('btn-train').disabled = false;
  document.getElementById('btn-stop').disabled = true;
  document.getElementById('model-status-badge').textContent = 'Trained';
  document.getElementById('info-status').textContent = 'Trained';
  log('Training complete!', 'success');
  toast('Training complete! 🎉', 'success');
}

// ============================================================
// CHART
// ============================================================
function drawChart(ctx, w, h) {
  ctx.clearRect(0, 0, w, h);
  const pd = 32;
  const cw = w - pd * 2, ch = h - pd * 2;

  // Background
  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--bg').trim() || '#f0efea';
  ctx.fillRect(0, 0, w, h);

  const loss = STATE.chartData.loss;
  const acc = STATE.chartData.acc;
  if (loss.length < 2) return;

  function drawLine(data, color, maxVal) {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    data.forEach((v, i) => {
      const x = pd + (i / (data.length - 1)) * cw;
      const y = pd + ch - (v / maxVal) * ch;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  // Grid
  ctx.strokeStyle = '#e0ddd6'; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pd + (i / 4) * ch;
    ctx.beginPath(); ctx.moveTo(pd, y); ctx.lineTo(pd + cw, y); ctx.stroke();
  }

  const maxLoss = Math.max(...loss, 0.1);
  drawLine(loss, '#e8652a', maxLoss);
  drawLine(acc, '#2a9de8', 1);

  // Labels
  ctx.fillStyle = '#6b6760'; ctx.font = '10px DM Sans';
  ctx.fillText('Loss', pd, pd - 8);
  ctx.fillStyle = '#2a9de8';
  ctx.fillText('Acc', pd + 40, pd - 8);
}

// ============================================================
// PREDICTION
// ============================================================
async function predict(imgSrc) {
  if (!STATE.trainedModel) { toast('Train a model first', 'error'); return; }
  if (STATE.classes.length === 0) return;

  const feat = await extractFeatures(imgSrc);
  const flatSize = feat.shape.reduce((a, b) => a * b, 1);
  const flat = feat.reshape([1, flatSize]);
  const preds = STATE.trainedModel.predict(flat);
  const data = await preds.data();
  flat.dispose(); feat.dispose(); preds.dispose();

  const results = STATE.classes.map((cls, i) => ({ name: cls.name, color: cls.color, conf: data[i] }));
  results.sort((a, b) => b.conf - a.conf);

  document.getElementById('pred-top-label').textContent = results[0].name;
  document.getElementById('pred-top-conf').textContent =
    `Confidence: ${(results[0].conf * 100).toFixed(1)}%`;

  const barsEl = document.getElementById('confidence-bars');
  barsEl.innerHTML = '';
  results.forEach(r => {
    const row = document.createElement('div');
    row.className = 'conf-bar-row';
    row.innerHTML = `
      <div class="conf-bar-label">${r.name}</div>
      <div class="conf-bar-track">
        <div class="conf-bar-fill" style="width:${(r.conf*100).toFixed(1)}%;background:${r.color}"></div>
      </div>
      <div class="conf-bar-val">${(r.conf*100).toFixed(0)}%</div>`;
    barsEl.appendChild(row);
  });

  return results[0];
}

// ============================================================
// EVALUATE MODEL
// ============================================================
async function evaluateModel() {
  if (!STATE.trainedModel) { toast('Train a model first', 'error'); return null; }
  const predictions = [], actuals = [];
  for (let ci = 0; ci < STATE.classes.length; ci++) {
    for (const img of STATE.classes[ci].images) {
      const feat = await extractFeatures(img.src);
      const flat = feat.reshape([1, feat.shape.reduce((a,b)=>a*b,1)]);
      const pred = STATE.trainedModel.predict(flat);
      const arr = await pred.data();
      predictions.push(arr.indexOf(Math.max(...arr)));
      actuals.push(ci);
      flat.dispose(); feat.dispose(); pred.dispose();
    }
  }
  const correct = predictions.filter((p, i) => p === actuals[i]).length;
  const accuracy = correct / predictions.length;
  return { predictions, actuals, accuracy };
}

// ============================================================
// MODEL SAVE / LOAD
// ============================================================
async function saveModel() {
  if (!STATE.trainedModel) { toast('No trained model', 'error'); return; }
  try {
    await STATE.trainedModel.save('localstorage://neurallens-model');
    const meta = {
      classes: STATE.classes.map(c => ({ id: c.id, name: c.name, color: c.color })),
      bestAcc: STATE.bestAcc,
    };
    localStorage.setItem('neurallens-meta', JSON.stringify(meta));
    toast('Model saved to LocalStorage', 'success');
    log('Model saved to LocalStorage', 'success');
  } catch (e) {
    toast('Save failed: ' + e.message, 'error');
  }
}

async function loadModel() {
  try {
    const metaStr = localStorage.getItem('neurallens-meta');
    if (!metaStr) { toast('No saved model found', 'error'); return; }
    const meta = JSON.parse(metaStr);
    STATE.classifier = await tf.loadLayersModel('localstorage://neurallens-model');
    STATE.trainedModel = STATE.classifier;
    STATE.classes = meta.classes.map(c => ({ ...c, images: [] }));
    STATE.bestAcc = meta.bestAcc || 0;
    await loadMobileNet();
    renderClasses();
    updateClassSelectors();
    updateSummary();
    document.getElementById('model-status-badge').textContent = 'Loaded';
    document.getElementById('info-status').textContent = 'Loaded';
    document.getElementById('info-acc').textContent = (STATE.bestAcc * 100).toFixed(1) + '%';
    toast('Model loaded from LocalStorage', 'success');
    log('Model loaded. Add images and predict!', 'success');
  } catch (e) {
    toast('Load failed: ' + e.message, 'error');
  }
}

function resetAll() {
  if (!confirm('Reset everything? This clears all classes and model.')) return;
  STATE.classes.forEach(cls => cls.images.forEach(img => img.tensor && img.tensor.dispose()));
  if (STATE.classifier) STATE.classifier.dispose();
  STATE.classes = [];
  STATE.trainedModel = null;
  STATE.classifier = null;
  STATE.bestAcc = 0;
  STATE.chartData = { loss: [], acc: [] };
  renderClasses();
  updateClassSelectors();
  updateSummary();
  renderPreviewGrid();
  document.getElementById('training-progress').style.display = 'none';
  document.getElementById('model-status-badge').textContent = 'Untrained';
  document.getElementById('info-status').textContent = 'Not Loaded';
  document.getElementById('info-classes').textContent = '0';
  document.getElementById('info-acc').textContent = '—';
  document.getElementById('confidence-bars').innerHTML = '';
  document.getElementById('pred-top-label').textContent = '—';
  document.getElementById('pred-top-conf').textContent = 'Upload image & predict';
  toast('Reset complete', 'warning');
}

function exportDatasetInfo() {
  const info = {
    classes: STATE.classes.map(c => ({ name: c.name, imageCount: c.images.length })),
    totalImages: STATE.classes.reduce((s, c) => s + c.images.length, 0),
    bestAccuracy: STATE.bestAcc,
    exportedAt: new Date().toISOString(),
  };
  const blob = new Blob([JSON.stringify(info, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'neurallens-dataset-info.json';
  a.click();
  toast('Dataset info exported', 'success');
}

// ============================================================
// UI EVENT BINDING
// ============================================================
document.addEventListener('DOMContentLoaded', async () => {

  // Preload MobileNet
  loadMobileNet();

  // Add Class Modal
  const COLORS = CLASS_COLORS;
  let selectedColor = COLORS[0];

  document.getElementById('btn-add-class').onclick = () => {
    const modal = document.getElementById('class-modal');
    modal.style.display = 'flex';
    document.getElementById('class-name-input').value = '';
    const colWrap = document.getElementById('modal-colors');
    colWrap.innerHTML = '';
    selectedColor = COLORS[STATE.classes.length % COLORS.length];
    COLORS.forEach(c => {
      const dot = document.createElement('div');
      dot.className = `color-dot ${c === selectedColor ? 'selected' : ''}`;
      dot.style.background = c;
      dot.onclick = () => {
        selectedColor = c;
        colWrap.querySelectorAll('.color-dot').forEach(d => d.classList.remove('selected'));
        dot.classList.add('selected');
      };
      colWrap.appendChild(dot);
    });
    setTimeout(() => document.getElementById('class-name-input').focus(), 50);
  };

  document.getElementById('modal-cancel').onclick = () => {
    document.getElementById('class-modal').style.display = 'none';
  };

  document.getElementById('modal-confirm').onclick = () => {
    const name = document.getElementById('class-name-input').value;
    addClass(name, selectedColor);
    document.getElementById('class-modal').style.display = 'none';
  };

  document.getElementById('class-name-input').onkeydown = (e) => {
    if (e.key === 'Enter') document.getElementById('modal-confirm').click();
    if (e.key === 'Escape') document.getElementById('modal-cancel').click();
  };

  // Webcam
  document.getElementById('btn-start-cam').onclick = async () => {
    const video = document.getElementById('webcam');
    if (STATE.webcamStream) {
      STATE.webcamStream.getTracks().forEach(t => t.stop());
      STATE.webcamStream = null;
      video.srcObject = null;
      document.getElementById('btn-start-cam').textContent = '📷 Start Webcam';
      document.getElementById('btn-capture').disabled = true;
      return;
    }
    const stream = await startWebcam(video);
    if (stream) {
      STATE.webcamStream = stream;
      document.getElementById('btn-start-cam').textContent = '⏹ Stop Webcam';
      document.getElementById('btn-capture').disabled = false;
    }
  };

  document.getElementById('btn-capture').onclick = () => {
    const classId = document.getElementById('active-class-select').value;
    if (!classId) { toast('Select a class first', 'error'); return; }
    const src = captureFrame(document.getElementById('webcam'));
    addImageToClass(classId, src);
    toast('Image captured!');
  };

  // File upload
  document.getElementById('file-input').onchange = (e) => {
    const classId = document.getElementById('active-class-select').value;
    if (!classId) { toast('Select a class first', 'error'); return; }
    Array.from(e.target.files).forEach(file => {
      const reader = new FileReader();
      reader.onload = ev => addImageToClass(classId, ev.target.result);
      reader.readAsDataURL(file);
    });
    e.target.value = '';
  };

  // Drag & drop
  const dropZone = document.getElementById('drop-zone');
  dropZone.ondragover = (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); };
  dropZone.ondragleave = () => dropZone.classList.remove('drag-over');
  dropZone.ondrop = (e) => {
    e.preventDefault(); dropZone.classList.remove('drag-over');
    const classId = document.getElementById('active-class-select').value;
    if (!classId) { toast('Select a class first', 'error'); return; }
    Array.from(e.dataTransfer.files).forEach(file => {
      if (!file.type.startsWith('image/')) return;
      const reader = new FileReader();
      reader.onload = ev => addImageToClass(classId, ev.target.result);
      reader.readAsDataURL(file);
    });
  };

  // Training
  document.getElementById('btn-train').onclick = trainModel;
  document.getElementById('btn-stop').onclick = () => {
    STATE.stopRequested = true;
    toast('Stopping training...', 'warning');
  };

  // Prediction tabs
  document.querySelectorAll('.pred-tab').forEach(tab => {
    tab.onclick = () => {
      document.querySelectorAll('.pred-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const target = tab.dataset.tab;
      document.getElementById('pred-webcam-tab').classList.toggle('hidden', target !== 'webcam');
      document.getElementById('pred-upload-tab').classList.toggle('hidden', target !== 'upload');
    };
  });

  // Prediction webcam
  document.getElementById('btn-pred-cam').onclick = async () => {
    const video = document.getElementById('pred-webcam');
    if (STATE.predWebcamStream) {
      clearInterval(STATE.predLiveInterval);
      STATE.predWebcamStream.getTracks().forEach(t => t.stop());
      STATE.predWebcamStream = null;
      video.srcObject = null;
      document.getElementById('btn-pred-cam').textContent = '📷 Start Webcam';
      document.getElementById('btn-pred-capture').disabled = true;
      return;
    }
    const stream = await startWebcam(video);
    if (stream) {
      STATE.predWebcamStream = stream;
      document.getElementById('btn-pred-cam').textContent = '⏹ Stop';
      document.getElementById('btn-pred-capture').disabled = false;
    }
  };

  document.getElementById('btn-pred-capture').onclick = async () => {
    if (!STATE.trainedModel) { toast('Train a model first', 'error'); return; }
    const src = captureFrame(document.getElementById('pred-webcam'));
    await predict(src);
  };

  // Prediction upload
  document.getElementById('pred-file-input').onchange = (e) => {
    const file = e.target.files[0]; if (!file) return;
    const reader = new FileReader();
    reader.onload = ev => {
      const img = document.getElementById('pred-preview');
      img.src = ev.target.result;
      img.style.display = 'block';
      document.querySelector('.pred-placeholder').style.display = 'none';
    };
    reader.readAsDataURL(file);
    e.target.value = '';
  };

  document.getElementById('btn-predict-upload').onclick = async () => {
    const src = document.getElementById('pred-preview').src;
    if (!src || src === location.href) { toast('Upload an image first', 'error'); return; }
    await predict(src);
  };

  // Save/Load
  document.getElementById('btn-save').onclick = saveModel;
  document.getElementById('btn-load').onclick = loadModel;
  document.getElementById('btn-reset').onclick = resetAll;
  document.getElementById('btn-export').onclick = exportDatasetInfo;

  // Add sample classes
  addClass('Apple', '#e8652a');
  addClass('Banana', '#f5a623');
  addClass('Orange', '#27ae60');
});
