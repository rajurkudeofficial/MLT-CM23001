/**
 * Assignment 1 — MobileNet Fruit Classifier
 * 3 fixed classes: Apple, Banana, Orange
 * Shows training accuracy + validation accuracy
 */

// ============================================================
// CONFIG
// ============================================================
const CLASSES = [
  { id: 'apple',  name: 'Apple',  emoji: '🍎', color: '#ff6b6b' },
  { id: 'banana', name: 'Banana', emoji: '🍌', color: '#ffd93d' },
  { id: 'orange', name: 'Orange', emoji: '🍊', color: '#ff9f43' },
];

const STATE = {
  mobilenet: null,
  featureModel: null,
  classifier: null,
  trainedModel: null,
  images: { apple: [], banana: [], orange: [] },  // {src}
  isTraining: false,
  stopRequested: false,
  chartData: { trainAcc: [], valAcc: [], loss: [] },
  webcamStream: null,
};

// ============================================================
// UTILS
// ============================================================
function toast(msg, type = '') {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  document.getElementById('toast-container').appendChild(el);
  setTimeout(() => el.remove(), 3000);
}

function setStatus(text, cls = '') {
  const pill = document.getElementById('status-pill');
  pill.textContent = text;
  pill.className = `status-pill ${cls}`;
}

// ============================================================
// MOBILENET LOAD
// ============================================================
async function loadMobileNet() {
  setStatus('⏳ Loading MobileNet...');
  try {
    STATE.mobilenet = await mobilenet.load({ version: 2, alpha: 0.5 });
    const layer = STATE.mobilenet.model.getLayer('conv_pw_13_relu');
    STATE.featureModel = tf.model({
      inputs: STATE.mobilenet.model.inputs,
      outputs: layer.output,
    });
    setStatus('✅ MobileNet Ready', 'ready');
    toast('MobileNet loaded!', 'success');
  } catch (e) {
    setStatus('❌ Load Failed');
    toast('MobileNet failed to load', 'error');
  }
}

// ============================================================
// IMAGE → TENSOR
// ============================================================
function imageToTensor(src) {
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => {
      const t = tf.tidy(() =>
        tf.browser.fromPixels(img).resizeBilinear([224,224]).toFloat().div(127.5).sub(1).expandDims(0)
      );
      resolve(t);
    };
    img.src = src;
  });
}

async function extractFeatures(src) {
  const t = await imageToTensor(src);
  const feat = tf.tidy(() => STATE.featureModel.predict(t).squeeze());
  t.dispose();
  return feat;
}

// ============================================================
// CLASS PANEL RENDERING
// ============================================================
function renderClassPanels() {
  const container = document.getElementById('class-panels');
  container.innerHTML = '';
  CLASSES.forEach(cls => {
    const panel = document.createElement('div');
    panel.className = 'class-panel';
    panel.innerHTML = `
      <div class="class-header">
        <div class="class-label">
          <span class="class-emoji">${cls.emoji}</span>
          <span>${cls.name}</span>
        </div>
        <span class="class-count-badge" id="count-${cls.id}">${STATE.images[cls.id].length} images</span>
      </div>
      <div class="class-upload-area">
        <label class="class-file-label" for="file-${cls.id}">+ Add Images</label>
        <input type="file" id="file-${cls.id}" accept="image/*" multiple hidden/>
        <div class="class-thumb-row" id="thumbs-${cls.id}"></div>
      </div>`;
    container.appendChild(panel);

    document.getElementById(`file-${cls.id}`).onchange = (e) => {
      Array.from(e.target.files).forEach(file => {
        const reader = new FileReader();
        reader.onload = ev => addImage(cls.id, ev.target.result);
        reader.readAsDataURL(file);
      });
      e.target.value = '';
    };
  });
}

function addImage(classId, src) {
  STATE.images[classId].push({ src });
  updateClassCount(classId);
  renderThumbs(classId);
  updateSummary();
}

function updateClassCount(classId) {
  const el = document.getElementById(`count-${classId}`);
  if (el) el.textContent = `${STATE.images[classId].length} images`;
}

function renderThumbs(classId) {
  const row = document.getElementById(`thumbs-${classId}`);
  if (!row) return;
  row.innerHTML = '';
  STATE.images[classId].slice(-5).forEach(img => {
    const el = document.createElement('img');
    el.className = 'class-thumb'; el.src = img.src;
    row.appendChild(el);
  });
}

function updateSummary() {
  const grid = document.getElementById('summary-grid');
  const maxCount = Math.max(...CLASSES.map(c => STATE.images[c.id].length), 1);
  grid.innerHTML = '';
  CLASSES.forEach(cls => {
    const count = STATE.images[cls.id].length;
    const row = document.createElement('div');
    row.className = 'summary-row';
    row.innerHTML = `
      <span>${cls.emoji} ${cls.name}</span>
      <div class="summary-bar-wrap">
        <div class="summary-bar" style="width:${(count/maxCount)*100}%;background:${cls.color}"></div>
      </div>
      <strong>${count}</strong>`;
    grid.appendChild(row);
  });
}

// ============================================================
// TRAINING
// ============================================================
async function trainModel() {
  if (STATE.isTraining) return;
  for (const cls of CLASSES) {
    if (STATE.images[cls.id].length < 2) {
      toast(`Need at least 2 images for ${cls.name}`, 'error'); return;
    }
  }

  STATE.isTraining = true;
  STATE.stopRequested = false;
  setStatus('🔄 Extracting features...', 'training');

  const epochs = parseInt(document.getElementById('epochs').value) || 30;
  const batchSize = parseInt(document.getElementById('batch-size').value) || 16;
  const valSplit = parseFloat(document.getElementById('val-split').value) || 0.2;

  document.getElementById('btn-train').disabled = true;
  document.getElementById('btn-stop').disabled = false;
  document.getElementById('progress-wrap').style.display = 'block';
  document.getElementById('results-card').style.display = 'none';
  STATE.chartData = { trainAcc: [], valAcc: [], loss: [] };

  // Feature extraction
  const xs = [], ys = [];
  for (let ci = 0; ci < CLASSES.length; ci++) {
    for (const img of STATE.images[CLASSES[ci].id]) {
      const feat = await extractFeatures(img.src);
      xs.push(feat); ys.push(ci);
    }
  }

  const xStack = tf.stack(xs);
  const flatSize = xStack.shape.slice(1).reduce((a,b)=>a*b,1);
  const xFlat = xStack.reshape([-1, flatSize]);
  const yTensor = tf.oneHot(tf.tensor1d(ys,'int32'), CLASSES.length);
  xStack.dispose(); xs.forEach(t=>t.dispose());

  // Build model
  if (STATE.classifier) STATE.classifier.dispose();
  STATE.classifier = tf.sequential();
  STATE.classifier.add(tf.layers.dense({ units: 256, activation: 'relu', inputShape: [flatSize], kernelInitializer: 'glorotNormal' }));
  STATE.classifier.add(tf.layers.batchNormalization());
  STATE.classifier.add(tf.layers.dropout({ rate: 0.4 }));
  STATE.classifier.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  STATE.classifier.add(tf.layers.dropout({ rate: 0.2 }));
  STATE.classifier.add(tf.layers.dense({ units: CLASSES.length, activation: 'softmax' }));
  STATE.classifier.compile({ optimizer: tf.train.adam(0.0001), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

  setStatus('🏋️ Training...', 'training');

  const accCtx = document.getElementById('acc-chart').getContext('2d');
  const lossCtx = document.getElementById('loss-chart').getContext('2d');

  await STATE.classifier.fit(xFlat, yTensor, {
    epochs, batchSize, shuffle: true, validationSplit: valSplit,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        if (STATE.stopRequested) { STATE.classifier.stopTraining = true; return; }
        const pct = ((epoch+1)/epochs)*100;
        document.getElementById('prog-fill').style.width = pct+'%';
        document.getElementById('epoch-info').textContent = `Epoch ${epoch+1}/${epochs}`;
        document.getElementById('train-acc-val').textContent = (logs.acc*100).toFixed(1)+'%';
        document.getElementById('val-acc-val').textContent = ((logs.val_acc||0)*100).toFixed(1)+'%';
        document.getElementById('loss-val').textContent = logs.loss.toFixed(4);
        STATE.chartData.trainAcc.push(logs.acc*100);
        STATE.chartData.valAcc.push((logs.val_acc||0)*100);
        STATE.chartData.loss.push(logs.loss);
        drawAccChart(accCtx, 520, 200);
        drawLossChart(lossCtx, 520, 160);
        await tf.nextFrame();
      }
    }
  });

  xFlat.dispose(); yTensor.dispose();
  STATE.trainedModel = STATE.classifier;
  STATE.isTraining = false;

  const finalTrainAcc = STATE.chartData.trainAcc.at(-1);
  const finalValAcc = STATE.chartData.valAcc.at(-1);

  document.getElementById('btn-train').disabled = false;
  document.getElementById('btn-stop').disabled = true;
  setStatus(`✅ Trained — Val Acc: ${finalValAcc.toFixed(1)}%`, 'ready');

  // Show results
  const resultsCard = document.getElementById('results-card');
  resultsCard.style.display = 'block';
  document.getElementById('final-metrics').innerHTML = `
    <div class="final-metric">
      <div class="val" style="color:#55efc4">${finalTrainAcc.toFixed(1)}%</div>
      <div class="lbl">Final Train Accuracy</div>
    </div>
    <div class="final-metric">
      <div class="val" style="color:#74b9ff">${finalValAcc.toFixed(1)}%</div>
      <div class="lbl">Final Val Accuracy</div>
    </div>
    <div class="final-metric">
      <div class="val" style="color:#ff6b6b">${STATE.chartData.loss.at(-1).toFixed(4)}</div>
      <div class="lbl">Final Loss</div>
    </div>
    <div class="final-metric">
      <div class="val">${STATE.chartData.trainAcc.length}</div>
      <div class="lbl">Epochs Completed</div>
    </div>`;

  toast('Training complete! 🎉', 'success');
}

// ============================================================
// CHARTS
// ============================================================
function drawAccChart(ctx, w, h) {
  const { trainAcc, valAcc } = STATE.chartData;
  if (trainAcc.length < 2) return;
  ctx.clearRect(0,0,w,h);
  const pd = 30, cw = w-pd*2, ch = h-pd*2;
  ctx.fillStyle = 'rgba(0,0,0,0.2)'; ctx.fillRect(0,0,w,h);

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.08)'; ctx.lineWidth = 1;
  [0,25,50,75,100].forEach(v => {
    const y = pd + ch - (v/100)*ch;
    ctx.beginPath(); ctx.moveTo(pd,y); ctx.lineTo(pd+cw,y); ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,0.3)'; ctx.font = '9px JetBrains Mono';
    ctx.fillText(v+'%', 2, y+3);
  });

  function line(data, color) {
    ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 2.5;
    data.forEach((v,i) => {
      const x = pd+(i/(data.length-1))*cw;
      const y = pd+ch-(v/100)*ch;
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    });
    ctx.stroke();
    // Dot at end
    const ex = pd+(data.length-1)/(data.length-1)*cw;
    const ey = pd+ch-(data[data.length-1]/100)*ch;
    ctx.beginPath(); ctx.arc(ex,ey,4,0,Math.PI*2);
    ctx.fillStyle=color; ctx.fill();
  }
  line(trainAcc, '#55efc4');
  line(valAcc, '#74b9ff');

  // Legend
  ctx.font = '10px Outfit';
  ctx.fillStyle = '#55efc4'; ctx.fillRect(pd, 6, 12, 4);
  ctx.fillStyle = '#fff'; ctx.fillText('Train Acc', pd+16, 12);
  ctx.fillStyle = '#74b9ff'; ctx.fillRect(pd+90, 6, 12, 4);
  ctx.fillStyle = '#fff'; ctx.fillText('Val Acc', pd+106, 12);
}

function drawLossChart(ctx, w, h) {
  const { loss } = STATE.chartData;
  if (loss.length < 2) return;
  ctx.clearRect(0,0,w,h);
  const pd = 30, cw = w-pd*2, ch = h-pd*2;
  ctx.fillStyle = 'rgba(0,0,0,0.2)'; ctx.fillRect(0,0,w,h);
  const maxLoss = Math.max(...loss, 0.1);
  ctx.beginPath(); ctx.strokeStyle = '#ff6b6b'; ctx.lineWidth = 2.5;
  loss.forEach((v,i) => {
    const x = pd+(i/(loss.length-1))*cw;
    const y = pd+ch-(v/maxLoss)*ch;
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  });
  ctx.stroke();
  ctx.font = '10px Outfit'; ctx.fillStyle = '#ff6b6b';
  ctx.fillRect(pd, 6, 12, 4); ctx.fillStyle='#fff'; ctx.fillText('Loss', pd+16, 12);
}

// ============================================================
// PREDICTION
// ============================================================
async function predict(src) {
  if (!STATE.trainedModel) { toast('Train the model first', 'error'); return; }
  const feat = await extractFeatures(src);
  const flat = feat.reshape([1, feat.shape.reduce((a,b)=>a*b,1)]);
  const preds = STATE.trainedModel.predict(flat);
  const data = await preds.data();
  flat.dispose(); feat.dispose(); preds.dispose();

  const results = CLASSES.map((cls, i) => ({ ...cls, conf: data[i] }));
  results.sort((a,b) => b.conf - a.conf);
  const top = results[0];

  const resultEl = document.getElementById('pred-result');
  resultEl.style.display = 'block';
  document.getElementById('pred-badge').textContent = `${top.emoji} ${top.name} — ${(top.conf*100).toFixed(1)}%`;
  document.getElementById('pred-badge').style.color = top.color;

  const barsEl = document.getElementById('pred-bars');
  barsEl.innerHTML = '';
  results.forEach(r => {
    const row = document.createElement('div');
    row.className = 'pred-bar-row';
    row.innerHTML = `
      <div class="pred-bar-label">${r.emoji} ${r.name}</div>
      <div class="pred-bar-track">
        <div class="pred-bar-fill" style="width:${(r.conf*100).toFixed(1)}%;background:${r.color}"></div>
      </div>
      <div class="pred-bar-pct">${(r.conf*100).toFixed(0)}%</div>`;
    barsEl.appendChild(row);
  });
}

// ============================================================
// INIT
// ============================================================
document.addEventListener('DOMContentLoaded', async () => {
  renderClassPanels();
  updateSummary();
  await loadMobileNet();

  document.getElementById('btn-train').onclick = trainModel;
  document.getElementById('btn-stop').onclick = () => { STATE.stopRequested = true; };

  document.getElementById('pred-input').onchange = (e) => {
    const file = e.target.files[0]; if (!file) return;
    const reader = new FileReader();
    reader.onload = ev => {
      const img = document.getElementById('pred-img');
      img.src = ev.target.result; img.style.display = 'block';
      document.getElementById('pred-empty').style.display = 'none';
    };
    reader.readAsDataURL(file);
    e.target.value = '';
  };

  document.getElementById('btn-predict').onclick = async () => {
    const src = document.getElementById('pred-img').src;
    if (!src || src === location.href) { toast('Upload image first', 'error'); return; }
    await predict(src);
  };

  // Webcam
  let webcamStream = null;
  document.getElementById('btn-cam').onclick = async () => {
    const video = document.getElementById('webcam');
    if (webcamStream) {
      webcamStream.getTracks().forEach(t=>t.stop());
      webcamStream = null; video.style.display='none';
      document.getElementById('btn-cam').textContent = '📷 Webcam';
      document.getElementById('btn-capture').disabled = true;
      return;
    }
    try {
      webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
      video.srcObject = webcamStream; video.style.display = 'block';
      document.getElementById('btn-cam').textContent = '⏹ Stop';
      document.getElementById('btn-capture').disabled = false;
    } catch { toast('Webcam denied', 'error'); }
  };

  document.getElementById('btn-capture').onclick = async () => {
    const video = document.getElementById('webcam');
    const canvas = document.createElement('canvas');
    canvas.width = 224; canvas.height = 224;
    canvas.getContext('2d').drawImage(video, 0, 0, 224, 224);
    const src = canvas.toDataURL('image/jpeg', 0.8);
    const img = document.getElementById('pred-img');
    img.src = src; img.style.display = 'block';
    document.getElementById('pred-empty').style.display = 'none';
    await predict(src);
  };
});
