/**
 * Assignment 2 — Confusion Matrix Evaluation
 * Evaluate model: Accuracy + Visual Confusion Matrix + Misclassifications
 */

// ============================================================
// STATE
// ============================================================
const PALETTE = ['#6366f1','#f59e0b','#22c55e','#ef4444','#8b5cf6','#06b6d4','#ec4899','#84cc16'];

const STATE = {
  mobilenet: null,
  featureModel: null,
  classifier: null,
  trainedModel: null,
  classes: [],
  images: {}, // classId -> [{src}]
  isTraining: false,
  stopRequested: false,
};

function uid() { return Math.random().toString(36).slice(2,8); }
function toast(msg, type='') {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  document.getElementById('toast-container').appendChild(el);
  setTimeout(()=>el.remove(), 3000);
}
function setStatus(t) { document.getElementById('hdr-status').textContent = t; }

// ============================================================
// MOBILENET
// ============================================================
async function loadMobileNet() {
  setStatus('Loading MobileNet...');
  STATE.mobilenet = await mobilenet.load({ version: 2, alpha: 0.5 });
  const layer = STATE.mobilenet.model.getLayer('conv_pw_13_relu');
  STATE.featureModel = tf.model({ inputs: STATE.mobilenet.model.inputs, outputs: layer.output });
  setStatus('Ready');
  toast('MobileNet loaded', 'success');
}

async function imageToTensor(src) {
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
// CLASS MANAGEMENT
// ============================================================
function addClass(name, color) {
  const id = uid();
  STATE.classes.push({ id, name, color });
  STATE.images[id] = [];
  renderAll();
}

function removeClassById(id) {
  STATE.classes = STATE.classes.filter(c => c.id !== id);
  delete STATE.images[id];
  renderAll();
}

function renderAll() {
  renderTrainList();
  renderConfigList();
}

function renderTrainList() {
  const cont = document.getElementById('class-list-train');
  cont.innerHTML = '';
  STATE.classes.forEach(cls => {
    const div = document.createElement('div');
    div.className = 'class-item';
    div.innerHTML = `
      <div class="class-item-hdr">
        <div><span class="class-color-dot" style="background:${cls.color}"></span>${cls.name}</div>
        <span style="font-size:11px;color:var(--text2)">${(STATE.images[cls.id]||[]).length} img</span>
      </div>
      <div class="class-item-body">
        <label class="class-upload-lbl" for="upl-${cls.id}">+ Upload Images</label>
        <input type="file" id="upl-${cls.id}" accept="image/*" multiple hidden/>
        <div class="class-thumbs" id="thumbs-${cls.id}"></div>
      </div>`;
    cont.appendChild(div);
    document.getElementById(`upl-${cls.id}`).onchange = e => {
      Array.from(e.target.files).forEach(f => {
        const r = new FileReader();
        r.onload = ev => { STATE.images[cls.id].push({src:ev.target.result}); renderTrainList(); };
        r.readAsDataURL(f);
      });
      e.target.value = '';
    };
    const thumbsCont = document.getElementById(`thumbs-${cls.id}`);
    (STATE.images[cls.id]||[]).slice(-6).forEach(img => {
      const el = document.createElement('img');
      el.className = 'class-thumb-img'; el.src = img.src;
      thumbsCont.appendChild(el);
    });
  });
}

function renderConfigList() {
  const cont = document.getElementById('class-config-list');
  cont.innerHTML = '';
  STATE.classes.forEach(cls => {
    const div = document.createElement('div');
    div.className = 'cfg-item';
    div.innerHTML = `
      <div style="display:flex;align-items:center;gap:8px">
        <div style="width:10px;height:10px;border-radius:50%;background:${cls.color}"></div>
        <span style="font-size:13px">${cls.name}</span>
      </div>
      <button class="cfg-del" onclick="removeClassById('${cls.id}')">✕</button>`;
    cont.appendChild(div);
  });
}

// ============================================================
// TRAINING
// ============================================================
async function trainModel() {
  if (STATE.isTraining) return;
  if (STATE.classes.length < 2) { toast('Need at least 2 classes', 'error'); return; }
  for (const cls of STATE.classes) {
    if (!STATE.images[cls.id] || STATE.images[cls.id].length < 2) {
      toast(`Need images for ${cls.name}`, 'error'); return;
    }
  }

  STATE.isTraining = true;
  STATE.stopRequested = false;
  setStatus('Extracting features...');
  document.getElementById('btn-train').disabled = true;
  document.getElementById('train-mini-progress').style.display = 'block';

  const epochs = parseInt(document.getElementById('epochs').value) || 25;
  const batchSize = parseInt(document.getElementById('batchSize').value) || 16;
  const valSplit = parseFloat(document.getElementById('valSplit').value) || 0.2;

  const xs = [], ys = [];
  for (let ci = 0; ci < STATE.classes.length; ci++) {
    for (const img of STATE.images[STATE.classes[ci].id]) {
      const feat = await extractFeatures(img.src);
      xs.push(feat); ys.push(ci);
    }
  }

  const xStack = tf.stack(xs);
  const flatSize = xStack.shape.slice(1).reduce((a,b)=>a*b,1);
  const xFlat = xStack.reshape([-1, flatSize]);
  const yTensor = tf.oneHot(tf.tensor1d(ys,'int32'), STATE.classes.length);
  xStack.dispose(); xs.forEach(t=>t.dispose());

  if (STATE.classifier) STATE.classifier.dispose();
  STATE.classifier = tf.sequential();
  STATE.classifier.add(tf.layers.dense({ units: 256, activation:'relu', inputShape:[flatSize], kernelInitializer:'glorotNormal' }));
  STATE.classifier.add(tf.layers.batchNormalization());
  STATE.classifier.add(tf.layers.dropout({ rate: 0.35 }));
  STATE.classifier.add(tf.layers.dense({ units: 128, activation:'relu' }));
  STATE.classifier.add(tf.layers.dropout({ rate: 0.2 }));
  STATE.classifier.add(tf.layers.dense({ units: STATE.classes.length, activation:'softmax' }));
  STATE.classifier.compile({ optimizer: tf.train.adam(0.0001), loss:'categoricalCrossentropy', metrics:['accuracy'] });

  setStatus('Training...');

  await STATE.classifier.fit(xFlat, yTensor, {
    epochs, batchSize, shuffle: true, validationSplit: valSplit,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        if (STATE.stopRequested) { STATE.classifier.stopTraining = true; return; }
        const pct = ((epoch+1)/epochs)*100;
        document.getElementById('mini-bar').style.width = pct+'%';
        document.getElementById('mini-label').textContent = `Epoch ${epoch+1}/${epochs}`;
        document.getElementById('mini-metrics').innerHTML =
          `<span class="mini-metric">Train: ${(logs.acc*100).toFixed(1)}%</span>
           <span class="mini-metric">Val: ${((logs.val_acc||0)*100).toFixed(1)}%</span>
           <span class="mini-metric">Loss: ${logs.loss.toFixed(3)}</span>`;
        await tf.nextFrame();
      }
    }
  });

  xFlat.dispose(); yTensor.dispose();
  STATE.trainedModel = STATE.classifier;
  STATE.isTraining = false;
  document.getElementById('btn-train').disabled = false;
  setStatus('Trained — click Evaluate');
  toast('Training complete!', 'success');
}

// ============================================================
// EVALUATION + CONFUSION MATRIX
// ============================================================
async function evaluateModel() {
  if (!STATE.trainedModel) { toast('Train model first', 'error'); return; }
  setStatus('Evaluating...');

  const predictions = [], actuals = [], imgSrcs = [];
  for (let ci = 0; ci < STATE.classes.length; ci++) {
    for (const img of STATE.images[STATE.classes[ci].id]) {
      const feat = await extractFeatures(img.src);
      const flat = feat.reshape([1, feat.shape.reduce((a,b)=>a*b,1)]);
      const pred = STATE.trainedModel.predict(flat);
      const arr = await pred.data();
      predictions.push(arr.indexOf(Math.max(...arr)));
      actuals.push(ci);
      imgSrcs.push({ src: img.src, actual: ci, predicted: arr.indexOf(Math.max(...arr)) });
      flat.dispose(); feat.dispose(); pred.dispose();
    }
  }

  const n = STATE.classes.length;
  const matrix = Array.from({length:n}, ()=>Array(n).fill(0));
  predictions.forEach((p,i) => matrix[actuals[i]][p]++);

  const correct = predictions.filter((p,i)=>p===actuals[i]).length;
  const overallAcc = correct / predictions.length;

  // Per-class metrics
  const perClass = STATE.classes.map((_,ci) => {
    const tp = matrix[ci][ci];
    const fp = STATE.classes.reduce((s,_,ri)=>ri!==ci?s+matrix[ri][ci]:s,0);
    const fn = STATE.classes.reduce((s,_,pi)=>pi!==ci?s+matrix[ci][pi]:s,0);
    const prec = tp+fp>0 ? tp/(tp+fp) : 0;
    const rec = tp+fn>0 ? tp/(tp+fn) : 0;
    const f1 = prec+rec>0 ? 2*prec*rec/(prec+rec) : 0;
    return { prec, rec, f1, tp, total: actuals.filter(a=>a===ci).length };
  });

  const macroPrecision = perClass.reduce((s,c)=>s+c.prec,0)/n;
  const macroRecall = perClass.reduce((s,c)=>s+c.rec,0)/n;
  const macroF1 = perClass.reduce((s,c)=>s+c.f1,0)/n;

  // Update summary
  document.getElementById('overall-acc').textContent = (overallAcc*100).toFixed(1)+'%';
  document.getElementById('macro-prec').textContent = (macroPrecision*100).toFixed(1)+'%';
  document.getElementById('macro-recall').textContent = (macroRecall*100).toFixed(1)+'%';
  document.getElementById('macro-f1').textContent = (macroF1*100).toFixed(2);

  renderConfusionMatrix(matrix);
  renderPerClassStats(perClass);
  renderMisclassifications(imgSrcs);

  setStatus(`Evaluated — ${(overallAcc*100).toFixed(1)}% acc`);
  toast(`Accuracy: ${(overallAcc*100).toFixed(1)}%`, 'success');
}

function renderConfusionMatrix(matrix) {
  const n = STATE.classes.length;
  const maxVal = Math.max(...matrix.flat(), 1);
  const cont = document.getElementById('confusion-matrix');

  let html = `<table class="cm-table"><thead><tr><th></th>`;
  STATE.classes.forEach(c => { html += `<th><span style="color:${c.color}">${c.name}</span></th>`; });
  html += '<th>Total</th></tr></thead><tbody>';

  STATE.classes.forEach((cls, ri) => {
    const rowSum = matrix[ri].reduce((a,b)=>a+b,0);
    html += `<tr><td class="cm-label-cell"><strong style="color:${cls.color}">${cls.name}</strong></td>`;
    matrix[ri].forEach((val, ci) => {
      const isCorrect = ri === ci;
      const intensity = val / maxVal;
      let bg, color;
      if (isCorrect) {
        bg = `rgba(99,102,241,${0.15 + intensity*0.85})`;
        color = intensity > 0.5 ? '#fff' : '#6366f1';
      } else if (val > 0) {
        bg = `rgba(239,68,68,${0.1 + intensity*0.7})`;
        color = intensity > 0.5 ? '#fff' : '#ef4444';
      } else {
        bg = '#f9f9f9'; color = '#ccc';
      }
      const title = `Actual: ${STATE.classes[ri].name}, Predicted: ${STATE.classes[ci].name}`;
      html += `<td title="${title}"><div class="cm-cell" style="background:${bg};color:${color}">${val}</div></td>`;
    });
    const acc = rowSum > 0 ? ((matrix[ri][ri]/rowSum)*100).toFixed(0)+'%' : '—';
    html += `<td><div class="cm-cell" style="background:#f3f4f6;color:#71717a;font-size:11px">${acc}</div></td>`;
    html += '</tr>';
  });

  // Predicted totals row
  html += '<tr><td class="cm-label-cell" style="color:var(--text2)">Predicted</td>';
  STATE.classes.forEach((_,ci) => {
    const colSum = STATE.classes.reduce((s,_,ri)=>s+matrix[ri][ci],0);
    html += `<td><div class="cm-cell" style="background:#f3f4f6;color:#71717a;font-size:11px">${colSum}</div></td>`;
  });
  html += '<td></td></tr>';

  html += '</tbody></table>';
  cont.innerHTML = html;
}

function renderPerClassStats(perClass) {
  const cont = document.getElementById('per-class-stats');
  let html = `<div class="pcls-header">
    <span>Class</span><span>Precision bar</span>
    <span>Prec</span><span>Recall</span><span>F1</span>
  </div>`;
  STATE.classes.forEach((cls,i) => {
    const p = perClass[i];
    html += `<div class="pcls-row">
      <div class="pcls-name"><span style="color:${cls.color}">●</span> ${cls.name}</div>
      <div class="pcls-bar-wrap">
        <div class="pcls-bar" style="width:${(p.prec*100).toFixed(0)}%;background:${cls.color}"></div>
      </div>
      <div class="pcls-val">${(p.prec*100).toFixed(0)}%</div>
      <div class="pcls-val">${(p.rec*100).toFixed(0)}%</div>
      <div class="pcls-val">${(p.f1*100).toFixed(0)}%</div>
    </div>`;
  });
  cont.innerHTML = html;
}

function renderMisclassifications(imgSrcs) {
  const misclassified = imgSrcs.filter(r => r.actual !== r.predicted);
  const panel = document.getElementById('misc-panel');
  if (misclassified.length === 0) { panel.style.display = 'none'; return; }
  panel.style.display = 'block';
  const grid = document.getElementById('misc-grid');
  grid.innerHTML = '';
  misclassified.slice(0,12).forEach(r => {
    const div = document.createElement('div');
    div.className = 'misc-item';
    div.innerHTML = `
      <img src="${r.src}" alt=""/>
      <div class="misc-label">
        ✓ ${STATE.classes[r.actual].name}<br>
        ✗ ${STATE.classes[r.predicted].name}
      </div>`;
    grid.appendChild(div);
  });
}

// ============================================================
// MODAL
// ============================================================
let selectedColor = PALETTE[0];

function openAddClassModal() {
  const modal = document.getElementById('modal');
  modal.style.display = 'flex';
  document.getElementById('modal-name').value = '';
  selectedColor = PALETTE[STATE.classes.length % PALETTE.length];
  const colorsCont = document.getElementById('modal-colors');
  colorsCont.innerHTML = '';
  PALETTE.forEach(c => {
    const dot = document.createElement('div');
    dot.className = `mcol ${c === selectedColor ? 'sel' : ''}`;
    dot.style.background = c;
    dot.onclick = () => {
      selectedColor = c;
      colorsCont.querySelectorAll('.mcol').forEach(d=>d.classList.remove('sel'));
      dot.classList.add('sel');
    };
    colorsCont.appendChild(dot);
  });
  setTimeout(()=>document.getElementById('modal-name').focus(), 50);
}

// ============================================================
// INIT
// ============================================================
document.addEventListener('DOMContentLoaded', async () => {
  // Default classes
  addClass('Apple', '#6366f1');
  addClass('Banana', '#f59e0b');
  addClass('Orange', '#22c55e');

  await loadMobileNet();

  document.getElementById('btn-add-class').onclick = openAddClassModal;
  document.getElementById('modal-cancel').onclick = ()=>{ document.getElementById('modal').style.display='none'; };
  document.getElementById('modal-ok').onclick = ()=>{
    const name = document.getElementById('modal-name').value.trim();
    if (!name) { toast('Enter class name', 'error'); return; }
    addClass(name, selectedColor);
    document.getElementById('modal').style.display='none';
  };
  document.getElementById('modal-name').onkeydown = e => {
    if (e.key==='Enter') document.getElementById('modal-ok').click();
    if (e.key==='Escape') document.getElementById('modal-cancel').click();
  };

  document.getElementById('btn-train').onclick = trainModel;
  document.getElementById('btn-evaluate').onclick = evaluateModel;
});
