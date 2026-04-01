/**
 * Assignment 3 — Incremental Learning
 * - Train base model
 * - Add new class → retrain
 * - Compare old vs new accuracy
 * - Show performance graph
 */

// ============================================================
// STATE
// ============================================================
const PALETTE = ['#7c3aed','#db2777','#d97706','#059669','#2563eb','#dc2626','#0891b2','#65a30d'];

const STATE = {
  mobilenet: null,
  featureModel: null,
  initialClasses: [],     // [{id, name, color, images:[{src}]}]
  initialModel: null,
  initialAcc: null,
  incrementalClasses: [], // copy + new class
  incrementalModel: null,
  incrementalAcc: null,
  isTraining: false,
  stopRequested: false,
  chartDataInitial: { acc: [], valAcc: [] },
  chartDataIncremental: { acc: [], valAcc: [] },
};

function uid() { return Math.random().toString(36).slice(2, 8); }
function toast(msg, type='') {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  document.getElementById('toast-container').appendChild(el);
  setTimeout(()=>el.remove(), 3200);
}
function setStatus(text, cls='') {
  document.getElementById('status-text').textContent = text;
  const dot = document.getElementById('status-dot');
  dot.className = 'status-dot ' + cls;
}

// ============================================================
// MOBILENET
// ============================================================
async function loadMobileNet() {
  setStatus('Loading MobileNet...', 'training');
  STATE.mobilenet = await mobilenet.load({ version: 2, alpha: 0.5 });
  const layer = STATE.mobilenet.model.getLayer('conv_pw_13_relu');
  STATE.featureModel = tf.model({ inputs: STATE.mobilenet.model.inputs, outputs: layer.output });
  setStatus('Ready', 'ready');
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
// INITIAL CLASSES
// ============================================================
function addInitialClass(name, color) {
  STATE.initialClasses.push({ id: uid(), name, color, images: [] });
  renderInitialClasses();
}

function renderInitialClasses() {
  const cont = document.getElementById('initial-classes');
  cont.innerHTML = '';
  STATE.initialClasses.forEach((cls, idx) => {
    const row = document.createElement('div');
    row.className = 'cls-row';
    row.innerHTML = `
      <div class="cls-color" style="background:${cls.color}"></div>
      <div class="cls-name">${cls.name}</div>
      <div class="cls-count">${cls.images.length} img</div>
      <div class="cls-thumbs" id="ithumb-${cls.id}"></div>
      <label class="cls-upload-lbl" for="ifile-${cls.id}">+ Images</label>
      <input type="file" id="ifile-${cls.id}" accept="image/*" multiple hidden/>`;
    cont.appendChild(row);

    const thumbCont = document.getElementById(`ithumb-${cls.id}`);
    cls.images.slice(-4).forEach(img => {
      const el = document.createElement('img');
      el.className = 'cls-thumb'; el.src = img.src;
      thumbCont.appendChild(el);
    });

    document.getElementById(`ifile-${cls.id}`).onchange = e => {
      Array.from(e.target.files).forEach(f => {
        const r = new FileReader();
        r.onload = ev => { cls.images.push({src: ev.target.result}); renderInitialClasses(); };
        r.readAsDataURL(f);
      });
      e.target.value = '';
    };
  });
}

// ============================================================
// BUILD + TRAIN CLASSIFIER HEAD
// ============================================================
async function buildAndTrain(classes, imageMap, epochs, batchSize, progressBarId, epochTxtId, statsId, chartId, chartData) {
  const n = classes.length;
  const xs = [], ys = [];

  for (let ci = 0; ci < n; ci++) {
    const imgs = imageMap[classes[ci].id] || classes[ci].images || [];
    for (const img of imgs) {
      const feat = await extractFeatures(img.src);
      xs.push(feat); ys.push(ci);
    }
  }

  const xStack = tf.stack(xs);
  const flatSize = xStack.shape.slice(1).reduce((a,b)=>a*b,1);
  const xFlat = xStack.reshape([-1, flatSize]);
  const yTensor = tf.oneHot(tf.tensor1d(ys,'int32'), n);
  xStack.dispose(); xs.forEach(t=>t.dispose());

  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 256, activation:'relu', inputShape:[flatSize], kernelInitializer:'glorotNormal' }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.dropout({ rate: 0.4 }));
  model.add(tf.layers.dense({ units: 128, activation:'relu' }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: n, activation:'softmax' }));
  model.compile({ optimizer: tf.train.adam(0.0001), loss:'categoricalCrossentropy', metrics:['accuracy'] });

  const canvas = document.getElementById(chartId);
  const ctx = canvas ? canvas.getContext('2d') : null;

  await model.fit(xFlat, yTensor, {
    epochs, batchSize, shuffle: true, validationSplit: 0.2,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        if (STATE.stopRequested) { model.stopTraining = true; return; }
        const pct = ((epoch+1)/epochs)*100;
        document.getElementById(progressBarId).style.width = pct+'%';
        document.getElementById(epochTxtId).textContent = `Epoch ${epoch+1}/${epochs}`;
        document.getElementById(statsId).textContent =
          `Train: ${(logs.acc*100).toFixed(1)}%  |  Val: ${((logs.val_acc||0)*100).toFixed(1)}%  |  Loss: ${logs.loss.toFixed(4)}`;
        chartData.acc.push(logs.acc*100);
        chartData.valAcc.push((logs.val_acc||0)*100);
        if (ctx) drawAccChart(ctx, canvas.width, canvas.height, chartData, classes);
        await tf.nextFrame();
      }
    }
  });

  xFlat.dispose(); yTensor.dispose();
  return model;
}

// ============================================================
// TRAIN INITIAL MODEL
// ============================================================
async function trainInitialModel() {
  if (STATE.isTraining) return;
  if (STATE.initialClasses.length < 2) { toast('Add at least 2 classes', 'error'); return; }
  for (const cls of STATE.initialClasses) {
    if (cls.images.length < 2) { toast(`Need images for ${cls.name}`, 'error'); return; }
  }

  STATE.isTraining = true;
  STATE.stopRequested = false;
  setStatus('Training initial model...', 'training');
  document.getElementById('btn-train-initial').disabled = true;
  document.getElementById('init-progress').style.display = 'block';
  STATE.chartDataInitial = { acc: [], valAcc: [] };

  const epochs = parseInt(document.getElementById('init-epochs').value) || 20;
  const batchSize = parseInt(document.getElementById('init-batch').value) || 16;

  if (STATE.initialModel) STATE.initialModel.dispose();
  STATE.initialModel = await buildAndTrain(
    STATE.initialClasses, null, epochs, batchSize,
    'init-prog', 'init-epoch-txt', 'init-stats', 'init-chart', STATE.chartDataInitial
  );

  STATE.initialAcc = STATE.chartDataInitial.acc.at(-1);
  STATE.isTraining = false;
  setStatus('Initial model trained', 'ready');
  document.getElementById('btn-train-initial').disabled = false;

  const result = document.getElementById('init-result');
  result.style.display = 'block';
  result.innerHTML = `✅ Initial Model Trained — ${STATE.initialClasses.length} classes — Acc: ${STATE.initialAcc.toFixed(1)}%`;

  // Enable step 2
  const step2 = document.getElementById('step2');
  step2.style.opacity = '1';
  step2.style.pointerEvents = 'auto';

  toast(`Initial training done! Acc: ${STATE.initialAcc.toFixed(1)}%`, 'success');
}

// ============================================================
// INCREMENTAL TRAINING (add new class)
// ============================================================
async function addAndRetrain() {
  if (STATE.isTraining) return;
  if (!STATE.initialModel) { toast('Train initial model first', 'error'); return; }

  const newName = document.getElementById('new-class-name').value.trim();
  if (!newName) { toast('Enter new class name', 'error'); return; }
  if (STATE.newClassImages.length < 2) { toast('Add at least 2 images for new class', 'error'); return; }

  STATE.isTraining = true;
  STATE.stopRequested = false;
  setStatus('Retraining with new class...', 'training');

  // Build combined class list = initial + new class
  const newCls = { id: uid(), name: newName, color: STATE.selectedNewColor, images: STATE.newClassImages };
  STATE.incrementalClasses = [...STATE.initialClasses, newCls];

  document.getElementById('btn-add-retrain').disabled = true;
  document.getElementById('incr-progress').style.display = 'block';
  STATE.chartDataIncremental = { acc: [], valAcc: [] };

  const epochs = parseInt(document.getElementById('incr-epochs').value) || 20;
  const batchSize = parseInt(document.getElementById('incr-batch').value) || 16;

  if (STATE.incrementalModel) STATE.incrementalModel.dispose();
  STATE.incrementalModel = await buildAndTrain(
    STATE.incrementalClasses, null, epochs, batchSize,
    'incr-prog', 'incr-epoch-txt', 'incr-stats', 'incr-chart', STATE.chartDataIncremental
  );

  STATE.incrementalAcc = STATE.chartDataIncremental.acc.at(-1);
  STATE.isTraining = false;
  setStatus('Incremental model ready', 'ready');
  document.getElementById('btn-add-retrain').disabled = false;

  showComparison();
  showPredictArea();
  toast(`Retrained with "${newName}"! Acc: ${STATE.incrementalAcc.toFixed(1)}%`, 'success');
}

// ============================================================
// COMPARISON
// ============================================================
function showComparison() {
  const panel = document.getElementById('comparison-panel');
  panel.style.display = 'block';

  const diff = STATE.incrementalAcc - STATE.initialAcc;
  const isUp = diff >= 0;
  const diffStr = (isUp ? '+' : '') + diff.toFixed(1) + '%';

  document.getElementById('comparison-grid').innerHTML = `
    <div class="cmp-box">
      <div class="cmp-val" style="color:#7c3aed">${STATE.initialAcc.toFixed(1)}%</div>
      <div class="cmp-lbl">Initial Accuracy</div>
      <div class="cmp-lbl" style="margin-top:4px;font-size:12px">${STATE.initialClasses.length} classes</div>
    </div>
    <div class="cmp-box highlight">
      <div class="cmp-val" style="color:#059669">${STATE.incrementalAcc.toFixed(1)}%</div>
      <div class="cmp-lbl">Incremental Accuracy</div>
      <div class="cmp-lbl" style="margin-top:4px;font-size:12px">${STATE.incrementalClasses.length} classes</div>
    </div>
    <div class="cmp-box">
      <div class="cmp-val ${isUp?'':'cmp-diff down'}" style="${isUp?'color:#059669':'color:#ef4444'}">${diffStr}</div>
      <div class="cmp-lbl">Accuracy Change</div>
    </div>
    <div class="cmp-box">
      <div class="cmp-val">+${STATE.incrementalClasses.length - STATE.initialClasses.length}</div>
      <div class="cmp-lbl">New Classes Added</div>
    </div>`;

  // Comparison chart
  const canvas = document.getElementById('compare-chart');
  const ctx = canvas.getContext('2d');
  drawComparisonChart(ctx, canvas.width, canvas.height);
}

function drawComparisonChart(ctx, w, h) {
  ctx.clearRect(0,0,w,h);
  const pd = 36, cw = w-pd*2, ch = h-pd*2;
  ctx.fillStyle = '#f0ede6'; ctx.fillRect(0,0,w,h);

  // Combine both charts: show both training curves
  const init = STATE.chartDataInitial.acc;
  const incr = STATE.chartDataIncremental.acc;
  const initVal = STATE.chartDataInitial.valAcc;
  const incrVal = STATE.chartDataIncremental.valAcc;
  const maxLen = Math.max(init.length, incr.length);
  if (maxLen < 2) return;

  // Grid
  ctx.strokeStyle = 'rgba(0,0,0,0.08)'; ctx.lineWidth = 1;
  [0,25,50,75,100].forEach(v => {
    const y = pd + ch - (v/100)*ch;
    ctx.beginPath(); ctx.moveTo(pd,y); ctx.lineTo(pd+cw,y); ctx.stroke();
    ctx.fillStyle = 'rgba(0,0,0,0.3)'; ctx.font = '9px Space Mono';
    ctx.fillText(v+'%', 2, y+3);
  });

  function drawLine(data, color, dashed=false) {
    if (data.length < 2) return;
    ctx.beginPath();
    ctx.strokeStyle = color; ctx.lineWidth = 2.5;
    if (dashed) ctx.setLineDash([5,3]); else ctx.setLineDash([]);
    data.forEach((v,i) => {
      const x = pd + (i/(maxLen-1))*cw;
      const y = pd + ch - (v/100)*ch;
      i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }

  drawLine(init, '#7c3aed');
  drawLine(initVal, '#7c3aed', true);
  drawLine(incr, '#059669');
  drawLine(incrVal, '#059669', true);

  // Vertical separator
  if (init.length > 0 && incr.length > 0) {
    const sepX = pd + ((init.length-1)/(maxLen-1))*cw;
    ctx.strokeStyle = 'rgba(0,0,0,0.2)'; ctx.lineWidth = 1;
    ctx.setLineDash([3,3]);
    ctx.beginPath(); ctx.moveTo(sepX, pd); ctx.lineTo(sepX, pd+ch); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(0,0,0,0.4)'; ctx.font = '9px Space Mono';
    ctx.fillText('New class', sepX+4, pd+12);
  }

  // Legend
  const items = [
    {color:'#7c3aed', label:'Initial Train'},
    {color:'#059669', label:'Incremental Train'},
  ];
  items.forEach((item,i) => {
    ctx.fillStyle = item.color; ctx.fillRect(pd + i*130, 6, 14, 3);
    ctx.fillStyle = '#1c1917'; ctx.font = '10px Plus Jakarta Sans';
    ctx.fillText(item.label, pd + i*130 + 18, 12);
  });
}

// ============================================================
// PREDICTION
// ============================================================
function showPredictArea() {
  document.getElementById('predict-area').style.display = 'block';
}

async function predict(src) {
  if (!STATE.incrementalModel) { toast('Train incremental model first', 'error'); return; }
  const feat = await extractFeatures(src);
  const flat = feat.reshape([1, feat.shape.reduce((a,b)=>a*b,1)]);
  const preds = STATE.incrementalModel.predict(flat);
  const data = await preds.data();
  flat.dispose(); feat.dispose(); preds.dispose();

  const results = STATE.incrementalClasses.map((cls,i) => ({ ...cls, conf: data[i] }));
  results.sort((a,b) => b.conf - a.conf);
  const top = results[0];

  const wrap = document.getElementById('pred-bars-wrap');
  wrap.innerHTML = `<div class="pred-top-label" style="color:${top.color}">${top.name} — ${(top.conf*100).toFixed(1)}%</div>`;
  results.forEach(r => {
    const row = document.createElement('div');
    row.className = 'pred-bar-item';
    row.innerHTML = `
      <div class="pred-bar-lbl" style="color:${r.color}">${r.name}</div>
      <div class="pred-bar-track">
        <div class="pred-bar-fill" style="width:${(r.conf*100).toFixed(1)}%;background:${r.color}"></div>
      </div>
      <div class="pred-bar-pct">${(r.conf*100).toFixed(0)}%</div>`;
    wrap.appendChild(row);
  });
}

// ============================================================
// CHART
// ============================================================
function drawAccChart(ctx, w, h, chartData, classes) {
  const { acc, valAcc } = chartData;
  if (acc.length < 2) return;
  ctx.clearRect(0,0,w,h);
  const pd = 30, cw = w-pd*2, ch = h-pd*2;
  ctx.fillStyle = '#f0ede6'; ctx.fillRect(0,0,w,h);

  ctx.strokeStyle = 'rgba(0,0,0,0.07)'; ctx.lineWidth = 1;
  [0,25,50,75,100].forEach(v => {
    const y = pd + ch - (v/100)*ch;
    ctx.beginPath(); ctx.moveTo(pd,y); ctx.lineTo(pd+cw,y); ctx.stroke();
  });

  function line(data, color, dashed=false) {
    if (data.length < 2) return;
    ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 2;
    if (dashed) ctx.setLineDash([4,3]); else ctx.setLineDash([]);
    data.forEach((v,i) => {
      const x = pd+(i/(data.length-1))*cw;
      const y = pd+ch-(v/100)*ch;
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    });
    ctx.stroke(); ctx.setLineDash([]);
  }
  line(acc, '#7c3aed');
  line(valAcc, '#db2777', true);
  ctx.fillStyle='#7c3aed'; ctx.fillRect(pd,6,12,3);
  ctx.fillStyle='#1c1917'; ctx.font='10px Plus Jakarta Sans'; ctx.fillText('Train Acc', pd+16, 12);
  ctx.fillStyle='#db2777'; ctx.fillRect(pd+90,6,12,3);
  ctx.fillStyle='#1c1917'; ctx.fillText('Val Acc', pd+106, 12);
}

// ============================================================
// NEW CLASS COLOR SELECTOR
// ============================================================
STATE.selectedNewColor = PALETTE[3];
STATE.newClassImages = [];

function renderNewClassColorPick() {
  const cont = document.getElementById('new-class-color-pick');
  cont.innerHTML = '';
  PALETTE.forEach(c => {
    const dot = document.createElement('div');
    dot.className = `cpick ${c === STATE.selectedNewColor ? 'sel' : ''}`;
    dot.style.background = c;
    dot.onclick = () => {
      STATE.selectedNewColor = c;
      cont.querySelectorAll('.cpick').forEach(d=>d.classList.remove('sel'));
      dot.classList.add('sel');
    };
    cont.appendChild(dot);
  });
}

// ============================================================
// MODAL
// ============================================================
let modalSelectedColor = PALETTE[0];

function openAddClassModal() {
  const modal = document.getElementById('modal');
  modal.style.display = 'flex';
  document.getElementById('modal-class-name').value = '';
  modalSelectedColor = PALETTE[STATE.initialClasses.length % PALETTE.length];
  const colors = document.getElementById('modal-colors');
  colors.innerHTML = '';
  PALETTE.forEach(c => {
    const dot = document.createElement('div');
    dot.className = `cpick ${c===modalSelectedColor?'sel':''}`;
    dot.style.background = c;
    dot.onclick = () => {
      modalSelectedColor = c;
      colors.querySelectorAll('.cpick').forEach(d=>d.classList.remove('sel'));
      dot.classList.add('sel');
    };
    colors.appendChild(dot);
  });
  setTimeout(()=>document.getElementById('modal-class-name').focus(), 50);
}

// ============================================================
// INIT
// ============================================================
document.addEventListener('DOMContentLoaded', async () => {
  // Default initial classes
  addInitialClass('Apple', PALETTE[0]);
  addInitialClass('Banana', PALETTE[2]);
  addInitialClass('Orange', PALETTE[3]);

  renderNewClassColorPick();
  await loadMobileNet();

  document.getElementById('btn-add-initial').onclick = openAddClassModal;
  document.getElementById('modal-cancel').onclick = ()=>{ document.getElementById('modal').style.display='none'; };
  document.getElementById('modal-ok').onclick = () => {
    const name = document.getElementById('modal-class-name').value.trim();
    if (!name) { toast('Enter a name', 'error'); return; }
    addInitialClass(name, modalSelectedColor);
    document.getElementById('modal').style.display = 'none';
  };
  document.getElementById('modal-class-name').onkeydown = e => {
    if (e.key==='Enter') document.getElementById('modal-ok').click();
    if (e.key==='Escape') document.getElementById('modal-cancel').click();
  };

  document.getElementById('btn-train-initial').onclick = trainInitialModel;
  document.getElementById('btn-add-retrain').onclick = addAndRetrain;

  // New class files
  document.getElementById('new-class-files').onchange = e => {
    Array.from(e.target.files).forEach(f => {
      const r = new FileReader();
      r.onload = ev => {
        STATE.newClassImages.push({ src: ev.target.result });
        const cont = document.getElementById('new-class-thumbs');
        cont.innerHTML = '';
        STATE.newClassImages.slice(-6).forEach(img => {
          const el = document.createElement('img');
          el.className = 'ncls-thumb'; el.src = img.src;
          cont.appendChild(el);
        });
        if (STATE.newClassImages.length > 0) {
          cont.querySelector('.placeholder-txt') && (cont.querySelector('.placeholder-txt').style.display='none');
        }
      };
      r.readAsDataURL(f);
    });
    e.target.value = '';
  };

  // Prediction
  document.getElementById('pred-file').onchange = e => {
    const f = e.target.files[0]; if (!f) return;
    const r = new FileReader();
    r.onload = ev => {
      const img = document.getElementById('pred-img');
      img.src = ev.target.result; img.style.display='block';
      document.getElementById('pred-empty-msg').style.display='none';
    };
    r.readAsDataURL(f); e.target.value='';
  };
  document.getElementById('btn-predict').onclick = async () => {
    const src = document.getElementById('pred-img').src;
    if (!src || src===location.href) { toast('Upload an image', 'error'); return; }
    await predict(src);
  };
});
