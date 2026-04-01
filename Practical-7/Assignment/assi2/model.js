// ============================================================
// Assignment 2 — Load model from LocalStorage, predict, compare
// ============================================================

let savedModel = null;   // loaded from LocalStorage (assi1-model)
let freshModel  = null;  // trained fresh in this session

const XS_DATA = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9];
const YS_DATA = XS_DATA.map(x => 2 * x + 1);
const BATCH_XS = [-5, -2, 0, 3, 7, 10, 15];

// ============================================================
// loadSavedModel() — load from LocalStorage key: assi1-model
// ============================================================
async function loadSavedModel() {
  try {
    savedModel = await tf.loadLayersModel('localstorage://assi1-model');
    setDot('saved-dot', 'dot-green');
    document.getElementById('saved-small').textContent = 'Loaded from LocalStorage';
    setStatus('load-status', 'Saved model loaded from LocalStorage successfully!', 'success');
  } catch (err) {
    setDot('saved-dot', 'dot-grey');
    setStatus('load-status', 'Could not load saved model. Did you run Assignment 1 first? ' + err.message, 'error');
  }
}

// ============================================================
// trainFreshModel() — quick retrain for comparison baseline
// ============================================================
async function trainFreshModel() {
  setStatus('load-status', 'Training fresh model (100 epochs)...', 'info');
  setDot('fresh-dot', 'dot-yellow');

  freshModel = tf.sequential();
  freshModel.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  freshModel.compile({ optimizer: tf.train.adam(0.1), loss: 'meanSquaredError' });

  const xs = tf.tensor2d(XS_DATA, [XS_DATA.length, 1]);
  const ys = tf.tensor2d(YS_DATA, [YS_DATA.length, 1]);

  await freshModel.fit(xs, ys, { epochs: 100, shuffle: true });

  xs.dispose();
  ys.dispose();

  setDot('fresh-dot', 'dot-green');
  document.getElementById('fresh-small').textContent = 'Trained in-browser';
  setStatus('load-status', 'Fresh model trained successfully!', 'success');
}

// ============================================================
// predict(model, x) — single prediction
// ============================================================
function predict(model, xVal) {
  return tf.tidy(() => {
    const input = tf.tensor2d([xVal], [1, 1]);
    return model.predict(input).dataSync()[0];
  });
}

// ============================================================
// runPredictions() — compare saved vs fresh for one x
// ============================================================
function runPredictions() {
  const xVal = parseFloat(document.getElementById('pred-x').value);
  if (isNaN(xVal)) { alert('Please enter a valid number.'); return; }

  const ideal = 2 * xVal + 1;
  document.getElementById('ideal-val').textContent = ideal.toFixed(4);

  const grid = document.getElementById('pred-grid');
  grid.style.display = 'none';

  let savedVal = null, freshVal = null;

  if (savedModel) {
    savedVal = predict(savedModel, xVal);
    document.getElementById('saved-pred').textContent = savedVal.toFixed(4);
    document.getElementById('saved-err').textContent = `error vs ideal: ${Math.abs(savedVal - ideal).toFixed(4)}`;
  } else {
    document.getElementById('saved-pred').textContent = 'N/A';
    document.getElementById('saved-err').textContent = 'Load model first';
  }

  if (freshModel) {
    freshVal = predict(freshModel, xVal);
    document.getElementById('fresh-pred').textContent = freshVal.toFixed(4);
    document.getElementById('fresh-err').textContent = `error vs ideal: ${Math.abs(freshVal - ideal).toFixed(4)}`;
  } else {
    document.getElementById('fresh-pred').textContent = 'N/A';
    document.getElementById('fresh-err').textContent = 'Train model first';
  }

  if (savedVal !== null && freshVal !== null) {
    document.getElementById('diff-val').textContent = Math.abs(savedVal - freshVal).toFixed(6);
  } else {
    document.getElementById('diff-val').textContent = '—';
  }

  grid.style.display = 'grid';
}

// ============================================================
// runBatch() — batch comparison table
// ============================================================
function runBatch() {
  if (!savedModel && !freshModel) {
    alert('Load or train at least one model first.');
    return;
  }

  const tbody = document.getElementById('batch-body');
  tbody.innerHTML = '';

  BATCH_XS.forEach(x => {
    const ideal = 2 * x + 1;
    const sVal = savedModel ? predict(savedModel, x).toFixed(4) : 'N/A';
    const fVal = freshModel ? predict(freshModel, x).toFixed(4) : 'N/A';
    const diff = (savedModel && freshModel)
      ? Math.abs(parseFloat(sVal) - parseFloat(fVal)).toFixed(6)
      : '—';

    const diffNum = parseFloat(diff);
    const diffClass = diffNum < 0.01 ? 'good' : diffNum < 0.1 ? 'ok' : '';

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${x}</td>
      <td>${ideal}</td>
      <td>${sVal}</td>
      <td>${fVal}</td>
      <td class="${diffClass}">${diff}</td>`;
    tbody.appendChild(tr);
  });

  document.getElementById('batch-wrap').style.display = 'block';
}

// ---- update ideal live ----
function updateIdeal() {
  const xVal = parseFloat(document.getElementById('pred-x').value);
  document.getElementById('ideal-val').textContent = isNaN(xVal) ? '—' : (2 * xVal + 1).toFixed(4);
}

// ---- UI helpers ----
function setDot(id, cls) {
  const el = document.getElementById(id);
  el.className = 'dot ' + cls;
}
function setStatus(id, msg, type) {
  const el = document.getElementById(id);
  if (type === 'hide') { el.style.display = 'none'; return; }
  el.textContent = msg;
  el.className = `status ${type}`;
  el.style.display = 'block';
}
