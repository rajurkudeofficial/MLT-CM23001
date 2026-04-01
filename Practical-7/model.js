// ============================================================
// Practical 7 — TensorFlow.js Browser Model
// y = 2x + 1  |  Sequential regression model
// ============================================================

// ---- Global state ----
let originalModel = null;
let loadedModel   = null;
let lossChart     = null;

// ---- Training dataset: y = 2x + 1 ----
const XS = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5];
const YS = XS.map(x => 2 * x + 1);

// ---- Populate data table on load ----
window.addEventListener('DOMContentLoaded', () => {
  const tbody = document.getElementById('training-data-body');
  XS.forEach((x, i) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${x}</td><td>${YS[i]}</td>`;
    tbody.appendChild(tr);
  });
});

// ============================================================
// createModel() — build a simple 1-layer dense regression net
// ============================================================
function createModel() {
  const model = tf.sequential();

  // Single dense layer: input shape 1 → output 1
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  // Compile with Adam optimizer + MSE loss
  model.compile({
    optimizer: tf.train.adam(parseFloat(document.getElementById('lr').value)),
    loss: 'meanSquaredError'
  });

  return model;
}

// ============================================================
// trainModel() — train and update UI with progress + chart
// ============================================================
async function trainModel() {
  const epochs = parseInt(document.getElementById('epochs').value);

  // Reset UI
  setStatus('train-status', '', 'hide');
  document.getElementById('progress-wrap').style.display = 'flex';
  document.getElementById('loss-wrap').style.display = 'block';
  document.getElementById('train-btn').disabled = true;
  document.getElementById('save-btn').disabled = true;

  // Create model & tensors
  originalModel = createModel();
  const xs = tf.tensor2d(XS, [XS.length, 1]);
  const ys = tf.tensor2d(YS, [YS.length, 1]);

  const lossHistory = [];
  initLossChart();

  // Train
  await originalModel.fit(xs, ys, {
    epochs,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        const pct = Math.round(((epoch + 1) / epochs) * 100);
        document.getElementById('progress-fill').style.width = pct + '%';
        document.getElementById('progress-label').textContent =
          `Epoch ${epoch + 1} / ${epochs}  |  Loss: ${logs.loss.toFixed(6)}`;

        lossHistory.push(logs.loss);
        updateLossChart(lossHistory);
      }
    }
  });

  xs.dispose();
  ys.dispose();

  // Done
  document.getElementById('train-btn').disabled = false;
  document.getElementById('save-btn').disabled = false;
  setStatus('train-status', '✅ Model trained successfully! You can now save it.', 'success');
}

// ============================================================
// saveModel() — persist to LocalStorage
// ============================================================
async function saveModel() {
  if (!originalModel) {
    setStatus('save-status', '⚠️ No trained model found. Train first!', 'error');
    return;
  }

  try {
    await originalModel.save('localstorage://practical7-model');
    setStatus('save-status', '💾 Model saved to LocalStorage successfully!', 'success');
  } catch (err) {
    setStatus('save-status', '❌ Save failed: ' + err.message, 'error');
  }
}

// ============================================================
// loadModel() — reload from LocalStorage
// ============================================================
async function loadModel() {
  try {
    loadedModel = await tf.loadLayersModel('localstorage://practical7-model');
    setStatus('load-status', '📂 Model loaded from LocalStorage successfully!', 'success');
  } catch (err) {
    setStatus('load-status', '❌ Load failed — train and save first. ' + err.message, 'error');
  }
}

// ============================================================
// predict(model, x) — run inference and return scalar value
// ============================================================
function predict(model, xVal) {
  return tf.tidy(() => {
    const input = tf.tensor2d([xVal], [1, 1]);
    const output = model.predict(input);
    return output.dataSync()[0];
  });
}

// ============================================================
// runPredictions() — compare original vs loaded model
// ============================================================
function runPredictions() {
  const xVal = parseFloat(document.getElementById('pred-x').value);

  if (isNaN(xVal)) {
    alert('Please enter a valid number for x.');
    return;
  }

  const resultsEl = document.getElementById('pred-results');
  resultsEl.style.display = 'none';

  let origVal = null;
  let loadVal = null;

  if (originalModel) {
    origVal = predict(originalModel, xVal);
    document.getElementById('orig-pred').textContent = origVal.toFixed(4);
  } else {
    document.getElementById('orig-pred').textContent = 'N/A';
  }

  if (loadedModel) {
    loadVal = predict(loadedModel, xVal);
    document.getElementById('load-pred').textContent = loadVal.toFixed(4);
  } else {
    document.getElementById('load-pred').textContent = 'N/A';
  }

  if (origVal !== null && loadVal !== null) {
    const diff = Math.abs(origVal - loadVal);
    document.getElementById('diff-pred').textContent = diff.toFixed(6);
  } else {
    document.getElementById('diff-pred').textContent = '—';
  }

  resultsEl.style.display = 'grid';
}

// ============================================================
// Chart helpers — loss curve using Chart.js
// ============================================================
function initLossChart() {
  const ctx = document.getElementById('loss-chart').getContext('2d');
  if (lossChart) lossChart.destroy();

  lossChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Training Loss',
        data: [],
        borderColor: '#e85d26',
        backgroundColor: 'rgba(232,93,38,0.08)',
        borderWidth: 2,
        pointRadius: 0,
        fill: true,
        tension: 0.4
      }]
    },
    options: {
      responsive: true,
      animation: false,
      plugins: { legend: { display: true } },
      scales: {
        x: { display: false },
        y: { beginAtZero: true, ticks: { font: { family: 'DM Mono' } } }
      }
    }
  });
}

function updateLossChart(history) {
  lossChart.data.labels = history.map((_, i) => i + 1);
  lossChart.data.datasets[0].data = history;
  lossChart.update('none');
}

// ============================================================
// UI utility — show/hide status messages
// ============================================================
function setStatus(id, msg, type) {
  const el = document.getElementById(id);
  if (type === 'hide') { el.style.display = 'none'; return; }
  el.textContent = msg;
  el.className = `status ${type}`;
  el.style.display = 'block';
}
