// ============================================================
// Assignment 1 — Train a model and save to LocalStorage
// Model: y = 2x + 1 (linear regression)
// Save key: localstorage://assi1-model
// ============================================================

let model = null;
let lossChart = null;

// ---- Extended dataset: y = 2x + 1 (20 points) ----
const XS_DATA = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9];
const YS_DATA = XS_DATA.map(x => 2 * x + 1);

// ============================================================
// createModel() — 1 Dense layer, linear activation
// ============================================================
function createModel() {
  const lr = parseFloat(document.getElementById('lr').value) || 0.1;
  const m = tf.sequential();
  m.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  m.compile({ optimizer: tf.train.adam(lr), loss: 'meanSquaredError' });
  return m;
}

// ============================================================
// trainModel() — fit model on hardcoded dataset
// ============================================================
async function trainModel() {
  const epochs = parseInt(document.getElementById('epochs').value) || 150;

  document.getElementById('train-btn').disabled = true;
  document.getElementById('save-btn').disabled = true;
  document.getElementById('progress-wrap').style.display = 'flex';
  document.getElementById('loss-wrap').style.display = 'block';
  setStatus('train-status', '', 'hide');

  model = createModel();
  const xs = tf.tensor2d(XS_DATA, [XS_DATA.length, 1]);
  const ys = tf.tensor2d(YS_DATA, [YS_DATA.length, 1]);
  const lossHistory = [];
  initChart();

  await model.fit(xs, ys, {
    epochs,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        const pct = Math.round(((epoch + 1) / epochs) * 100);
        document.getElementById('progress-fill').style.width = pct + '%';
        document.getElementById('progress-label').textContent =
          `Epoch ${epoch + 1} / ${epochs}  —  Loss: ${logs.loss.toFixed(6)}`;
        lossHistory.push(logs.loss);
        updateChart(lossHistory);
      }
    }
  });

  xs.dispose();
  ys.dispose();
  document.getElementById('train-btn').disabled = false;
  document.getElementById('save-btn').disabled = false;

  const weights = model.layers[0].getWeights();
  const w = weights[0].dataSync()[0].toFixed(4);
  const b = weights[1].dataSync()[0].toFixed(4);
  setStatus('train-status', `Training complete! Learned: y ≈ ${w}x + ${b}  (ideal: 2x + 1)`, 'success');
}

// ============================================================
// saveModel() — save to LocalStorage with confirmation
// ============================================================
async function saveModel() {
  if (!model) {
    setStatus('save-status', 'No model to save. Train first.', 'error');
    return;
  }
  try {
    await model.save('localstorage://assi1-model');
    setStatus('save-status', 'Model saved to LocalStorage under key: assi1-model', 'success');
    document.getElementById('key-info').style.display = 'block';
  } catch (err) {
    setStatus('save-status', 'Save failed: ' + err.message, 'error');
  }
}

// ============================================================
// Chart helpers
// ============================================================
function initChart() {
  const ctx = document.getElementById('loss-chart').getContext('2d');
  if (lossChart) lossChart.destroy();
  lossChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'MSE Loss',
        data: [],
        borderColor: '#7c3aed',
        backgroundColor: 'rgba(124,58,237,0.08)',
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
        y: { beginAtZero: true, ticks: { font: { family: 'DM Mono', size: 10 } } }
      }
    }
  });
}

function updateChart(history) {
  lossChart.data.labels = history.map((_, i) => i + 1);
  lossChart.data.datasets[0].data = history;
  lossChart.update('none');
}

function setStatus(id, msg, type) {
  const el = document.getElementById(id);
  if (type === 'hide') { el.style.display = 'none'; return; }
  el.textContent = msg;
  el.className = `status ${type}`;
  el.style.display = 'block';
}
