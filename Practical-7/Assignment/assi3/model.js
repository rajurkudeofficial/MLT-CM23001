// ============================================================
// Assignment 3 — Export model to files, re-upload, predict
// Uses tf.io.browserDownloads + tf.io.browserFiles
// ============================================================

let originalModel  = null;   // trained in-session
let importedModel  = null;   // loaded from uploaded files
let uploadedFiles  = null;   // FileList from input

const XS_DATA = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9];
const YS_DATA = XS_DATA.map(x => 2 * x + 1);

// ============================================================
// trainAndExport() — train model then trigger file downloads
// ============================================================
async function trainAndExport() {
  document.getElementById('train-btn').disabled = true;
  document.getElementById('progress-wrap').style.display = 'flex';
  setStatus('train-status', '', 'hide');

  // Build model
  originalModel = tf.sequential();
  originalModel.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  originalModel.compile({ optimizer: tf.train.adam(0.1), loss: 'meanSquaredError' });

  const xs = tf.tensor2d(XS_DATA, [XS_DATA.length, 1]);
  const ys = tf.tensor2d(YS_DATA, [YS_DATA.length, 1]);
  const epochs = 150;

  await originalModel.fit(xs, ys, {
    epochs,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        const pct = Math.round(((epoch + 1) / epochs) * 100);
        document.getElementById('progress-fill').style.width = pct + '%';
        document.getElementById('progress-label').textContent =
          `Epoch ${epoch + 1} / ${epochs}  —  Loss: ${logs.loss.toFixed(6)}`;
      }
    }
  });

  xs.dispose();
  ys.dispose();

  document.getElementById('train-btn').disabled = false;

  const w = originalModel.layers[0].getWeights()[0].dataSync()[0].toFixed(4);
  const b = originalModel.layers[0].getWeights()[1].dataSync()[0].toFixed(4);
  setStatus('train-status', `Model trained! Learned: y ≈ ${w}x + ${b}`, 'success');

  // Show download area
  document.getElementById('download-area').style.display = 'block';
}

// ============================================================
// downloadJson() — save model topology JSON
// ============================================================
async function downloadJson() {
  if (!originalModel) return;
  // Save topology only (JSON)
  const modelJSON = originalModel.toJSON();
  const blob = new Blob([JSON.stringify(modelJSON)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'model.json'; a.click();
  URL.revokeObjectURL(url);
}

// ============================================================
// downloadBin() — save model weights as binary
// Using tf's built-in download handler
// ============================================================
async function downloadBin() {
  if (!originalModel) return;
  // Use TF's download handler to get both files; we capture weights manually
  const weightData = await originalModel.getWeights();
  const weightSpecs = [];
  const buffers = [];

  for (const w of weightData) {
    const data = w.dataSync();
    const buffer = new Float32Array(data).buffer;
    buffers.push(buffer);
    weightSpecs.push({ name: w.name, shape: w.shape, dtype: w.dtype });
  }

  // Concatenate all weight buffers
  const totalBytes = buffers.reduce((s, b) => s + b.byteLength, 0);
  const combined = new Uint8Array(totalBytes);
  let offset = 0;
  for (const buf of buffers) {
    combined.set(new Uint8Array(buf), offset);
    offset += buf.byteLength;
  }

  const blob = new Blob([combined.buffer], { type: 'application/octet-stream' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'weights.bin'; a.click();
  URL.revokeObjectURL(url);
}

// ============================================================
// handleFileUpload() — track uploaded files
// ============================================================
function handleFileUpload(files) {
  uploadedFiles = files;
  const listEl = document.getElementById('file-list');
  listEl.innerHTML = '';
  Array.from(files).forEach(f => {
    const tag = document.createElement('span');
    tag.className = 'file-tag';
    tag.textContent = f.name;
    listEl.appendChild(tag);
  });
  document.getElementById('load-files-btn').disabled = files.length === 0;
}

// ============================================================
// loadFromFiles() — rebuild model from uploaded JSON + BIN
// ============================================================
async function loadFromFiles() {
  if (!uploadedFiles || uploadedFiles.length === 0) {
    setStatus('import-status', 'No files selected.', 'error');
    return;
  }

  // Find json and bin files from the upload
  const jsonFile = Array.from(uploadedFiles).find(f => f.name.endsWith('.json'));
  const binFile  = Array.from(uploadedFiles).find(f => f.name.endsWith('.bin'));

  if (!jsonFile) {
    setStatus('import-status', 'model.json file not found in upload.', 'error');
    return;
  }

  try {
    // Load using tf.io.browserFiles — handles both topology + weights
    if (binFile) {
      importedModel = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, binFile]));
    } else {
      // JSON-only (weights embedded)
      importedModel = await tf.loadLayersModel(tf.io.browserFiles([jsonFile]));
    }

    importedModel.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
    setStatus('import-status', 'Model imported from files successfully!', 'success');
  } catch (err) {
    setStatus('import-status', 'Import failed: ' + err.message, 'error');
  }
}

// ============================================================
// predict(model, x)
// ============================================================
function predict(model, xVal) {
  return tf.tidy(() => {
    const input = tf.tensor2d([xVal], [1, 1]);
    return model.predict(input).dataSync()[0];
  });
}

// ============================================================
// runPredictions() — original vs imported
// ============================================================
function runPredictions() {
  const xVal = parseFloat(document.getElementById('pred-x').value);
  if (isNaN(xVal)) { alert('Enter a valid number.'); return; }

  const ideal = 2 * xVal + 1;
  document.getElementById('ideal-pred').textContent = ideal.toFixed(4);

  const grid = document.getElementById('pred-grid');
  grid.style.display = 'none';

  let origVal = null, impVal = null;

  if (originalModel) {
    origVal = predict(originalModel, xVal);
    document.getElementById('orig-pred').textContent = origVal.toFixed(4);
  } else {
    document.getElementById('orig-pred').textContent = 'N/A (train first)';
  }

  if (importedModel) {
    impVal = predict(importedModel, xVal);
    document.getElementById('imported-pred').textContent = impVal.toFixed(4);
  } else {
    document.getElementById('imported-pred').textContent = 'N/A (import first)';
  }

  if (origVal !== null && impVal !== null) {
    document.getElementById('diff-pred').textContent = Math.abs(origVal - impVal).toFixed(6);
  } else {
    document.getElementById('diff-pred').textContent = '—';
  }

  grid.style.display = 'grid';
}

// ---- UI helper ----
function setStatus(id, msg, type) {
  const el = document.getElementById(id);
  if (type === 'hide') { el.style.display = 'none'; return; }
  el.textContent = msg;
  el.className = `status ${type}`;
  el.style.display = 'block';
}
