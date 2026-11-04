'use strict';
(() => {
  const D = JSON.parse(document.getElementById('project-data').textContent);
  const $ = s => document.querySelector(s);
  const $$ = s => [...document.querySelectorAll(s)];
  const scenes = $$('.scene');
  const fmt = (v, n = 2) => new Intl.NumberFormat('en-US', { style: 'percent', minimumFractionDigits: n, maximumFractionDigits: n }).format(v);
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const top = probabilities => probabilities.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p || a.i - b.i);
  let current = -1, selectedDemo = 4, selectedClass = 6, errorModel = 'CNN', matrixOpen = false;
  let leavingTimer;
  const selectedChannels = { '2': 0, '5': 0 };
  const noteMap = [0, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11];
  const knownSources = [
    'notebooks/01-data-exploration.ipynb',
    'data/splits/train-validation-v1.json',
    'data/splits/train-validation-v1.npz',
    'results/logistic-regression-regularization/comparison.csv',
    'results/knn-comparison/comparison.csv',
    'results/cnn-improvements/comparison.csv',
    'results/cnn-improvements/wider-history.csv',
    'results/cnn-stability/summary.csv',
    'results/final-evaluation-v1/metrics.csv',
    'results/final-evaluation-v1/per-class-metrics.csv',
    'results/final-evaluation-v1/cnn-confusion-counts.csv',
    'results/final-evaluation-v1/predictions.csv',
    'results/final-evaluation-v1/class-counts.csv',
    'results/final-evaluation-v1/sampled-errors.csv',
    'docs/final-evaluation-protocol.md',
    'scripts/final_evaluation.py',
    'presentation/editorial/evidence-manifest.json',
  ];

  function announce(message) { $('#status').textContent = message; }
  function closeAll() { $$('dialog[open]').forEach(d => d.close()); }
  function openDialog(id) { closeAll(); $(id).showModal(); }
  function go(index, hash = true) {
    index = Math.max(0, Math.min(scenes.length - 1, index));
    if (index === current) return;
    clearTimeout(leavingTimer);
    const moveFocus = current >= 0 && scenes[current].contains(document.activeElement);
    if (moveFocus) document.activeElement.blur();
    scenes.forEach((scene, i) => {
      scene.classList.toggle('is-exiting', i === current && i !== index);
      scene.classList.toggle('is-active', i === index);
      scene.inert = i !== index;
      scene.setAttribute('aria-hidden', String(i !== index));
      if (i === index) scene.scrollTop = 0;
    });
    current = index;
    if (moveFocus) {
      const heading = scenes[index].querySelector('h1,h2');
      heading.tabIndex = -1;
      heading.focus({ preventScroll: true });
    }
    leavingTimer = setTimeout(() => scenes.forEach(s => s.classList.remove('is-exiting')), 650);
    document.body.dataset.scene = String(index + 1);
    $('#previous').disabled = index === 0;
    $('#next').disabled = index === scenes.length - 1;
    if (hash) history.replaceState(null, '', `#${index + 1}`);
    document.title = `${scenes[index].dataset.title} — Fashion-MNIST`;
    announce(`Scene ${index + 1} of ${scenes.length}: ${scenes[index].dataset.title}`);
  }

  // All thumbnail pixels come from the original dataset, embedded in this file.
  const orbit = document.createElement('div'); orbit.className = 'image-orbit';
  for (let i = 0; i < 20; i++) {
    const sample = D.corpus[(i * 7) % D.corpus.length];
    const img = document.createElement('img'); img.src = sample.image; img.alt = ''; img.style.setProperty('--i', i);
    orbit.append(img);
  }
  $('#hero-art').append(orbit);

  function inspectImage(image, title, description) {
    $('#image-title').textContent = title;
    $('#image-detail').src = image;
    $('#image-detail').alt = title;
    $('#image-description').textContent = description;
    openDialog('#image-dialog');
  }
  function renderDataset(category = -1) {
    const items = category < 0
      ? [...D.classes.map((_, i) => D.corpus.find(r => r.label === i)), ...D.classes.slice(0, 6).map((_, i) => D.corpus.filter(r => r.label === i)[1])]
      : D.corpus.filter(r => r.label === category);
    const wall = $('#dataset-wall'); wall.replaceChildren();
    items.forEach(r => {
      const b = document.createElement('button'); b.className = 'dataset-thumb';
      b.innerHTML = `<img src="${r.image}" alt="${esc(D.classes[r.label])}, validation image ${r.id}"><span>${esc(D.classes[r.label])}</span>`;
      b.addEventListener('click', () => inspectImage(r.image, D.classes[r.label], `Real Fashion-MNIST validation image, source index ${r.id}. Original dimensions: 28 × 28 grayscale pixels. Enlarged using nearest-pixel rendering.`));
      wall.append(b);
    });
    $$('#dataset-selector button').forEach(b => b.setAttribute('aria-pressed', String(Number(b.dataset.category) === category)));
  }
  [-1, ...D.classes.map((_, i) => i)].forEach(i => {
    const b = document.createElement('button'); b.textContent = i < 0 ? 'All categories' : D.classes[i];
    b.dataset.category = i; b.addEventListener('click', () => renderDataset(i)); $('#dataset-selector').append(b);
  });
  renderDataset();
  // Proportions and counts come from the saved split and original labels.
  const shortClasses = ['Top', 'Trou.', 'Pull.', 'Dress', 'Coat', 'Sand.', 'Shirt', 'Sneak.', 'Bag', 'Boot'];
  $('#fold-strip').innerHTML = Array.from({ length: D.split_details.n_splits }, (_, i) => `<div class="fold-cell${i === D.split_details.validation_fold ? ' validation-fold' : ''}" style="--i:${i}"><b>Fold ${i}</b><span>${i === D.split_details.validation_fold ? 'Validation' : 'Training'}</span></div>`).join('');
  D.split_details.subsets.forEach((subset, subsetIndex) => {
    const row = document.createElement('div'); row.className = 'balance-row';
    row.innerHTML = `<div class="balance-row-label"><strong>${subset.name}</strong><span>${subset.total.toLocaleString('en-US')} images · ${subset.per_class[0].toLocaleString('en-US')} per class</span></div><div class="balance-segments" role="group" aria-label="${subset.name}: ten classes, each ten percent">${D.classes.map((name, i) => `<button class="balance-segment" data-balance-class="${i}" style="--i:${i + subsetIndex * 3}" title="${esc(name)}: ${subset.per_class[i].toLocaleString('en-US')} images, ${fmt(subset.proportions[i], 0)} of ${subset.name.toLowerCase()}" aria-label="${esc(name)}, ${subset.per_class[i]} images, ${fmt(subset.proportions[i], 0)} of ${subset.name.toLowerCase()}">${shortClasses[i]}</button>`).join('')}</div>`;
    $('#balance-breakdown').append(row);
  });
  function highlightBalance(category) {
    $$('[data-balance-class]').forEach(b => { const chosen = Number(b.dataset.balanceClass) === category; b.classList.toggle('highlight', chosen); b.setAttribute('aria-pressed', String(chosen)); });
    $('#balance-detail').textContent = `${D.classes[category]}: ` + D.split_details.subsets.map(s => `${s.per_class[category].toLocaleString('en-US')} / ${s.total.toLocaleString('en-US')} = ${fmt(s.proportions[category], 0)}`).join(' · ');
  }
  $$('[data-balance-class]').forEach(b => {
    b.addEventListener('pointerenter', () => highlightBalance(Number(b.dataset.balanceClass)));
    b.addEventListener('focus', () => highlightBalance(Number(b.dataset.balanceClass)));
    b.addEventListener('click', () => highlightBalance(Number(b.dataset.balanceClass)));
  });
  highlightBalance(6);

  function drawPixels(highlight = null) {
    const demo = D.demos[selectedDemo], canvas = $('#pixel-canvas'), ctx = canvas.getContext('2d');
    const cell = canvas.width / 28;
    demo.pixels.forEach((v, i) => {
      ctx.fillStyle = `rgb(${v},${v},${v})`; ctx.fillRect((i % 28) * cell, Math.floor(i / 28) * cell, cell, cell);
    });
    ctx.strokeStyle = '#ffffff12'; ctx.lineWidth = .7;
    for (let i = 1; i < 28; i++) { ctx.beginPath(); ctx.moveTo(i * cell, 0); ctx.lineTo(i * cell, canvas.height); ctx.stroke(); ctx.beginPath(); ctx.moveTo(0, i * cell); ctx.lineTo(canvas.width, i * cell); ctx.stroke(); }
    if (highlight !== null) {
      ctx.strokeStyle = '#c3ce9c'; ctx.lineWidth = 4;
      ctx.strokeRect((highlight % 28) * cell + 1, Math.floor(highlight / 28) * cell + 1, cell - 2, cell - 2);
    }
    const vctx = $('#vector-canvas').getContext('2d'), w = $('#vector-canvas').width, h = $('#vector-canvas').height;
    vctx.clearRect(0, 0, w, h); vctx.strokeStyle = '#c3ce9c'; vctx.lineWidth = 1;
    vctx.beginPath(); demo.pixels.forEach((v, i) => { const x = i / 783 * w, y = h - 5 - v / 255 * (h - 10); if (i === 0) vctx.moveTo(x, y); else vctx.lineTo(x, y); }); vctx.stroke();
  }
  $('#pixel-canvas').addEventListener('pointermove', e => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.max(0, Math.min(27, Math.floor((e.clientX - rect.left) / rect.width * 28)));
    const y = Math.max(0, Math.min(27, Math.floor((e.clientY - rect.top) / rect.height * 28)));
    const i = y * 28 + x, value = D.demos[selectedDemo].pixels[i];
    $('#pixel-raw').textContent = value; $('#pixel-scaled').textContent = (value / 255).toFixed(3);
    $('#pixel-readout').textContent = `Row ${y + 1}, column ${x + 1} · feature ${i + 1} of 784`;
    drawPixels(i);
  });
  $('#pixel-canvas').addEventListener('pointerleave', () => drawPixels());
  $('#representation-button').addEventListener('click', e => {
    const flat = e.currentTarget.getAttribute('aria-pressed') !== 'true';
    e.currentTarget.setAttribute('aria-pressed', String(flat));
    e.currentTarget.closest('.representation').classList.toggle('is-flat', flat);
    e.currentTarget.textContent = flat ? 'Restore the spatial grid' : 'Flatten into 784 features';
    $('#representation-label').textContent = flat ? 'kNN / logistic regression: the 784 pixels in fixed row order.' : 'CNN: preserve the 28 × 28 spatial grid.';
  });

  function probabilityRows(container, probs, count = 4) {
    container.innerHTML = top(probs).slice(0, count).map(({ p, i }) => `<div class="prob-row"><span>${esc(D.classes[i])}</span><div class="prob-bar"><span style="--width:${p * 100}%"></span></div><span>${fmt(p, 1)}</span></div>`).join('') + `<p class="micro">Top ${count} of 10 classes · recorded output</p>`;
  }
  function renderDemos() {
    const d = D.demos[selectedDemo], actual = D.classes[d.label];
    $$('.demo-selector button').forEach(b => b.setAttribute('aria-pressed', String(Number(b.dataset.demo) === selectedDemo)));
    $$('[data-demo-label]').forEach(p => p.textContent = `Recorded validation example #${d.id} · True class: ${actual} · Same input across all three models`);
    ['knn-input', 'lr-input', 'cnn-input'].forEach(id => { $('#' + id).src = d.image; $('#' + id).alt = `${actual}, validation source index ${d.id}`; });
    $('#knn-input-label').textContent = `Actual: ${actual}\nImage #${d.id}`;
    $('#neighbours').innerHTML = d.neighbours.map((n, i) => `<div class="neighbour" style="--i:${i}"><img src="${n.image}" alt="Training neighbour ${i + 1}: ${esc(D.classes[n.label])}"><h4>${esc(D.classes[n.label])}</h4><p>Distance ${n.distance.toFixed(3)}<br>Training image #${n.id}</p></div>`).join('');
    const votes = new Map(); d.neighbours.forEach(n => votes.set(n.label, (votes.get(n.label) || 0) + 1));
    $('#knn-result').textContent = D.classes[top(d.knn_probabilities)[0].i];
    $('#knn-votes').innerHTML = [...votes].sort((a, b) => b[1] - a[1]).map(([label, n]) => `${n} / 3 → ${esc(D.classes[label])}`).join('<br>');
    $('#lr-heat').src = d.lr_contribution_image;
    $('#lr-heat-label').textContent = `Pixel × weight for “${D.classes[d.lr_contribution_class]}”`;
    $('#lr-equation').textContent = `${d.lr_contribution_sum.toFixed(3)} ${d.lr_bias < 0 ? '−' : '+'} ${Math.abs(d.lr_bias).toFixed(3)} ≈ ${d.lr_logits[d.lr_contribution_class].toFixed(3)}`;
    probabilityRows($('#lr-probabilities'), d.lr_probabilities);
    probabilityRows($('#cnn-probabilities'), d.cnn_probabilities, 3);
    [['cnn-maps-1', '2'], ['cnn-maps-2', '5']].forEach(([id, layer]) => {
      const grid = $('#' + id); grid.replaceChildren();
      const plate = document.createElement('button'); plate.className = 'activation-plate';
      const img = document.createElement('img'); plate.append(img); grid.append(plate);
      const chooser = document.createElement('label'); chooser.className = 'channel-chooser'; chooser.append('Channel');
      const picker = document.createElement('select'); picker.setAttribute('aria-label', `Recorded channel for convolution ${layer === '2' ? 1 : 2}`); chooser.append(picker);
      function selectChannel(channel) {
        selectedChannels[layer] = channel;
        img.src = d.feature_maps[layer][channel];
        img.alt = `Actual pooled feature map, layer ${layer === '2' ? 1 : 2}, channel ${channel + 1}`;
        plate.setAttribute('aria-label', `Enlarge convolution ${layer === '2' ? 1 : 2}, channel ${channel + 1}`);
        picker.value = String(channel);
        if (!matchMedia('(prefers-reduced-motion: reduce)').matches && grid.closest('.is-active')) img.animate([{ opacity: .3 }, { opacity: 1 }], { duration: 200, easing: 'ease-out' });
      }
      d.feature_maps[layer].forEach((_, i) => {
        const option = document.createElement('option'); option.textContent = String(i + 1); option.value = String(i); picker.append(option);
      });
      picker.addEventListener('change', () => selectChannel(Number(picker.value)));
      plate.addEventListener('click', () => inspectImage(img.src, `Convolution ${layer === '2' ? 1 : 2} · channel ${selectedChannels[layer] + 1}`, `Recorded activation after ReLU and max pooling for validation image #${d.id}. ${layer === '2' ? '14 × 14' : '7 × 7'} values. Common intensity scale across the original tensor; not original pixel intensity.`));
      grid.append(chooser); selectChannel(selectedChannels[layer]);
    });
    drawPixels();
  }
  $$('[data-demo-selector]').forEach(group => D.classes.forEach((name, i) => {
    const b = document.createElement('button'); b.dataset.demo = i; b.textContent = name;
    b.addEventListener('click', () => { selectedDemo = i; renderDemos(); announce(`Validation example ${D.demos[i].id}, actual ${name}`); }); group.append(b);
  }));
  renderDemos();
  $('[data-replay="knn"]').addEventListener('click', () => {
    $$('.neighbour, .vote-result').forEach(el => { el.style.animation = 'none'; void el.offsetWidth; el.style.animation = ''; });
  });

  function renderTuning(kind) {
    $$('[data-tuning]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.tuning === kind)));
    let rows, explanation;
    if (kind === 'lr') {
      rows = D.logistic.map(r => ({ label: `C = ${r.C}`, score: r.validation_accuracy, chosen: r.C === .1 }));
      explanation = '<h3>C = 0.1</h3><p>Best validation accuracy among four settings.<br><br>Smaller C means stronger L2 regularisation. C = 10 raised training accuracy but reduced validation performance.</p><p class="micro">Only 10 images ahead of C = 1. A modest validation preference.</p>';
    } else if (kind === 'knn') {
      rows = D.knn.map(r => ({ label: `k = ${r.k}`, score: r.validation_accuracy, chosen: r.k === 3 }));
      explanation = '<h3>k = 3</h3><p>The smallest tested neighbourhood gave the highest validation accuracy.<br><br>Euclidean distance. Uniform voting. Same 48,000 training observations.</p><p class="micro">Only 7 images ahead of k = 5. Not a universal optimum.</p>';
    } else {
      rows = D.cnn.map(r => ({ label: ({ original_baseline: 'Original', original_extended: 'Extended', wider: 'Wider', wider_weight_decay: '+ Weight decay' })[r.name], score: r.validation_accuracy, chosen: r.name === 'wider' }));
      explanation = '<h3>Wider CNN</h3><p>32 / 64 convolutional filters and a 128-unit hidden layer.<br><br>Two architecture widths. Eight training runs overall, including the seed study.</p><p class="micro">Across 3 seeds: 92.400% without weight decay vs 92.339% with it. Weight decay was more stable; keep the original selected seed-42 checkpoint.</p>';
    }
    $('#tuning-chart').innerHTML = rows.map(r => `<div class="tune-row${r.chosen ? ' selected' : ''}"><span>${r.label}</span><div class="tune-track"><span style="--width:${r.score * 100}%"></span></div><span>${fmt(r.score)}</span></div>`).join('') + '<p class="tune-note">Validation accuracy · bar scale 0–100%</p>';
    $('#tuning-explanation').innerHTML = explanation;
  }
  $$('[data-tuning]').forEach(b => b.addEventListener('click', () => renderTuning(b.dataset.tuning)));
  renderTuning('lr');

  function learningChart() {
    const narrow = window.innerWidth <= 600;
    const W = narrow ? 420 : 760, H = 340, left = narrow ? 48 : 62, right = 16, bottom = 65, upper = 34;
    const textSize = narrow ? 18 : 16;
    const x = epoch => left + (epoch - 1) / 16 * (W - left - right);
    const y = loss => upper + (1 - loss / .45) * (H - upper - bottom);
    const path = key => D.history.map((r, i) => `${i ? 'L' : 'M'}${x(r.epoch).toFixed(2)},${y(r[key]).toFixed(2)}`).join(' ');
    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Training and validation cross-entropy loss over 17 epochs. Validation loss is lowest at epoch 12.">`;
    for (let loss = 0; loss <= .4; loss += .1) svg += `<line x1="${left}" x2="${W - right}" y1="${y(loss)}" y2="${y(loss)}" stroke="#eeeae121"/><text x="${left - 12}" y="${y(loss) + 5}" fill="#c0c3b9" text-anchor="end" font-size="${textSize}">${loss.toFixed(1)}</text>`;
    for (let e = 1; e <= 17; e += narrow ? 4 : 2) svg += `<text x="${x(e)}" y="${H - 38}" fill="#c0c3b9" text-anchor="middle" font-size="${textSize}">${e}</text>`;
    svg += `<text x="${W - 15}" y="${H - 2}" text-anchor="end" fill="#c0c3b9" font-size="${textSize}">Epoch</text><text x="${left}" y="20" fill="#c0c3b9" font-size="${textSize}">Cross-entropy loss</text>`;
    svg += `<path class="curve" d="${path('train_loss')}" fill="none" stroke="#bcc4b2" stroke-width="2.5"/><path class="curve validation" d="${path('validation_loss')}" fill="none" stroke="#c3ce9c" stroke-width="3"/>`;
    svg += `<line id="epoch-line" y1="${upper}" y2="${H - bottom}" stroke="#ffffff55" stroke-dasharray="4 5"/><circle id="epoch-point" r="5" fill="#c3ce9c" stroke="#20231f" stroke-width="2"/>`;
    svg += `<text x="${left}" y="${H - 2}" fill="#bcc4b2" font-size="${textSize}">— Training</text><text x="${left + 140}" y="${H - 2}" fill="#c3ce9c" font-size="${textSize}">— Validation</text></svg>`;
    $('#learning-chart').innerHTML = svg;
    $$('.curve').forEach(p => p.style.setProperty('--length', p.getTotalLength()));
    function updateEpoch() {
      const epoch = Number($('#epoch-slider').value), row = D.history[epoch - 1];
      $('#epoch-number').textContent = epoch;
      $('#epoch-metrics').innerHTML = `Train loss <b>${row.train_loss.toFixed(3)}</b><br>Validation loss <b>${row.validation_loss.toFixed(3)}</b><br>Validation accuracy <b>${fmt(row.validation_accuracy)}</b>`;
      $('#epoch-line').setAttribute('x1', x(epoch)); $('#epoch-line').setAttribute('x2', x(epoch));
      $('#epoch-point').setAttribute('cx', x(epoch)); $('#epoch-point').setAttribute('cy', y(row.validation_loss));
    }
    $('#epoch-slider').oninput = updateEpoch; updateEpoch();
  }
  learningChart();
  let chartResizeTimer;
  window.addEventListener('resize', () => { clearTimeout(chartResizeTimer); chartResizeTimer = setTimeout(learningChart, 120); });
  const modelNames = ['Logistic regression', 'kNN', 'CNN'];
  $('#results').innerHTML = modelNames.map((name, i) => {
    const test = D.metrics.find(r => r.model === name && r.split === 'Test');
    const val = D.metrics.find(r => r.model === name && r.split === 'Validation');
    return `<div class="result-row" style="--i:${i}"><div class="result-name">${name}</div><div class="result-score">${fmt(test.accuracy)}</div><div class="result-track" aria-label="${name} test ${fmt(test.accuracy)}, validation ${fmt(val.accuracy)}"><div class="result-fill" style="--score:${test.accuracy * 100}%"></div><span class="result-marker" style="--validation:${val.accuracy * 100}%" title="Validation: ${fmt(val.accuracy)}"></span></div><p class="result-support">Validation ${fmt(val.accuracy)} · Macro-F1 ${test['macro_f1-score'].toFixed(4)}</p></div>`;
  }).join('');

  function renderClass() {
    const label = D.classes[selectedClass];
    $$('#analysis-classes button').forEach(b => b.setAttribute('aria-pressed', String(Number(b.dataset.class) === selectedClass)));
    $('#analysis-title').textContent = matrixOpen ? 'CNN confusion matrix' : label;
    $('#recall-bars').hidden = matrixOpen; $('#matrix-wrap').hidden = !matrixOpen;
    $('#matrix-toggle').textContent = matrixOpen ? 'Class recall' : 'Confusion matrix';
    $('#matrix-toggle').setAttribute('aria-pressed', String(matrixOpen));
    $('#recall-bars').innerHTML = modelNames.map(name => {
      const r = D.per_class.find(r => r.model === name && r.split === 'Test' && r.class_id === selectedClass);
      return `<div class="recall-row"><span>${name}</span><div class="recall-track"><span style="--score:${r.recall * 100}%"></span></div><strong>${fmt(r.recall, 1)}</strong></div>`;
    }).join('');
    const cnn = D.per_class.find(r => r.model === 'CNN' && r.split === 'Test' && r.class_id === selectedClass);
    $('#class-takeaway').textContent = matrixOpen ? 'Rows are actual classes; columns are predictions. Hover a cell for its full meaning. Every row totals 1,000.' : selectedClass === 6 ? '676 shirts correctly classified. 324 missed. The strongest overall model still has a substantial weakness.' : `${Math.round(cnn.recall * 1000).toLocaleString('en-US')} of 1,000 actual ${label.toLowerCase()} images correctly classified by the CNN.`;
    if (matrixOpen) {
      const short = ['Top', 'Trou.', 'Pull.', 'Dress', 'Coat', 'Sand.', 'Shirt', 'Sneak.', 'Bag', 'Boot'];
      $('#matrix-wrap').innerHTML = `<table class="matrix-table"><caption class="sr-only">CNN test confusion matrix. Rows actual, columns predicted, counts.</caption><thead><tr><th>True ↓ / Pred →</th>${short.map(s => `<th scope="col">${s}</th>`).join('')}</tr></thead><tbody>${D.cnn_matrix.map((row, i) => `<tr><th scope="row">${short[i]}</th>${row.map((v, j) => `<td tabindex="0" title="Actual ${D.classes[i]}, predicted ${D.classes[j]}: ${v} images" aria-label="Actual ${D.classes[i]}, predicted ${D.classes[j]}, ${v} images" style="background:rgba(195,206,156,${.035 + v / 1000 * .8});color:${v > 400 ? '#11140f' : '#f0f1e8'}">${v}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
    }
  }
  D.classes.forEach((name, i) => { const b = document.createElement('button'); b.textContent = name; b.dataset.class = i; b.addEventListener('click', () => { selectedClass = i; matrixOpen = false; renderClass(); }); $('#analysis-classes').append(b); });
  $('#matrix-toggle').addEventListener('click', () => { matrixOpen = !matrixOpen; renderClass(); }); renderClass();

  function renderErrors() {
    $$('[data-error-model]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.errorModel === errorModel)));
    $('#error-grid').replaceChildren();
    D.errors.filter(r => r.model === errorModel).slice(0, 6).forEach(r => {
      const b = document.createElement('button'); b.className = 'error-sample';
      b.innerHTML = `<img src="${r.image}" alt="Actual ${esc(r.actual)}, ${esc(errorModel)} predicted ${esc(r.predicted)}"><span><b>${esc(r.actual)}</b>→ ${esc(r.predicted)}<small>Test image #${r.id}</small></span>`;
      b.addEventListener('click', () => inspectImage(r.image, `${r.actual} → ${r.predicted}`, `${errorModel} misclassified official test image #${r.id}. This example comes from the saved seed-42 random error sample. It illustrates an error; it does not prove its cause.`));
      $('#error-grid').append(b);
    });
    $('#confusion-list').innerHTML = D.confusions.filter(r => r.model === errorModel).slice(0, 4).map(r => `${esc(r.actual)} → ${esc(r.predicted)} <b>${r.count}</b>`).join('<br>');
  }
  $$('[data-error-model]').forEach(b => b.addEventListener('click', () => { errorModel = b.dataset.errorModel; renderErrors(); })); renderErrors();

  const maxDirectionalCount = Math.max(...D.shirt_analysis.pairs.flatMap(p => [p.forward.count, p.reverse.count]));
  D.shirt_analysis.pairs.forEach((pair, i) => {
    const row = document.createElement('div'); row.className = 'shirt-pair';
    function photo(example) {
      const b = document.createElement('button'); b.className = 'pair-example';
      b.innerHTML = `<img src="${example.image}" alt="Actual ${esc(example.actual)}, CNN predicted ${esc(example.predicted)}, official test image ${example.id}"><span>Actual ${esc(example.actual)}<small>Test #${example.id}</small></span>`;
      b.addEventListener('click', () => inspectImage(example.image, `${example.actual} → ${example.predicted}`, `Genuine official test image #${example.id}. True label: ${example.actual}; saved CNN prediction: ${example.predicted}. Selected as the first saved test index matching this error direction, not as a random or representative example. No new inference was performed.`));
      return b;
    }
    const bars = document.createElement('div'); bars.className = 'paired-directions';
    bars.innerHTML = [pair.forward, pair.reverse].map((e, direction) => `<div class="direction-line${direction ? ' reverse-direction' : ''}"><div class="direction-meta"><span>${esc(e.actual)} → ${esc(e.predicted)}</span><strong>${e.count}</strong></div><div class="direction-track"><span style="--width:${e.count / maxDirectionalCount * 100}%;--i:${i * 2 + direction}"></span></div></div>`).join('');
    row.append(photo(pair.forward), bars, photo(pair.reverse));
    $('#shirt-pairs').append(row);
  });
  $$('[data-error-view]').forEach(b => b.addEventListener('click', () => {
    const gallery = b.dataset.errorView === 'gallery';
    $$('[data-error-view]').forEach(button => button.setAttribute('aria-pressed', String(button === b)));
    $('#shirt-view').hidden = gallery; $('#sampled-error-view').hidden = !gallery;
    $('#error-heading').textContent = gallery ? 'Examples of classification errors' : `Shirt recall: ${fmt(D.shirt_analysis.recall, 1)}`;
    $('#error-subtitle').textContent = gallery ? 'Real errors from the saved, seed-42 random sample. Click an image to inspect it.' : `${D.shirt_analysis.correct} of ${D.shirt_analysis.support.toLocaleString('en-US')} actual shirts correctly identified. Each arrow reads actual → predicted.`;
    $('#error-foot').textContent = gallery ? 'Six of the 12 saved random error examples per model. Return to Shirt confusions for the directional comparison.' : 'Equal class sizes do not imply equal difficulty. Errors need not be symmetric. Each direction starts with 1,000 actual images.';
  }));

  function showNotes() {
    const s = scenes[current], original = D.notes[noteMap[current]];
    $('#notes-title').textContent = s.dataset.title;
    $('#notes-time').textContent = `SCENE ${current + 1} / TARGET ${s.dataset.time} SECONDS`;
    let text = D.scene_notes?.[String(current)] || original.notes;
    if (current >= 4 && current <= 6) text += '\n\nInteractive walkthrough: use the same validation class across the three model scenes. Examples are the first validation observation of each class, not selected by correctness. All three predictions match the original saved outputs. These are recorded model outputs; the browser is not fitting a model. Feature maps show eight channels after pooling, normalised with a common scale within each tensor. Logistic-regression colour shows signed pixel × weight contributions for the predicted class, excluding its intercept.';
    $('#notes-content').textContent = text;
    openDialog('#notes-dialog');
  }
  const qa = [
    ['Did you stratify? Did you cross-validate?', 'Yes. StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) and next(splitter.split(...)) produce the first yielded split. Fold 0 is validation; the other four folds are training. We used that one saved 80/20 split for every model, not five-fold cross-validation. All ten classes are exactly 10% of every actual subset. The official test set was supplied by Fashion-MNIST; we checked its balance.'],
    ['How do stratification and grouping differ?', 'Stratification preserves class proportions. Grouping uses hashes of original pixels so exact duplicate images stay together. No exact duplicate groups were found in the development pool. Grouping was a precaution, not duplicate removal; exact balance is verified for this saved split, not guaranteed for every possible grouped dataset.'],
    ['Why keep validation and test separate?', 'Validation chooses k, C, configurations and checkpoints. Test data assess the frozen choices. Do not tune replacement models using the now-observed test scores.'],
    ['Why C = 0.1 and k = 3?', 'Highest validation accuracy among the settings tested. The margins over nearby choices were small: 10 validation images for C = 0.1 versus 1, and 7 for k = 3 versus 5.'],
    ['How many CNNs?', 'Two architecture widths. Four initial training runs and four additional seed runs: eight total. The two wider configurations each have three seeds, including their existing seed-42 runs. Every network has two convolutional layers.'],
    ['Why ReLU instead of Leaky ReLU?', 'ReLU is a simple conventional nonlinear activation. It was used consistently; this experiment did not demonstrate superiority over Leaky ReLU.'],
    ['Where is softmax?', 'The CNN outputs raw logits to CrossEntropyLoss, which handles log-softmax internally. Softmax is applied to logits when displaying probabilities. These probabilities are not automatically calibrated confidence estimates.'],
    ['Why epoch 12, if a later epoch had higher accuracy?', 'Within-run checkpoint selection used lowest validation loss, which occurred at epoch 12. Between configurations, validation accuracy at the restored checkpoints determined selection. The two decisions use different predeclared rules.'],
    ['Are these real features and probabilities?', 'Yes. They were computed using the frozen checkpoints on deterministic validation examples and checked against saved predictions. The browser animates their presentation; it does not train or run arbitrary new inference.'],
    ['What does the logistic-regression heatmap mean?', 'Each colour is the signed product of this pixel value and the weight for the predicted class. The sum plus the intercept equals that class logit. It is not a causal explanation. Different maps are normalised separately.'],
    ['What evidence supports overfitting?', 'CNN training loss keeps falling after epoch 12 while validation loss does not beat its minimum. For logistic regression, weaker regularisation raises training accuracy and eventually reduces validation accuracy. Ordinary kNN training accuracy is optimistic because queries can include themselves as neighbours.'],
    ['Is logistic regression underfitting?', 'Its linear pixel representation may limit fit, but its lower accuracy alone does not prove underfitting. Training accuracy is 87.42% and validation 85.40%; the evidence supports a limited representation plus some generalisation gap.'],
    ['Why divide by 255?', 'It scales pixel intensities into [0,1]; it is not feature-wise standardisation. Because all pixels are scaled by the same constant, this step alone does not change Euclidean nearest-neighbour rankings.'],
    ['What does per-class accuracy mean here?', 'We report recall: correct predictions among actual members of a class. This avoids the misleadingly high one-versus-rest accuracy that the many negative examples could produce.'],
    ['Why do shirts fail?', 'Observed shirt recall is 676/1,000 = 67.6%, despite balanced classes. Shirt → T-shirt/top is 127 versus 63 in reverse; Shirt → Coat is 89 versus 32; Shirt → Pullover is 71 versus 32. Overlapping silhouettes, low resolution and potentially more within-shirt variation are hypotheses. We have not proven those causes or measured greater within-class variation. Equal class sizes do not imply equal difficulty.'],
    ['Should confusion be symmetric when classes are balanced?', 'No. Balanced true-class supports do not force equal errors in both directions. Read rows as actual class and columns as prediction. Each direction here starts with 1,000 actual examples, but the learned decision boundaries and observations differ.'],
    ['Where did the paired error examples come from?', 'Each is the first official test index in the saved CNN predictions matching that specific direction. They are genuine, deliberately illustrative examples, not a random or representative sample. The optional gallery separately retains the original seed-42 random sample.'],
    ['Does the CNN always win?', 'No. It leads overall, but kNN has slightly higher ankle-boot test recall. The deterministic shirt walkthrough is also an example where the classical models are correct and the CNN is wrong.'],
    ['Why not refit on all 60,000 images?', 'The frozen protocol retained the exact checkpoints trained on 48,000 images, preserving the reviewed validation evidence. This gives up possible gains from more data; it does not prove refitting would perform worse.'],
    ['Is spatial structure the only explanation?', 'No. Models differ in capacity and training as well as their use of image structure. A controlled ablation would be needed to isolate the effect of convolution.'],
  ];
  const summaryTable = `<table><thead><tr><th>Model</th><th>Validation</th><th>Test</th><th>Test macro-F1</th></tr></thead><tbody>${modelNames.map(name => { const t = D.metrics.find(r => r.model === name && r.split === 'Test'), v = D.metrics.find(r => r.model === name && r.split === 'Validation'); return `<tr><td>${name}</td><td>${fmt(v.accuracy)}</td><td>${fmt(t.accuracy)}</td><td>${t['macro_f1-score'].toFixed(4)}</td></tr>`; }).join('')}</tbody></table>`;
  const perClassTable = `<table><thead><tr><th>Class</th>${modelNames.map(n => `<th>${n}<br>Val / Test recall</th>`).join('')}</tr></thead><tbody>${D.classes.map((name, i) => `<tr><td>${name}</td>${modelNames.map(n => { const v = D.per_class.find(r => r.model === n && r.split === 'Validation' && r.class_id === i), t = D.per_class.find(r => r.model === n && r.split === 'Test' && r.class_id === i); return `<td>${fmt(v.recall, 1)} / ${fmt(t.recall, 1)}</td>`; }).join('')}</tr>`).join('')}</tbody></table>`;
  $('#evidence-content').innerHTML = `<h3>Results</h3>${summaryTable}<h3>Per-class validation and test recall</h3><div style="overflow:auto">${perClassTable}</div><h3>Exact selected CNN</h3><p>Conv(1→32, 3×3, padding 1) → ReLU → MaxPool(2) → Conv(32→64, 3×3, padding 1) → ReLU → MaxPool(2) → Flatten → Dense(3,136→128) → ReLU → Dropout(0.25) → Dense(128→10). 421,642 trainable parameters.</p><p>Adam, learning rate 0.001, batch size 128, max 30 epochs. ReduceLROnPlateau halves the rate after 2 stagnant validation-loss epochs; threshold 0, minimum rate 0.00001. Early-stopping patience 5. Original seed 42, lowest-loss checkpoint at epoch 12. No weight decay in the selected model.</p><h3>Integrity and provenance</h3><p>All three model hashes matched the frozen protocol. CPU validation predictions reproduced the saved outputs exactly before test scoring. Publisher checksums and IDX dimensions were verified. Exact development/test overlap: zero. Every test category has 1,000 observations.</p><p>${esc(D.provenance.examples)}. ${esc(D.provenance.features)}. ${esc(D.provenance.test)}.</p><h3>Dataset and project evidence</h3><p><a href="https://github.com/zalandoresearch/fashion-mnist/tree/b2617bb6d3ffa2e429640350f613e3291e10b141" target="_blank" rel="noopener">Fashion-MNIST / Zalando Research</a></p><p>Project-relative source paths:</p><ul>${knownSources.map(s => `<li><code>${esc(s)}</code></li>`).join('')}</ul><h3>Frozen checkpoint fingerprints</h3>${Object.entries(D.model_hashes).map(([name, hash]) => `<p>${name}<br><code>${hash}</code></p>`).join('')}<h3>Practise the questions</h3>${qa.map(([q, a]) => `<details><summary>${esc(q)}</summary><p>${esc(a)}</p></details>`).join('')}<h3>Ask Álvaro</h3><p>Does the balance of methods and findings match the assignment? Is the checkpoint-selection rule clear? Is the error analysis deep enough? Which details should move into or out of the main talk?</p>`;

  scenes.forEach((s, i) => {
    const b = document.createElement('button');
    b.innerHTML = `<span>${String(i + 1).padStart(2, '0')}</span>${esc(s.dataset.title)}`;
    b.addEventListener('click', () => { closeAll(); go(i); }); $('#overview-grid').append(b);
  });
  $('#previous').addEventListener('click', () => go(current - 1)); $('#next').addEventListener('click', () => go(current + 1));
  $$('[data-next]').forEach(b => b.addEventListener('click', () => go(current + 1)));
  $('#presenter-tools').addEventListener('click', () => openDialog('#tools-dialog'));
  $('#tools-evidence').addEventListener('click', () => openDialog('#evidence-dialog'));
  $('#overview-button').addEventListener('click', () => openDialog('#overview-dialog'));
  $('#notes-button').addEventListener('click', showNotes);
  $('#evidence-button')?.addEventListener('click', () => openDialog('#evidence-dialog'));
  $('#notes-evidence').addEventListener('click', () => openDialog('#evidence-dialog'));
  $$('[data-close]').forEach(b => b.addEventListener('click', () => b.closest('dialog').close()));
  $$('dialog').forEach(dialog => dialog.addEventListener('click', e => {
    if (e.target !== dialog) return;
    const r = dialog.getBoundingClientRect();
    if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) dialog.close();
  }));
  async function fullscreen() {
    try { if (document.fullscreenElement) await document.exitFullscreen(); else await document.documentElement.requestFullscreen(); }
    catch { announce('Fullscreen is unavailable here. Open this HTML file directly in a browser.'); }
  }
  function syncFullscreenControl() {
    const active = Boolean(document.fullscreenElement);
    $('#fullscreen-button').textContent = active ? 'Exit full screen' : 'Full screen';
    $('#fullscreen-button').setAttribute('aria-pressed', String(active));
  }
  document.addEventListener('fullscreenchange', syncFullscreenControl);
  syncFullscreenControl();
  $('#fullscreen-button').addEventListener('click', () => { closeAll(); fullscreen(); });
  document.addEventListener('keydown', e => {
    if (e.ctrlKey || e.altKey || e.metaKey || e.repeat) return;
    if (e.key === 'Escape' && document.fullscreenElement && !$('dialog[open]')) {
      e.preventDefault();
      fullscreen();
      return;
    }
    // Fullscreen is global for non-text controls; range/select arrows stay native.
    if (e.key.toLowerCase() === 'f' && !e.target.closest('textarea,[contenteditable=true],input:not([type=range])')) {
      e.preventDefault();
      fullscreen();
      return;
    }
    if (e.target.matches('input,textarea,select,[contenteditable=true]')) return;
    if ($('dialog[open]')) return;
    if (['ArrowRight', 'PageDown', ' '].includes(e.key) && (e.key !== ' ' || e.target === document.body)) { e.preventDefault(); go(current + 1); }
    else if (['ArrowLeft', 'PageUp'].includes(e.key)) { e.preventDefault(); go(current - 1); }
    else if (e.key === 'Home') { e.preventDefault(); go(0); }
    else if (e.key === 'End') { e.preventDefault(); go(scenes.length - 1); }
    else if (e.key.toLowerCase() === 'n') showNotes();
    else if (e.key.toLowerCase() === 'o') openDialog('#overview-dialog');
    else if (e.key.toLowerCase() === 'i') openDialog('#evidence-dialog');
  });
  let touch = null;
  $('#deck').addEventListener('touchstart', e => {
    if (e.target.closest('button,input,a,canvas,[role=button]')) return;
    touch = { x: e.changedTouches[0].clientX, y: e.changedTouches[0].clientY };
  }, { passive: true });
  $('#deck').addEventListener('touchend', e => {
    if (!touch) return;
    const dx = e.changedTouches[0].clientX - touch.x, dy = e.changedTouches[0].clientY - touch.y;
    if (Math.abs(dx) > 80 && Math.abs(dx) > Math.abs(dy) * 1.5) go(current + (dx < 0 ? 1 : -1));
    touch = null;
  }, { passive: true });
  window.addEventListener('hashchange', () => { const index = Number(location.hash.slice(1)); if (Number.isInteger(index) && index >= 1 && index <= scenes.length) go(index - 1, false); });
  const initial = Number(location.hash.slice(1));
  go(Number.isInteger(initial) && initial >= 1 && initial <= scenes.length ? initial - 1 : 0);
})();
