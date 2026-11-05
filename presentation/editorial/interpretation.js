'use strict';
// Displays recorded validation diagnostics. No model is executed in the browser.
window.setupInterpretation = function (data, classes, announce) {
  const $ = id => document.getElementById(id);
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  let example, target, demo, frame = 0, timer = null, method = 'saliency';
  let active = false, paused = false;
  const canvas = $('explanation-map'), ctx = canvas.getContext('2d');
  const patchCanvas = $('occlusion-input'), patchCtx = patchCanvas.getContext('2d');

  function stop() {
    clearInterval(timer); timer = null;
    $('occlusion-play').textContent = reducedMotion.matches ? 'Advance one patch' : 'Resume patch scan';
    $('occlusion-play').setAttribute('aria-pressed', 'false');
  }
  function start() {
    stop();
    if (!active || !target || paused || document.hidden || reducedMotion.matches) return;
    $('occlusion-play').textContent = 'Pause patch scan';
    $('occlusion-play').setAttribute('aria-pressed', 'true');
    timer = setInterval(() => { frame = (frame + 1) % data.patches.length; drawPatch(); }, 200);
  }
  function setActive(value) {
    if (value && !active) paused = false;
    active = value;
    start();
  }
  function drawPatch() {
    const patch = data.patches[frame], cell = patchCanvas.width / 28;
    demo.pixels.forEach((v, i) => {
      patchCtx.fillStyle = `rgb(${v},${v},${v})`;
      patchCtx.fillRect((i % 28) * cell, Math.floor(i / 28) * cell, cell, cell);
    });
    patchCtx.fillStyle = 'rgb(128,128,128)';
    patchCtx.fillRect(patch.col * cell, patch.row * cell, patch.width * cell, patch.height * cell);
    patchCtx.strokeStyle = '#eeeae1'; patchCtx.lineWidth = 2;
    patchCtx.strokeRect(patch.col * cell, patch.row * cell, patch.width * cell, patch.height * cell);
    const delta = target.patch_deltas[frame], occluded = target.base_score - delta;
    $('patch-position').value = String(frame);
    $('patch-position').setAttribute('aria-valuetext', `Patch ${frame + 1} of ${data.patches.length}, row ${patch.row + 1}, column ${patch.col + 1}`);
    $('patch-location').textContent = `Patch ${frame + 1} / ${data.patches.length} · row ${patch.row + 1}, column ${patch.col + 1}`;
    $('occlusion-score').textContent = occluded.toFixed(3);
    $('occlusion-delta').textContent = `${delta >= 0 ? '+' : '−'}${Math.abs(delta).toFixed(3)}`;
    $('occlusion-direction').textContent = delta >= 0 ? 'score drop after covering this patch' : 'score increased after covering this patch';
  }
  function drawMap() {
    const values = method === 'saliency' ? target.saliency : target.occlusion.map(v => Math.max(0, v));
    const bound = Math.max(...values), cell = canvas.width / 28;
    values.forEach((v, i) => {
      const f = bound > 0 ? v / bound : 0;
      // One sequential scale: graphite at zero, mineral at the image maximum.
      ctx.fillStyle = `rgb(${Math.round(32 + f * 163)},${Math.round(35 + f * 171)},${Math.round(31 + f * 125)})`;
      ctx.fillRect((i % 28) * cell, Math.floor(i / 28) * cell, cell, cell);
    });
    $('explanation-title').textContent = method === 'saliency' ? 'Gradient saliency' : 'Occlusion sensitivity';
    $('explanation-description').textContent = method === 'saliency'
      ? 'Larger absolute input gradients mean greater local sensitivity of this class score.'
      : 'Mean score drop for patches covering each pixel. Only positive drops are highlighted.';
    $('explanation-scale-max').textContent = bound > 0 ? 'Largest value in this map' : 'All values zero';
    canvas.setAttribute('aria-label', `${method === 'saliency' ? 'Absolute input gradient' : 'Positive mean occlusion score drop'} for ${classes[target.class_id]}, validation image ${demo.id}. Scale normalised independently for this map.`);
    $('explanation-formula').textContent = method === 'saliency' ? '|∂ class logit / ∂ pixel|' : 'Original logit − occluded logit';
    document.querySelectorAll('[data-explanation-method]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.explanationMethod === method)));
  }
  function chooseTarget(classId) {
    stop(); target = example.targets.find(t => t.class_id === classId);
    $('explanation-target').value = String(classId);
    $('original-score').textContent = target.base_score.toFixed(3);
    $('explanation-class').textContent = `Class score: ${classes[classId]}`;
    drawMap(); drawPatch();
    paused = false; start();
  }
  function render(selectedDemo) {
    stop(); demo = selectedDemo;
    example = data.examples.find(e => e.id === demo.id);
    frame = 0;
    $('interpretation-original').src = demo.image;
    $('interpretation-original').alt = `Actual ${classes[demo.label]}, validation image ${demo.id}`;
    $('interpretation-example').textContent = `Actual: ${classes[demo.label]} · CNN prediction: ${classes[example.predicted]}`;
    const selector = $('explanation-target'); selector.replaceChildren();
    example.targets.forEach(t => {
      const option = document.createElement('option'); option.value = t.class_id;
      option.textContent = `${classes[t.class_id]}${t.class_id === example.predicted ? ' (predicted)' : ' (true class)'}`;
      selector.append(option);
    });
    selector.disabled = example.targets.length === 1;
    $('target-hint').textContent = example.targets.length === 1
      ? 'Prediction matches the true class. Choose Shirt below to compare two class scores.'
      : 'Prediction differs from the true class. Compare Coat and Shirt without changing the image.';
    $('patch-position').max = String(data.patches.length - 1);
    chooseTarget(example.predicted);
  }
  $('explanation-target').addEventListener('change', e => {
    chooseTarget(Number(e.target.value)); announce(`Explaining the ${classes[target.class_id]} score`);
  });
  document.querySelectorAll('[data-explanation-method]').forEach(b => b.addEventListener('click', () => {
    method = b.dataset.explanationMethod; drawMap(); announce(`${$('explanation-title').textContent} selected`);
  }));
  $('patch-position').addEventListener('input', e => { frame = Number(e.target.value); drawPatch(); start(); });
  $('occlusion-play').addEventListener('click', () => {
    if (reducedMotion.matches) {
      frame = (frame + 1) % data.patches.length; drawPatch();
      announce('Reduced motion: advanced one patch. Use the slider to explore.'); return;
    }
    paused = !paused;
    start();
  });
  reducedMotion.addEventListener('change', start);
  document.addEventListener('visibilitychange', start);
  return { render, stop, setActive };
};
