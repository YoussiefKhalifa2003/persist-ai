const sceneList = document.getElementById("scene-list");
const rawFrame = document.getElementById("raw-frame");
const rawPlayBtn = document.getElementById("raw-play-btn");
const rawScrub = document.getElementById("raw-scrub");
const rawFrameLabel = document.getElementById("raw-frame-label");
const selectionLayer = document.getElementById("selection-layer");
const selectionFrame = document.getElementById("selection-frame");
const candidateSvg = document.getElementById("candidate-svg");
const targetBadge = document.getElementById("target-badge");
const targetPreview = document.getElementById("target-preview");
const targetPreviewCanvas = document.getElementById("target-preview-canvas");
const targetPreviewLabel = document.getElementById("target-preview-label");
const enhanceBtn = document.getElementById("enhance-btn");
const activateBtn = document.getElementById("activate-btn");
const resetBtn = document.getElementById("reset-btn");
const statusEl = document.getElementById("status");
const progressWrap = document.getElementById("progress-wrap");
const progressBar = document.getElementById("progress-bar");
const progressLabel = document.getElementById("progress-label");
const resultPanel = document.getElementById("result-panel");
const resultFrame = document.getElementById("result-frame");
const resultPlayBtn = document.getElementById("result-play-btn");
const resultScrub = document.getElementById("result-scrub");
const resultFrameLabel = document.getElementById("result-frame-label");
const downloadLink = document.getElementById("download-link");

let scenes = [];
let activeScene = null;
let selectedCandidate = null;
let selectedFrame = 0;
let rawTimer = null;
let resultTimer = null;
let currentJobId = null;
let resultFrames = 0;
let latestCandidates = [];
let candidateGroups = new Map();

function setStatus(text, tone = "") {
  statusEl.textContent = text;
  statusEl.dataset.tone = tone;
}

function setRawFrame(index) {
  if (!activeScene) return;
  const max = Math.max(0, (activeScene.frames || 1) - 1);
  const frame = Math.max(0, Math.min(max, Number(index)));
  rawScrub.value = String(frame);
  rawFrameLabel.textContent = `Frame ${frame}`;
  rawFrame.src = `/api/scenes/${activeScene.scene_id}/frame?index=${frame}&t=${Date.now()}`;
}

function setResultFrame(index) {
  if (!currentJobId) return;
  const max = Math.max(0, resultFrames - 1);
  const frame = Math.max(0, Math.min(max, Number(index)));
  resultScrub.value = String(frame);
  resultFrameLabel.textContent = `Frame ${frame}`;
  resultFrame.src = `/api/jobs/${currentJobId}/frame?index=${frame}&t=${Date.now()}`;
}

function stopRaw() {
  if (rawTimer) window.clearInterval(rawTimer);
  rawTimer = null;
  rawPlayBtn.textContent = "Play";
}

function stopResult() {
  if (resultTimer) window.clearInterval(resultTimer);
  resultTimer = null;
  resultPlayBtn.textContent = "Play";
}

function toggleRawPlay() {
  if (!activeScene) return;
  if (rawTimer) {
    stopRaw();
    return;
  }
  rawPlayBtn.textContent = "Pause";
  rawTimer = window.setInterval(() => {
    const next = Number(rawScrub.value) + 1;
    if (next >= Number(rawScrub.max)) {
      setRawFrame(0);
    } else {
      setRawFrame(next);
    }
  }, 1000 / (activeScene.fps || 15));
}

function toggleResultPlay() {
  if (!currentJobId) return;
  if (resultTimer) {
    stopResult();
    return;
  }
  resultPlayBtn.textContent = "Pause";
  resultTimer = window.setInterval(() => {
    const next = Number(resultScrub.value) + 1;
    if (next >= Number(resultScrub.max)) {
      setResultFrame(0);
    } else {
      setResultFrame(next);
    }
  }, 1000 / 15);
}

function resetSelection() {
  selectedCandidate = null;
  activateBtn.disabled = true;
  targetBadge.classList.add("hidden");
  targetPreview.classList.add("hidden");
  candidateSvg.querySelectorAll(".candidate").forEach((el) => el.classList.remove("selected", "hovered"));
}

function hideEnhanceLayer() {
  selectionLayer.classList.add("hidden");
  candidateSvg.innerHTML = "";
  resetSelection();
}

function selectScene(scene) {
  activeScene = scene;
  stopRaw();
  stopResult();
  rawScrub.max = String(Math.max(0, scene.frames - 1));
  rawScrub.value = String(scene.recommended_frame || 0);
  resultPanel.classList.add("hidden");
  progressWrap.classList.add("hidden");
  hideEnhanceLayer();
  setRawFrame(scene.recommended_frame || 0);
  setStatus("Scene loaded. Click Enhance Scene when the target is visible.");
  [...sceneList.children].forEach((child) => {
    child.classList.toggle("active", child.dataset.sceneId === scene.scene_id);
  });
}

function renderScenes() {
  sceneList.innerHTML = "";
  scenes.forEach((scene) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "scene-card";
    button.dataset.sceneId = scene.scene_id;
    button.innerHTML = `<strong>${scene.name}</strong><span>${scene.description}</span>`;
    button.addEventListener("click", () => selectScene(scene));
    sceneList.appendChild(button);
  });
}

function drawCandidates(candidates) {
  candidateSvg.innerHTML = "";
  candidateGroups = new Map();
  latestCandidates = candidates;
  const width = selectionFrame.naturalWidth || selectionFrame.clientWidth;
  const height = selectionFrame.naturalHeight || selectionFrame.clientHeight;
  candidateSvg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  candidateSvg.style.height = `${selectionFrame.clientHeight}px`;
  selectionLayer.style.height = `${selectionFrame.clientHeight}px`;

  candidates.forEach((candidate) => {
    const [x1, y1, x2, y2] = candidate.bbox;
    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    group.classList.add("candidate");
    group.dataset.candidateId = candidate.id;
    group.dataset.classId = candidate.class_id;

    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", x1);
    rect.setAttribute("y", y1);
    rect.setAttribute("width", x2 - x1);
    rect.setAttribute("height", y2 - y1);
    rect.setAttribute("rx", 2);

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", x1);
    label.setAttribute("y", Math.max(14, y1 - 7));
    label.textContent = candidate.tracking_quality && candidate.tracking_quality !== "high"
      ? `${candidate.class_name} ${candidate.tracking_quality}`
      : candidate.class_name;

    group.appendChild(rect);
    group.appendChild(label);
    group.style.pointerEvents = "none";
    candidateSvg.appendChild(group);
    candidateGroups.set(candidate.id, group);
  });
}

function svgPointFromEvent(event) {
  const pt = candidateSvg.createSVGPoint();
  pt.x = event.clientX;
  pt.y = event.clientY;
  return pt.matrixTransform(candidateSvg.getScreenCTM().inverse());
}

function candidateClickScore(candidate, pt) {
  const [x1, y1, x2, y2] = candidate.bbox;
  const inside = pt.x >= x1 && pt.x <= x2 && pt.y >= y1 && pt.y <= y2;
  const cx = (x1 + x2) / 2;
  const cy = (y1 + y2) / 2;
  const area = Math.max(1, (x2 - x1) * (y2 - y1));
  const dist = Math.hypot(pt.x - cx, pt.y - cy);
  if (inside) return dist + area * 0.0002;
  return 100000 + dist;
}

function candidateAtPoint(pt) {
  const ranked = latestCandidates
    .map((candidate) => ({ candidate, score: candidateClickScore(candidate, pt) }))
    .sort((a, b) => a.score - b.score);
  return ranked[0] && ranked[0].score < 100000 ? ranked[0].candidate : null;
}

function setHoveredCandidate(candidate) {
  candidateSvg.querySelectorAll(".candidate").forEach((el) => el.classList.remove("hovered"));
  if (!candidate) return;
  const group = candidateGroups.get(candidate.id);
  if (group && selectedCandidate?.id !== candidate.id) group.classList.add("hovered");
}

function lockCandidate(candidate) {
  selectedCandidate = candidate;
  candidateSvg.querySelectorAll(".candidate").forEach((el) => el.classList.remove("selected"));
  const group = candidateGroups.get(candidate.id);
  if (group) group.classList.add("selected");
  activateBtn.disabled = false;
  targetBadge.textContent = `TARGET LOCKED: ${candidate.class_name.toUpperCase()} ${candidate.id}`;
  targetBadge.classList.remove("hidden");
  const [x1, y1, x2, y2] = candidate.bbox.map((v) => Math.round(v));
  drawTargetPreview(candidate);
  if (candidate.tracking_quality && candidate.tracking_quality !== "high") {
    setStatus(
      `Target locked on ${candidate.class_name} ${candidate.id}. Tracking quality is limited - rendering with uncertainty.`,
      "warn",
    );
  } else {
    setStatus(`Target locked on ${candidate.class_name} ${candidate.id} [${x1}, ${y1}, ${x2}, ${y2}].`, "ok");
  }
}

function drawTargetPreview(candidate) {
  const [x1, y1, x2, y2] = candidate.bbox;
  const ctx = targetPreviewCanvas.getContext("2d");
  ctx.clearRect(0, 0, targetPreviewCanvas.width, targetPreviewCanvas.height);
  if (!selectionFrame.complete || !selectionFrame.naturalWidth) return;
  const sx = Math.max(0, x1);
  const sy = Math.max(0, y1);
  const sw = Math.max(1, x2 - x1);
  const sh = Math.max(1, y2 - y1);
  const scale = Math.min(targetPreviewCanvas.width / sw, targetPreviewCanvas.height / sh);
  const dw = sw * scale;
  const dh = sh * scale;
  const dx = (targetPreviewCanvas.width - dw) / 2;
  const dy = (targetPreviewCanvas.height - dh) / 2;
  ctx.drawImage(selectionFrame, sx, sy, sw, sh, dx, dy, dw, dh);
  ctx.strokeStyle = "#00ff7f";
  ctx.lineWidth = 3;
  ctx.strokeRect(dx + 1, dy + 1, dw - 2, dh - 2);
  targetPreviewLabel.textContent = `${candidate.class_name.toUpperCase()} ${candidate.id}`;
  targetPreview.classList.remove("hidden");
}

candidateSvg.addEventListener("click", (event) => {
  if (!latestCandidates.length) return;
  const pt = svgPointFromEvent(event);
  const candidate = candidateAtPoint(pt);
  if (candidate) lockCandidate(candidate);
});

candidateSvg.addEventListener("mousemove", (event) => {
  if (!latestCandidates.length) return;
  setHoveredCandidate(candidateAtPoint(svgPointFromEvent(event)));
});

candidateSvg.addEventListener("mouseleave", () => {
  setHoveredCandidate(null);
});

async function enhanceScene() {
  if (!activeScene) return;
  stopRaw();
  resetSelection();
  selectedFrame = Number(rawScrub.value || activeScene.recommended_frame || 0);
  setStatus("Enhancing scene perception...");
  selectionLayer.classList.remove("hidden");
  selectionLayer.classList.add("scanning");
  selectionFrame.src = `/api/scenes/${activeScene.scene_id}/frame?index=${selectedFrame}&t=${Date.now()}`;

  const candidateResp = await fetch(`/api/scenes/${activeScene.scene_id}/candidates?frame=${selectedFrame}`);
  if (!candidateResp.ok) {
    setStatus("Could not analyze this frame. Try another moment.", "error");
    return;
  }
  const payload = await candidateResp.json();
  selectionFrame.onload = () => {
    selectionLayer.style.height = `${selectionFrame.clientHeight}px`;
    drawCandidates(payload.candidates || []);
    selectionLayer.classList.remove("scanning");
    if (!payload.candidates || payload.candidates.length === 0) {
      setStatus("No selectable targets on this frame. Scrub to another moment.", "error");
    } else {
      setStatus("Enhanced scene ready. Select a highlighted target.");
    }
  };
}

async function activatePersist() {
  if (!activeScene || !selectedCandidate) return;
  activateBtn.disabled = true;
  enhanceBtn.disabled = true;
  progressWrap.classList.remove("hidden");
  progressBar.style.width = "0%";
  progressLabel.textContent = "Queued";
  setStatus("Rendering PERSIST-AI comparison...");

  const response = await fetch("/api/render", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      scene_id: activeScene.scene_id,
      candidate_id: selectedCandidate.id,
      selection_frame: selectedFrame,
    }),
  });
  if (!response.ok) {
    setStatus("Could not start render.", "error");
    enhanceBtn.disabled = false;
    activateBtn.disabled = false;
    return;
  }
  const job = await response.json();
  pollJob(job.job_id);
}

async function pollJob(jobId) {
  const response = await fetch(`/api/jobs/${jobId}`);
  if (!response.ok) {
    setStatus("Render job disappeared.", "error");
    enhanceBtn.disabled = false;
    return;
  }
  const job = await response.json();
  const pct = Math.round((job.progress || 0) * 100);
  progressBar.style.width = `${pct}%`;
  progressLabel.textContent = job.message || `${pct}%`;

  if (job.status === "failed") {
    setStatus(job.error || "Render failed. Try another frame or target.", "error");
    enhanceBtn.disabled = false;
    resetSelection();
    return;
  }
  if (job.status === "complete") {
    currentJobId = jobId;
    const manifestResp = await fetch(`/api/jobs/${jobId}/manifest`);
    const manifest = manifestResp.ok ? await manifestResp.json() : { frames: 1, fps: 15 };
    resultFrames = manifest.frames || 1;
    resultScrub.max = String(Math.max(0, resultFrames - 1));
    downloadLink.href = `/api/jobs/${jobId}/video`;
    resultPanel.classList.remove("hidden");
    setResultFrame(0);
    progressLabel.textContent = "Complete";
    const quality = manifest.tracking_quality || "high";
    if (quality === "high") {
      setStatus("Comparison ready. Use the result player below or download the MP4.", "ok");
    } else {
      setStatus("Comparison ready with limited tracking quality. Use the result player below or download the MP4.", "warn");
    }
    enhanceBtn.disabled = false;
    return;
  }
  window.setTimeout(() => pollJob(jobId), 700);
}

async function loadScenes() {
  const response = await fetch("/api/scenes");
  if (!response.ok) {
    setStatus("Could not load scenes. Start the FastAPI demo server.", "error");
    return;
  }
  const payload = await response.json();
  scenes = payload.scenes || [];
  renderScenes();
  if (scenes[0]) selectScene(scenes[0]);
}

enhanceBtn.addEventListener("click", enhanceScene);
activateBtn.addEventListener("click", activatePersist);
resetBtn.addEventListener("click", () => {
  hideEnhanceLayer();
  resultPanel.classList.add("hidden");
  progressWrap.classList.add("hidden");
  enhanceBtn.disabled = false;
  setStatus("Reset. Scrub to a clear target, then click Enhance Scene.");
});
rawPlayBtn.addEventListener("click", toggleRawPlay);
resultPlayBtn.addEventListener("click", toggleResultPlay);
rawScrub.addEventListener("input", () => {
  stopRaw();
  hideEnhanceLayer();
  setRawFrame(rawScrub.value);
});
resultScrub.addEventListener("input", () => {
  stopResult();
  setResultFrame(resultScrub.value);
});

loadScenes();
