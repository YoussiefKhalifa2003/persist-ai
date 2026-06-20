/** PERSIST-AI interactive viewer: raw-only → Activate PERSIST-AI → split comparison */

const RAW_SRC = "../../results/demo_videos/VIDEO3_RAW_ONLY.mp4";
const SPLIT_SRC = "../../results/demo_videos/VIDEO3_PERSIST_SPLIT.mp4";

const phaseRaw = document.getElementById("phase-raw");
const phaseSplit = document.getElementById("phase-split");
const rawVideo = document.getElementById("raw-video");
const splitVideo = document.getElementById("split-video");
const activateBtn = document.getElementById("activate-btn");
const replayBtn = document.getElementById("replay-btn");

function showRaw() {
  phaseRaw.classList.remove("hidden");
  phaseSplit.classList.add("hidden");
  splitVideo.pause();
  rawVideo.currentTime = 0;
  rawVideo.play().catch(() => {});
}

function showSplit() {
  phaseRaw.classList.add("hidden");
  phaseSplit.classList.remove("hidden");
  rawVideo.pause();
  splitVideo.currentTime = 0;
  splitVideo.play().catch(() => {});
}

rawVideo.src = RAW_SRC;
splitVideo.src = SPLIT_SRC;

rawVideo.loop = true;
rawVideo.muted = true;

activateBtn.addEventListener("click", showSplit);
replayBtn.addEventListener("click", showRaw);

rawVideo.addEventListener("loadeddata", () => {
  rawVideo.play().catch(() => {});
});
