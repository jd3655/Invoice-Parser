const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const folderInput = document.getElementById("folderInput");
const startBtn = document.getElementById("startBtn");
const fileList = document.getElementById("fileList");
const progressArea = document.getElementById("progressArea");
const downloadArea = document.getElementById("downloadArea");
const downloadBtn = document.getElementById("downloadBtn");

let selectedFiles = [];
let currentJobId = null;
let pollInterval = null;

function updateFileList() {
  if (!selectedFiles.length) {
    fileList.innerHTML = "<p class='hint'>No files selected.</p>";
    return;
  }
  fileList.innerHTML = selectedFiles
    .map((file) => `<div class="file-item">${file.name}</div>`)
    .join("");
}

function addFiles(files) {
  const allowed = Array.from(files).filter((f) => {
    const name = f.name.toLowerCase();
    return name.endsWith(".pdf") || name.endsWith(".png");
  });
  selectedFiles = [...selectedFiles, ...allowed];
  updateFileList();
}

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  if (e.dataTransfer?.files?.length) {
    addFiles(e.dataTransfer.files);
  }
});

fileInput.addEventListener("change", (e) => addFiles(e.target.files || []));
folderInput.addEventListener("change", (e) => addFiles(e.target.files || []));

async function createJob(config) {
  const res = await fetch("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  if (!res.ok) throw new Error("Failed to create job");
  const data = await res.json();
  return data.job_id;
}

async function uploadFiles(jobId) {
  const formData = new FormData();
  selectedFiles.forEach((file) => formData.append("files", file, file.name));
  const res = await fetch(`/api/jobs/${jobId}/upload`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Upload failed: ${err}`);
  }
}

async function startProcessing(jobId) {
  const res = await fetch(`/api/jobs/${jobId}/start`, { method: "POST" });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Start failed: ${err}`);
  }
}

function renderStatus(status) {
  progressArea.innerHTML = "";
  Object.entries(status.files).forEach(([name, info]) => {
    const card = document.createElement("div");
    card.className = "progress-card";
    const statusClass = info.status === "done" ? "done" : info.status === "error" ? "error" : "processing";
    const pageInfo =
      info.pages?.total > 0
        ? `Page ${info.pages.current}/${info.pages.total}`
        : "Pending";
    card.innerHTML = `
      <div class="title">${name}</div>
      <div class="status ${statusClass}">${info.status.toUpperCase()}</div>
      <div class="hint">${pageInfo}</div>
      ${info.output_path ? `<div class="hint">Output: ${info.output_path}</div>` : ""}
      ${info.error ? `<div class="hint" style="color:#991b1b;">${info.error}</div>` : ""}
    `;
    progressArea.appendChild(card);
  });
}

async function pollStatus() {
  if (!currentJobId) return;
  try {
    const res = await fetch(`/api/jobs/${currentJobId}/status`);
    if (!res.ok) return;
    const data = await res.json();
    renderStatus(data);
    if (data.status === "completed" || data.status === "error") {
      clearInterval(pollInterval);
      startBtn.disabled = false;
      downloadArea.classList.remove("hidden");
    }
  } catch (err) {
    console.error(err);
  }
}

startBtn.addEventListener("click", async () => {
  if (!selectedFiles.length) {
    alert("Please select at least one PDF or PNG.");
    return;
  }
  startBtn.disabled = true;
  downloadArea.classList.add("hidden");
  progressArea.innerHTML = "<p class='hint'>Starting job...</p>";

  const config = {
    base_url: document.getElementById("baseUrl").value.trim(),
    model: document.getElementById("model").value.trim(),
    save_per_page: document.getElementById("perPage").checked,
  };

  try {
    currentJobId = await createJob(config);
    await uploadFiles(currentJobId);
    await startProcessing(currentJobId);
    pollInterval = setInterval(pollStatus, 1200);
  } catch (err) {
    alert(err.message);
    startBtn.disabled = false;
  }
});

downloadBtn.addEventListener("click", () => {
  if (!currentJobId) return;
  window.location.href = `/api/jobs/${currentJobId}/download`;
});

updateFileList();
