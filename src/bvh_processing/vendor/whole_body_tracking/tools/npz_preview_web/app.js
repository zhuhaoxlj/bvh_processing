const $ = (selector) => document.querySelector(selector);
const JOB_STORAGE_KEY = "motionInspector.activeJobId";

const state = {
  job: null,
  pollTimer: null,
  renderStartedAt: null,
  elapsedTimer: null,
  trainingPollTimer: null,
  trainingStartedAt: null,
  trainingElapsedTimer: null,
  gpuMonitorTimer: null,
  gpuMonitorLoading: false,
  gpuMonitorHasData: false,
  gpuMonitorFailureCount: 0,
  trainingCheckpoints: [],
  checkpointRequestKey: null,
  checkpointLoading: false,
  historicalJobs: [],
  historyLoading: false,
  trainingPackageImporting: false,
  preferredCheckpointRunDirectory: null,
  activeTrainingJobs: [],
  activeTrainingJobsLoading: false,
};
const dropZone = $("#dropZone");
const fileInput = $("#fileInput");
const trainingPackageDropZone = $("#trainingPackageDropZone");
const trainingPackageInput = $("#trainingPackageInput");

dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") fileInput.click();
});
fileInput.addEventListener("change", () => fileInput.files[0] && uploadFile(fileInput.files[0]));
$("#replaceButton").addEventListener("click", () => fileInput.click());
$("#skipValidation").addEventListener("change", () => {
  const quickMode = $("#skipValidation").checked;
  dropZone.classList.toggle("quick-mode", quickMode);
  dropZone.querySelector(".drop-kicker").textContent = quickMode ? "FAST TRAINING DROP" : "DROP MOTION FILE";
  dropZone.querySelector("h2").textContent = quickMode ? "拖入已检查的 NPZ" : "把 NPZ 或 PKL 拖到这里";
  dropZone.querySelector("p:not(.drop-kicker)").textContent = quickMode
    ? "跳过本次检查 · 上传后直接配置训练"
    : "或者点击选择文件 · 支持 NPZ / HHTools PKL · 最大 512 MiB";
});

["dragenter", "dragover"].forEach((name) => dropZone.addEventListener(name, (event) => {
  event.preventDefault();
  dropZone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((name) => dropZone.addEventListener(name, (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
}));
dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (file) uploadFile(file);
});

$("#renderButton").addEventListener("click", startRender);
$("#cameraLayout").addEventListener("change", updateRenderControls);
$("#resolution").addEventListener("change", updateRenderControls);
$("#focalLength").addEventListener("change", updateRenderControls);
$("#numEnvs").addEventListener("change", () => {
  const runName = $("#runName");
  const gpuCount = Math.max(1, selectedTrainingDevices().length);
  if (/_local_\d+gpu_\d+$/.test(runName.value)) {
    runName.value = runName.value.replace(/_local_\d+gpu_\d+$/, `_local_${gpuCount}gpu_${$("#numEnvs").value}`);
  }
});
$("#trainingDevice").addEventListener("change", updateTrainingSelectionSummary);
$("#trainButton").addEventListener("click", () => startTraining());
$("#resumeTrainingButton").addEventListener("click", resumeTraining);
$("#stopTrainingButton").addEventListener("click", stopTraining);
$("#refreshGpuButton").addEventListener("click", loadSystemInfo);
$("#refreshActiveTrainingButton").addEventListener("click", loadActiveTrainingJobs);
$("#activeTrainingSelect").addEventListener("change", updateActiveTrainingSelection);
$("#stopSelectedTrainingButton").addEventListener("click", stopSelectedTraining);
$("#refreshTrainingHistoryButton").addEventListener("click", loadTrainingHistory);
$("#trainingHistorySelect").addEventListener("change", updateTrainingHistorySelection);
$("#openTrainingHistoryButton").addEventListener("click", openSelectedHistoricalJob);
$("#exportTrainingPackageButton").addEventListener("click", exportSelectedTrainingPackage);
trainingPackageDropZone.addEventListener("click", () => {
  if (!state.trainingPackageImporting) trainingPackageInput.click();
});
trainingPackageDropZone.addEventListener("keydown", (event) => {
  if (!state.trainingPackageImporting && (event.key === "Enter" || event.key === " ")) {
    event.preventDefault();
    trainingPackageInput.click();
  }
});
trainingPackageInput.addEventListener("change", () => {
  if (trainingPackageInput.files[0]) importTrainingPackage(trainingPackageInput.files[0]);
  trainingPackageInput.value = "";
});
["dragenter", "dragover"].forEach((name) => trainingPackageDropZone.addEventListener(name, (event) => {
  event.preventDefault();
  if (!state.trainingPackageImporting) trainingPackageDropZone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((name) => trainingPackageDropZone.addEventListener(name, (event) => {
  event.preventDefault();
  trainingPackageDropZone.classList.remove("dragging");
}));
trainingPackageDropZone.addEventListener("drop", (event) => {
  if (!state.trainingPackageImporting && event.dataTransfer.files[0]) {
    importTrainingPackage(event.dataTransfer.files[0]);
  }
});
$("#toggleLog").addEventListener("click", () => {
  const log = $("#renderLog");
  log.classList.toggle("hidden");
  $("#toggleLog").textContent = log.classList.contains("hidden") ? "展开日志" : "收起日志";
});
$("#toggleTrainingLog").addEventListener("click", () => {
  const log = $("#trainingLog");
  log.classList.toggle("hidden");
  $("#toggleTrainingLog").textContent = log.classList.contains("hidden") ? "展开日志" : "收起日志";
});

function uploadFile(file) {
  const lowerName = file.name.toLowerCase();
  const isPkl = file.name.toLowerCase().endsWith(".pkl");
  const isNpz = lowerName.endsWith(".npz");
  if (!isNpz && !isPkl) return toast("请选择 .npz 或 HHTools .pkl 文件。", true);
  if (isPkl && $("#skipValidation").checked) {
    $("#skipValidation").checked = false;
    $("#skipValidation").dispatchEvent(new Event("change"));
    toast("快速训练通道只适用于已经检查过的 NPZ；PKL 将自动转换并完整检查。", false);
  }
  const quickMode = isNpz && $("#skipValidation").checked;
  resetResult();
  $("#dropZone").classList.add("hidden");
  $("#quickTrainToggle").classList.add("hidden");
  $("#fileStrip").classList.remove("hidden");
  $(".file-icon").textContent = isPkl ? "PKL" : quickMode ? "FAST" : "NPZ";
  $("#fileName").textContent = file.name;
  $("#fileSize").textContent = formatBytes(file.size);
  $("#uploadProgress").classList.remove("hidden");

  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/upload");
  xhr.setRequestHeader("Content-Type", "application/octet-stream");
  xhr.setRequestHeader("X-Filename", encodeURIComponent(file.name));
  xhr.setRequestHeader("X-Skip-Validation", quickMode ? "true" : "false");
  xhr.upload.onprogress = (event) => {
    if (!event.lengthComputable) return;
    const percentage = Math.round((event.loaded / event.total) * 88);
    updateProgress(percentage, "正在上传动作文件…");
  };
  xhr.upload.onload = () => {
    if (isPkl) updateProgress(90, "PKL 正在转换为 50 Hz 训练 NPZ…");
    else if (!quickMode) updateProgress(90, "正在完整读取并检查 NPZ…");
  };
  xhr.onload = () => {
    try {
      const response = JSON.parse(xhr.responseText);
      if (xhr.status >= 400) throw new Error(response.error || "上传失败");
      updateProgress(
        100,
        isPkl ? "PKL 转换与 NPZ 检查完成" : quickMode ? "已跳过检查，进入训练配置" : "完整性检查完成",
      );
      setCurrentJob(response);
      loadTrainingHistory();
      setTimeout(() => {
        $("#uploadProgress").classList.add("hidden");
        renderReport(response.report);
      }, 350);
    } catch (error) { uploadFailed(error.message); }
  };
  xhr.onerror = () => uploadFailed("无法连接本地服务。");
  updateProgress(2, isPkl ? "准备安全解析 HHTools PKL…" : quickMode ? "快速上传，不读取数组…" : "准备读取…");
  xhr.send(file);
}

function uploadFailed(message) {
  updateProgress(100, "检查失败");
  toast(message, true);
  setTimeout(() => {
    $("#uploadProgress").classList.add("hidden");
    $("#fileStrip").classList.add("hidden");
    $("#dropZone").classList.remove("hidden");
    $("#quickTrainToggle").classList.remove("hidden");
  }, 1200);
}

function updateProgress(value, label) {
  $("#progressBar").style.width = `${value}%`;
  $("#progressPercent").textContent = `${value}%`;
  $("#progressLabel").textContent = label;
}

function renderReport(report, options = {}) {
  $("#reportSection").classList.remove("hidden");
  $("#renderSection").classList.remove("hidden");
  $("#trainingSection").classList.remove("hidden");
  const failCount = report.checks.filter((check) => check.status === "fail").length;
  const warnCount = report.checks.filter((check) => check.status === "warn").length;
  const verdict = $("#verdict");
  verdict.className = `verdict ${failCount ? "fail" : warnCount ? "warn" : "pass"}`;
  verdict.textContent = report.validation_skipped
    ? "VALIDATION SKIPPED"
    : failCount ? `${failCount} FAILED` : warnCount ? `PASS · ${warnCount} WARNING` : "ALL CHECKS PASSED";

  const summary = report.summary || {};
  const metrics = [
    ["FRAMES", summary.frames ?? "—", "帧"],
    ["FRAME RATE", formatNumber(summary.fps), "FPS"],
    ["DURATION", formatNumber(summary.duration_seconds), "秒"],
    ["JOINTS", summary.joint_count ?? "—", "DOF"],
    ["BODIES", summary.body_count ?? "—", "刚体"],
  ];
  $("#metrics").classList.toggle("hidden", Boolean(report.validation_skipped));
  $("#metrics").innerHTML = metrics.map(([label, value, unit]) =>
    `<div class="metric"><small>${label}</small><strong>${value}<span>${unit}</span></strong></div>`).join("");

  $("#checkCount").textContent = `${report.checks.length} ITEMS`;
  $("#checks").innerHTML = report.checks.map((check) => `
    <div class="check ${check.status}"><span class="check-light"></span><strong>${escapeHtml(check.label)}</strong><p>${escapeHtml(check.detail)}</p></div>`).join("");
  const arrayEntries = Object.entries(report.arrays || {});
  $("#arrayRows").innerHTML = report.validation_skipped
    ? '<tr><td colspan="4">本次快速通道未读取数组</td></tr>'
    : arrayEntries.map(([name, array]) => `
      <tr><td>${escapeHtml(name)}</td><td>${escapeHtml(array.shape.join(" × "))}</td><td>${escapeHtml(array.dtype)}</td><td>${formatRange(array.min, array.max)}</td></tr>`).join("");

  const renderButton = $("#renderButton");
  renderButton.disabled = !report.renderable;
  const trainButton = $("#trainButton");
  trainButton.disabled = !report.trainable;
  const restoredTraining = state.job?.training || {};
  const restoredConfig = restoredTraining.config || {};
  if (restoredConfig.num_envs && [...$("#numEnvs").options].some((option) => Number(option.value) === restoredConfig.num_envs)) {
    $("#numEnvs").value = String(restoredConfig.num_envs);
  }
  const restoredDevices = Array.isArray(restoredConfig.devices)
    ? restoredConfig.devices
    : restoredConfig.device ? [restoredConfig.device] : [];
  if (restoredDevices.length) {
    for (const option of $("#trainingDevice").options) option.selected = restoredDevices.includes(option.value);
    for (const option of $("#resumeTrainingDevices").options) option.selected = restoredDevices.includes(option.value);
  }
  updateTrainingSelectionSummary();
  if (restoredConfig.max_iterations) {
    $("#maxIterations").value = String(restoredConfig.max_iterations);
    $("#resumeIterations").value = String(restoredConfig.max_iterations);
  }
  if (restoredConfig.save_interval) $("#saveInterval").value = String(restoredConfig.save_interval);
  if (restoredConfig.num_steps_per_env) $("#numStepsPerEnv").value = String(restoredConfig.num_steps_per_env);
  if (restoredConfig.num_mini_batches) $("#numMiniBatches").value = String(restoredConfig.num_mini_batches);
  if (restoredConfig.num_learning_epochs) $("#numLearningEpochs").value = String(restoredConfig.num_learning_epochs);
  if (restoredConfig.learning_rate) $("#learningRate").value = String(restoredConfig.learning_rate);
  if (restoredConfig.desired_kl) $("#desiredKl").value = String(restoredConfig.desired_kl);
  $("#runName").value = restoredConfig.requested_run_name
    || stripTrainingAttemptSuffix(restoredTraining.run_name, state.job?.job_id)
    || suggestedRunName(state.job?.filename || report.filename, Number($("#numEnvs").value));
  $("#trainingHint").textContent = report.validation_skipped
    ? "快速通道：本次未验证 NPZ 内容，将直接使用你之前的检查结论启动训练。"
    : report.trainable
    ? "训练兼容性检查通过。点击后会创建独立 tmux 会话，关闭浏览器不会停止训练。"
    : "当前 NPZ 与本机 30-body G1 训练配置不兼容，训练按钮已禁用。";
  $("#renderHint").textContent = report.validation_skipped
    ? "快速训练通道未读取动作数组；如需生成预览，请重新上传并执行完整检查。"
    : report.renderable
    ? "检查通过，可使用当前本机 G1 模型录制完整动作周期；body 数警告只影响训练兼容性判断。"
    : "该文件未通过预览渲染所需的完整性检查，请先处理失败项目。";
  if (options.scroll !== false) $("#reportSection").scrollIntoView({ behavior: "smooth", block: "start" });
}

function stripTrainingAttemptSuffix(runName, jobId) {
  if (!runName || !jobId) return runName || "";
  return runName.replace(new RegExp(`_${jobId.slice(0, 8)}_a\\d+$`), "");
}

async function startTraining(resumeCheckpoint = null) {
  if (!state.job) return;
  const numEnvs = Number($("#numEnvs").value);
  const maxIterations = Number(resumeCheckpoint ? $("#resumeIterations").value : $("#maxIterations").value);
  const saveInterval = Number($("#saveInterval").value);
  const numStepsPerEnv = Number($("#numStepsPerEnv").value);
  const numMiniBatches = Number($("#numMiniBatches").value);
  const numLearningEpochs = Number($("#numLearningEpochs").value);
  const learningRate = Number($("#learningRate").value);
  const desiredKl = Number($("#desiredKl").value);
  const trainingDevices = selectedTrainingDevices(
    resumeCheckpoint ? $("#resumeTrainingDevices") : $("#trainingDevice"),
  );
  const runName = $("#runName").value.trim();
  if (!trainingDevices.length) return toast("请至少选择一张空闲 GPU。", true);
  if (!Number.isInteger(maxIterations) || maxIterations < 1 || maxIterations > 100000) {
    return toast("本次训练迭代数必须是 1 到 100000 之间的整数。", true);
  }
  if (!Number.isInteger(saveInterval) || saveInterval < 1 || saveInterval > 100000) {
    return toast("模型保存间隔必须是 1 到 100000 之间的整数。", true);
  }
  const numericSettings = [numStepsPerEnv, numMiniBatches, numLearningEpochs, learningRate, desiredKl];
  if (numericSettings.some((value) => !Number.isFinite(value) || value <= 0)) {
    return toast("PPO 参数必须是大于 0 的有效数字。", true);
  }
  const ppoConfig = {
    save_interval: saveInterval,
    num_steps_per_env: numStepsPerEnv,
    num_mini_batches: numMiniBatches,
    num_learning_epochs: numLearningEpochs,
    learning_rate: learningRate,
    desired_kl: desiredKl,
  };
  const validationWarning = state.job.report.validation_skipped
    ? "\n\n注意：本次使用快速通道，系统没有读取或检查 NPZ 内容。"
    : "";
  const resumeSummary = resumeCheckpoint
    ? `\n恢复点：Iteration ${resumeCheckpoint.iteration}\n来源：${resumeCheckpoint.checkpoint_name}\n本次追加迭代：${maxIterations}`
    : `\n最大迭代：${maxIterations}`;
  const confirmed = window.confirm(
    `确认${resumeCheckpoint ? "继续" : "开始"}训练？\n\n动作：${state.job.filename}\n每张 GPU 环境数：${numEnvs}\n总环境数：${numEnvs * trainingDevices.length}\n模型保存间隔：每 ${saveInterval} 次迭代\n每环境采样：${numStepsPerEnv} steps\nMini-batches：${numMiniBatches}\nLearning epochs：${numLearningEpochs}\n学习率：${learningRate}\nDesired KL：${desiredKl}${resumeSummary}\n设备：${trainingDevices.join(", ")}${validationWarning}`,
  );
  if (!confirmed) return;
  $("#trainButton").disabled = true;
  clearTrainingTimers();
  const pendingJob = {
    ...state.job,
    training: {
      status: "submitting",
      error: null,
      session: null,
      run_name: runName || suggestedRunName(state.job.filename, numEnvs, trainingDevices.length),
      pane_pid: null,
      config: {
        num_envs: numEnvs,
        max_iterations: maxIterations,
        ...ppoConfig,
        device: trainingDevices[0],
        devices: trainingDevices,
        distributed: trainingDevices.length > 1,
        world_size: trainingDevices.length,
        ...(resumeCheckpoint ? {
          resume: true,
          resume_checkpoint_id: resumeCheckpoint.id,
          resume_checkpoint: resumeCheckpoint.checkpoint_name,
          resume_iteration: resumeCheckpoint.iteration,
        } : {}),
      },
      attempt: Number(state.job.training?.attempt || 0) + 1,
      started_at: Date.now() / 1000,
      finished_at: null,
      log: "",
      tensorboard_url: state.job.training?.tensorboard_url,
    },
  };
  state.trainingStartedAt = Date.now();
  showTrainingStatus(pendingJob);
  $("#trainingLog").textContent = "正在向本地服务提交训练任务…";
  $("#trainingStatus").scrollIntoView({ behavior: "smooth" });
  try {
    const response = await fetch(`/api/jobs/${state.job.job_id}/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        num_envs: numEnvs,
        max_iterations: maxIterations,
        run_name: runName,
        devices: trainingDevices,
        ...(resumeCheckpoint ? { resume_checkpoint_id: resumeCheckpoint.id } : {}),
        ...ppoConfig,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "无法启动训练");
    setCurrentJob(payload);
    state.trainingStartedAt = timestampMilliseconds(payload.training?.started_at) || Date.now();
    showTrainingStatus(payload);
    state.trainingCheckpoints = [];
    state.checkpointRequestKey = null;
    state.trainingPollTimer = setInterval(pollTraining, 2200);
    state.trainingElapsedTimer = setInterval(updateTrainingElapsed, 1000);
  } catch (error) {
    const failedJob = {
      ...state.job,
      training: {
        ...pendingJob.training,
        status: "launch_failed",
        error: error.message,
        finished_at: Date.now() / 1000,
        log: `训练未启动：${error.message}`,
      },
    };
    showTrainingStatus(failedJob);
    $("#trainButton").disabled = !state.job.report.trainable;
    toast(error.message, true);
  }
}

async function pollTraining() {
  try {
    const response = await fetch(`/api/jobs/${state.job.job_id}`, { cache: "no-store" });
    if (response.status === 404) {
      clearTrainingTimers();
      forgetCurrentJob();
      throw new Error("训练任务已不存在，已停止轮询。请刷新页面后重新上传。");
    }
    const job = await response.json();
    if (!response.ok) throw new Error(job.error || "训练状态读取失败");
    setCurrentJob(job);
    showTrainingStatus(job);
    if (["completed", "failed", "stopped"].includes(job.training.status)) {
      clearInterval(state.trainingPollTimer);
      clearInterval(state.trainingElapsedTimer);
      state.trainingPollTimer = null;
      state.trainingElapsedTimer = null;
      $("#trainButton").disabled = !job.report.trainable;
      loadTrainingHistory();
      if (job.training.status === "failed") toast(job.training.error || "训练异常结束，请查看日志。", true);
      else toast(job.training.status === "completed" ? "训练已完成。" : "训练已停止。", false);
    }
  } catch (error) { toast(error.message, true); }
}

function showTrainingStatus(job) {
  const training = job.training || { status: "idle" };
  $("#trainingStatus").classList.remove("hidden");
  const tensorboardLink = $("#tensorboardLink");
  if (training.tensorboard_url) {
    tensorboardLink.href = training.tensorboard_url;
    tensorboardLink.textContent = `打开 TensorBoard :${new URL(training.tensorboard_url).port} ↗`;
    tensorboardLink.removeAttribute("aria-disabled");
  } else {
    tensorboardLink.href = "#";
    tensorboardLink.textContent = "TensorBoard 尚未启动";
    tensorboardLink.setAttribute("aria-disabled", "true");
  }
  const labels = {
    submitting: "正在提交训练任务", starting: "正在创建训练环境", running: "RSL-RL 正在训练",
    stopping: "正在正常停止训练", completed: "训练已完成", failed: "训练异常结束",
    launch_failed: "训练启动失败", stopped: "训练已停止",
  };
  $("#trainingStatusTitle").textContent = labels[training.status] || training.status;
  $("#trainingStatusDot").style.background = training.status === "completed"
    ? "var(--signal)" : ["failed", "launch_failed"].includes(training.status) ? "var(--red)" : "var(--orange)";
  $("#stopTrainingButton").disabled = !["starting", "running"].includes(training.status);
  $("#trainButton").disabled = ["submitting", "starting", "running", "stopping"].includes(training.status)
    || !job.report.trainable;
  const terminalTrainingState = ["completed", "failed", "launch_failed", "stopped"].includes(training.status);
  if (terminalTrainingState) {
    const checkpointRequestKey = `${job.job_id}:${training.attempt || 0}:${training.status}:${training.finished_at || 0}`;
    if (state.checkpointRequestKey !== checkpointRequestKey && !state.checkpointLoading) {
      loadTrainingCheckpoints(job, checkpointRequestKey);
    }
  } else {
    $("#resumeTrainingControls").classList.add("hidden");
  }
  const config = training.config || {};
  const configuredDevices = Array.isArray(config.devices)
    ? config.devices
    : config.device ? [config.device] : [];
  $("#trainingMeta").innerHTML = [
    training.session && `TMUX ${training.session}`,
    training.pane_pid && `PANE PID ${training.pane_pid}`,
    training.run_name && `RUN ${training.run_name}`,
    training.attempt && `ATTEMPT ${training.attempt}`,
    configuredDevices.length && `GPUS ${configuredDevices.join(", ")}`,
    config.num_envs && `${config.num_envs} ENVS / GPU`,
    config.num_envs && configuredDevices.length > 1 && `${config.num_envs * configuredDevices.length} TOTAL ENVS`,
    config.max_iterations && `${config.max_iterations} ITERATIONS`,
    config.num_steps_per_env && `${config.num_steps_per_env} STEPS / ENV`,
    config.num_mini_batches && `${config.num_mini_batches} MINI-BATCHES`,
    config.num_learning_epochs && `${config.num_learning_epochs} EPOCHS`,
    config.learning_rate && `LR ${config.learning_rate}`,
    config.desired_kl && `KL ${config.desired_kl}`,
    config.resume_iteration !== undefined && `RESUMED FROM ${config.resume_iteration}`,
  ].filter(Boolean).map((item) => `<span>${escapeHtml(item)}</span>`).join("");
  const log = $("#trainingLog");
  log.textContent = training.log || "正在等待训练进程输出…";
  log.scrollTop = log.scrollHeight;
}

async function loadTrainingCheckpoints(job, checkpointRequestKey) {
  state.checkpointLoading = true;
  state.checkpointRequestKey = checkpointRequestKey;
  const controls = $("#resumeTrainingControls");
  const select = $("#resumeCheckpoint");
  const resumeButton = $("#resumeTrainingButton");
  controls.classList.remove("hidden");
  select.replaceChildren();
  resumeButton.disabled = true;
  $("#resumeTrainingHint").textContent = "正在查找当前动作任务的 checkpoint…";
  try {
    const response = await fetch(`/api/jobs/${job.job_id}/checkpoints`, { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "无法读取 checkpoint");
    state.trainingCheckpoints = Array.isArray(payload.checkpoints) ? payload.checkpoints : [];
    for (const checkpoint of state.trainingCheckpoints) {
      const option = document.createElement("option");
      option.value = checkpoint.id;
      option.textContent = `Iteration ${checkpoint.iteration} · ${checkpoint.checkpoint_name}`;
      select.append(option);
    }
    const preferredCheckpoint = state.trainingCheckpoints.find(
      (checkpoint) => checkpoint.run_directory === state.preferredCheckpointRunDirectory,
    );
    if (preferredCheckpoint) select.value = preferredCheckpoint.id;
    state.preferredCheckpointRunDirectory = null;
    if (state.trainingCheckpoints.length) {
      resumeButton.disabled = false;
      $("#resumeTrainingHint").textContent = `找到 ${state.trainingCheckpoints.length} 个恢复点；最大迭代数将作为本次追加迭代数。`;
    } else {
      $("#resumeTrainingHint").textContent = "尚未生成可恢复的 model_N.pt checkpoint，只能重新开始训练。";
    }
  } catch (error) {
    state.trainingCheckpoints = [];
    $("#resumeTrainingHint").textContent = `checkpoint 读取失败：${error.message}`;
  } finally {
    state.checkpointLoading = false;
  }
}

function resumeTraining() {
  const checkpointId = $("#resumeCheckpoint").value;
  const checkpoint = state.trainingCheckpoints.find((candidate) => candidate.id === checkpointId);
  if (!checkpoint) return toast("请选择一个有效的 checkpoint。", true);
  return startTraining(checkpoint);
}

function clearTrainingTimers() {
  if (state.trainingPollTimer) clearInterval(state.trainingPollTimer);
  if (state.trainingElapsedTimer) clearInterval(state.trainingElapsedTimer);
  state.trainingPollTimer = null;
  state.trainingElapsedTimer = null;
}

async function stopTraining() {
  if (!state.job || !window.confirm("确认向训练会话发送 Ctrl+C 并正常停止？")) return;
  const stopButton = $("#stopTrainingButton");
  stopButton.disabled = true;
  stopButton.textContent = "正在停止…";
  try {
    const response = await fetch(`/api/jobs/${state.job.job_id}/stop-training`, { method: "POST" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "无法停止训练");
    setCurrentJob(payload);
    showTrainingStatus(payload);
    toast("已发送停止信号，正在等待训练进程退出。", false);
  } catch (error) {
    stopButton.disabled = false;
    toast(error.message, true);
  } finally {
    stopButton.textContent = "停止训练";
  }
}

async function startRender() {
  if (!state.job) return;
  const [width, height] = $("#resolution").value.split("x").map(Number);
  const cameraLayout = $("#cameraLayout").value;
  const focalLength = Number($("#focalLength").value);
  $("#renderButton").disabled = true;
  try {
    const response = await fetch(`/api/jobs/${state.job.job_id}/render`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        width, height, device: $("#device").value, camera_layout: cameraLayout, focal_length: focalLength,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "无法启动渲染");
    setCurrentJob(payload);
    state.renderStartedAt = timestampMilliseconds(payload.created_at) || Date.now();
    showRenderStatus(payload);
    state.pollTimer = setInterval(pollJob, 1800);
    state.elapsedTimer = setInterval(updateElapsed, 1000);
    $("#renderStatus").scrollIntoView({ behavior: "smooth" });
  } catch (error) {
    $("#renderButton").disabled = false;
    toast(error.message, true);
  }
}

async function pollJob() {
  try {
    const response = await fetch(`/api/jobs/${state.job.job_id}`, { cache: "no-store" });
    if (response.status === 404) {
      if (state.pollTimer) clearInterval(state.pollTimer);
      if (state.elapsedTimer) clearInterval(state.elapsedTimer);
      state.pollTimer = null;
      state.elapsedTimer = null;
      forgetCurrentJob();
      throw new Error("渲染任务已不存在，已停止轮询。请刷新页面后重新上传。");
    }
    const job = await response.json();
    if (!response.ok) throw new Error(job.error || "任务状态读取失败");
    setCurrentJob(job);
    showRenderStatus(job);
    if (["completed", "failed"].includes(job.status)) {
      clearInterval(state.pollTimer); clearInterval(state.elapsedTimer);
      state.pollTimer = null; state.elapsedTimer = null;
      if (job.status === "completed") showVideo(job);
      else { $("#renderButton").disabled = false; toast(job.error || "渲染失败，请查看日志。", true); }
    }
  } catch (error) { toast(error.message, true); }
}

function showRenderStatus(job) {
  $("#renderStatus").classList.remove("hidden");
  const labels = { queued: "等待 GPU 渲染槽位", rendering: "正在启动 Isaac Sim 并逐帧录制", completed: "预览视频已生成", failed: "渲染失败" };
  $("#statusTitle").textContent = labels[job.status] || job.status;
  $("#statusDot").style.background = job.status === "completed" ? "var(--signal)" : job.status === "failed" ? "var(--red)" : "var(--orange)";
  const log = $("#renderLog");
  log.textContent = job.log || (job.status === "queued" ? "任务已进入队列，等待当前 GPU 渲染任务结束…" : "正在等待 Isaac Sim 输出…");
  log.scrollTop = log.scrollHeight;
}

function showVideo(job, options = {}) {
  $("#videoSection").classList.remove("hidden");
  const cacheBust = `?t=${Date.now()}`;
  const video = job.video || {};
  $("#previewVideo").src = job.video_url + cacheBust;
  $("#videoFrame").classList.toggle("single-camera", video.camera_count !== 2);
  $("#downloadLink").href = `/api/jobs/${job.job_id}/download`;
  $("#videoMeta").innerHTML = [
    video.codec && `CODEC ${video.codec.toUpperCase()}`,
    video.width && `${video.width} × ${video.height}`,
    video.frames && `${video.frames} FRAMES`,
    video.duration_seconds && `${formatNumber(video.duration_seconds)} SEC`,
    video.camera_count && `${video.camera_count} CAMERAS`,
    video.focal_length_mm && `${formatNumber(video.focal_length_mm)}MM LENS`,
    video.horizontal_fov_degrees && `${formatNumber(video.horizontal_fov_degrees)}° HFOV`,
    video.size_bytes && formatBytes(video.size_bytes),
  ].filter(Boolean).map((item) => `<span>${item}</span>`).join("");
  if (options.announce !== false) {
    setTimeout(() => $("#videoSection").scrollIntoView({ behavior: "smooth" }), 250);
    toast("MP4 预览已生成。", false);
  }
}

function updateRenderControls() {
  const dual = $("#cameraLayout").value === "front_rear";
  const [width, height] = $("#resolution").value.split("x").map(Number);
  const focalLength = Number($("#focalLength").value);
  const button = $("#renderButton");
  button.querySelector("span").textContent = dual
    ? `FIXED FRONT + REAR / ${focalLength}MM / ${width * 2} × ${height}`
    : `OBLIQUE / ${focalLength}MM / ${width} × ${height}`;
  button.querySelector("strong").textContent = dual ? "生成双机位 MP4" : "生成单机位 MP4";
}

function updateElapsed() {
  if (!state.renderStartedAt) return;
  const seconds = Math.floor((Date.now() - state.renderStartedAt) / 1000);
  $("#elapsed").textContent = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}

function updateTrainingElapsed() {
  if (!state.trainingStartedAt) return;
  const seconds = Math.floor((Date.now() - state.trainingStartedAt) / 1000);
  $("#trainingElapsed").textContent = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}

function resetResult(options = {}) {
  if (state.pollTimer) clearInterval(state.pollTimer);
  if (state.elapsedTimer) clearInterval(state.elapsedTimer);
  if (state.trainingPollTimer) clearInterval(state.trainingPollTimer);
  if (state.trainingElapsedTimer) clearInterval(state.trainingElapsedTimer);
  state.job = null;
  state.renderStartedAt = null;
  state.trainingStartedAt = null;
  if (options.forgetJob !== false) forgetCurrentJob();
  ["#reportSection", "#trainingSection", "#trainingStatus", "#renderSection", "#renderStatus", "#videoSection"]
    .forEach((id) => $(id).classList.add("hidden"));
  $("#previewVideo").removeAttribute("src");
}

function setCurrentJob(job) {
  state.job = job;
  if (!job?.job_id) return;
  try { window.localStorage.setItem(JOB_STORAGE_KEY, job.job_id); } catch (_) { /* Storage can be disabled. */ }
}

function forgetCurrentJob() {
  try { window.localStorage.removeItem(JOB_STORAGE_KEY); } catch (_) { /* Storage can be disabled. */ }
}

function timestampMilliseconds(value) {
  const seconds = Number(value);
  return Number.isFinite(seconds) && seconds > 0 ? seconds * 1000 : null;
}

function startRestoredTimers(job) {
  const trainingStatus = job.training?.status;
  if (["starting", "running", "stopping"].includes(trainingStatus)) {
    state.trainingStartedAt = timestampMilliseconds(job.training.started_at) || Date.now();
    updateTrainingElapsed();
    state.trainingPollTimer = setInterval(pollTraining, 2200);
    state.trainingElapsedTimer = setInterval(updateTrainingElapsed, 1000);
  }
  if (["queued", "rendering"].includes(job.status)) {
    state.renderStartedAt = timestampMilliseconds(job.created_at) || Date.now();
    updateElapsed();
    state.pollTimer = setInterval(pollJob, 1800);
    state.elapsedTimer = setInterval(updateElapsed, 1000);
  }
}

function restoreJob(job) {
  resetResult({ forgetJob: false });
  setCurrentJob(job);
  $("#dropZone").classList.add("hidden");
  $("#quickTrainToggle").classList.add("hidden");
  $("#fileStrip").classList.remove("hidden");
  $(".file-icon").textContent = job.report?.conversion ? "PKL" : job.report?.validation_skipped ? "FAST" : "NPZ";
  $("#fileName").textContent = job.filename;
  $("#fileSize").textContent = formatBytes(job.report?.conversion?.source_size_bytes || job.report?.size_bytes);
  renderReport(job.report, { scroll: false });

  if (job.training?.status && job.training.status !== "idle") showTrainingStatus(job);
  if (["queued", "rendering", "completed", "failed"].includes(job.status)) showRenderStatus(job);
  if (job.video_url) showVideo(job, { announce: false });
  startRestoredTimers(job);
}

async function fetchJob(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (response.status === 404) return null;
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "任务状态读取失败");
  return payload;
}

async function restorePersistedJob() {
  let storedJobId = null;
  try { storedJobId = window.localStorage.getItem(JOB_STORAGE_KEY); } catch (_) { /* Storage can be disabled. */ }
  try {
    let job = storedJobId ? await fetchJob(`/api/jobs/${storedJobId}`) : null;
    if (!job) {
      if (storedJobId) forgetCurrentJob();
      job = await fetchJob("/api/active-job");
    }
    if (job) restoreJob(job);
  } catch (error) {
    toast(`恢复上次任务失败：${error.message}`, true);
  }
}

async function loadActiveTrainingJobs() {
  if (state.activeTrainingJobsLoading) return;
  state.activeTrainingJobsLoading = true;
  const select = $("#activeTrainingSelect");
  const previousJobId = select.value;
  $("#refreshActiveTrainingButton").disabled = true;
  try {
    const response = await fetch("/api/active-jobs", { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "无法读取正在训练的任务");
    state.activeTrainingJobs = Array.isArray(payload.jobs) ? payload.jobs : [];
    select.replaceChildren();
    if (!state.activeTrainingJobs.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "当前没有正在训练的任务";
      select.append(option);
      $("#activeTrainingHint").textContent = "没有检测到 Motion Inspector 管理的活动训练会话。";
      updateActiveTrainingSelection();
      return;
    }
    for (const job of state.activeTrainingJobs) {
      const option = document.createElement("option");
      const devices = Array.isArray(job.devices) && job.devices.length ? job.devices.join(", ") : "GPU 未知";
      const hasProgress = job.iteration !== null && job.iteration !== undefined
        && job.max_iterations !== null && job.max_iterations !== undefined;
      const iteration = Number(job.iteration);
      const maxIterations = Number(job.max_iterations);
      const progress = hasProgress && Number.isFinite(iteration) && Number.isFinite(maxIterations)
        ? `${iteration}/${maxIterations}`
        : "进度读取中";
      option.value = job.job_id;
      option.textContent = `[${job.status} · ${devices} · ${progress}] ${job.run_name || job.filename}`;
      select.append(option);
    }
    if (state.activeTrainingJobs.some((job) => job.job_id === previousJobId)) {
      select.value = previousJobId;
    }
    $("#activeTrainingHint").textContent = `检测到 ${state.activeTrainingJobs.length} 个正在训练的任务；请选择后停止。`;
    updateActiveTrainingSelection();
  } catch (error) {
    state.activeTrainingJobs = [];
    select.replaceChildren();
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "活动任务读取失败";
    select.append(option);
    $("#activeTrainingHint").textContent = `活动任务读取失败：${error.message}`;
    updateActiveTrainingSelection();
  } finally {
    state.activeTrainingJobsLoading = false;
    $("#refreshActiveTrainingButton").disabled = false;
  }
}

function updateActiveTrainingSelection() {
  const job = state.activeTrainingJobs.find(
    (candidate) => candidate.job_id === $("#activeTrainingSelect").value,
  );
  $("#stopSelectedTrainingButton").disabled = !job || job.status === "stopping";
}

async function stopSelectedTraining() {
  const jobId = $("#activeTrainingSelect").value;
  const job = state.activeTrainingJobs.find((candidate) => candidate.job_id === jobId);
  if (!job) return toast("请先选择一个正在训练的任务。", true);
  const label = job.run_name || job.filename;
  if (!window.confirm(`确认停止训练任务“${label}”？`)) return;

  const button = $("#stopSelectedTrainingButton");
  button.disabled = true;
  button.textContent = "正在停止…";
  try {
    const response = await fetch(`/api/jobs/${jobId}/stop-training`, { method: "POST" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "无法停止训练");
    if (state.job?.job_id === jobId) {
      setCurrentJob(payload);
      showTrainingStatus(payload);
    }
    toast(`已向“${label}”发送停止信号。`, false);
    await loadActiveTrainingJobs();
    loadSystemInfo();
  } catch (error) {
    toast(error.message, true);
    updateActiveTrainingSelection();
  } finally {
    button.textContent = "停止所选任务";
  }
}

async function loadTrainingHistory() {
  if (state.historyLoading) return;
  state.historyLoading = true;
  const select = $("#trainingHistorySelect");
  const openButton = $("#openTrainingHistoryButton");
  const exportButton = $("#exportTrainingPackageButton");
  const previousRunDirectory = select.value;
  const currentJobId = state.job?.job_id || "";
  select.replaceChildren();
  const loadingOption = document.createElement("option");
  loadingOption.value = "";
  loadingOption.textContent = "正在读取历史任务…";
  select.append(loadingOption);
  openButton.disabled = true;
  exportButton.disabled = true;
  try {
    const response = await fetch("/api/training-runs", { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "无法读取历史任务");
    state.historicalJobs = Array.isArray(payload.runs) ? payload.runs : [];
    select.replaceChildren();
    if (!state.historicalJobs.length) {
      const emptyOption = document.createElement("option");
      emptyOption.value = "";
      emptyOption.textContent = "暂无已保存的训练任务";
      select.append(emptyOption);
      $("#trainingHistoryHint").textContent = "尚未找到持久化训练任务。";
      return;
    }
    for (const trainingRun of state.historicalJobs) {
      const option = document.createElement("option");
      const iterationLabel = trainingRun.latest_iteration !== null
        ? `iter ${trainingRun.latest_iteration}`
        : "无 checkpoint";
      const resumableLabel = trainingRun.resumable
        ? "可续训"
        : trainingRun.missing_motion_filename
          ? `待补动作 ${trainingRun.missing_motion_filename}`
          : "外部训练 Run";
      option.value = trainingRun.run_directory;
      option.textContent = `[${resumableLabel} · ${iterationLabel}] ${trainingRun.run_directory}`;
      select.append(option);
    }
    if ([...select.options].some((option) => option.value === previousRunDirectory)) {
      select.value = previousRunDirectory;
    } else {
      const currentRun = state.historicalJobs.find((trainingRun) => trainingRun.job_id === currentJobId);
      if (currentRun) select.value = currentRun.run_directory;
    }
    $("#trainingHistoryHint").textContent = `找到 ${state.historicalJobs.length} 个训练 Run；可续训项打开后会默认选择该 Run 的最新 checkpoint。`;
    updateTrainingHistorySelection();
  } catch (error) {
    state.historicalJobs = [];
    select.replaceChildren();
    const errorOption = document.createElement("option");
    errorOption.value = "";
    errorOption.textContent = "历史任务读取失败";
    select.append(errorOption);
    $("#trainingHistoryHint").textContent = `历史任务读取失败：${error.message}`;
  } finally {
    state.historyLoading = false;
  }
}

function updateTrainingHistorySelection() {
  const runDirectory = $("#trainingHistorySelect").value;
  const trainingRun = state.historicalJobs.find((candidate) => candidate.run_directory === runDirectory);
  const exportable = Boolean(
    trainingRun?.resumable
    && trainingRun.latest_checkpoint_name
    && /^[0-9a-f]{32}$/.test(trainingRun.job_id || ""),
  );
  $("#openTrainingHistoryButton").disabled = !exportable;
  $("#exportTrainingPackageButton").disabled = !exportable;
  if (trainingRun && !trainingRun.resumable) {
    $("#trainingHistoryHint").textContent = trainingRun.missing_motion_filename
      ? `该 Run 的 checkpoint 完整，但原动作文件已丢失。请在下方重新上传 ${trainingRun.missing_motion_filename}，系统会自动关联并开放续训。`
      : "该 Run 来自网页外部，无法确认动作归属；当前只展示磁盘 checkpoint，不能从网页续训。";
  }
}

function exportSelectedTrainingPackage() {
  const runDirectory = $("#trainingHistorySelect").value;
  const trainingRun = state.historicalJobs.find((candidate) => candidate.run_directory === runDirectory);
  if (
    !trainingRun?.resumable
    || !trainingRun.latest_checkpoint_name
    || !/^[0-9a-f]{32}$/.test(trainingRun.job_id || "")
  ) {
    return toast("请选择一个可续训且包含 checkpoint 的训练 Run。", true);
  }
  const query = new URLSearchParams({ run_directory: trainingRun.run_directory });
  window.location.assign(`/api/jobs/${trainingRun.job_id}/training-package?${query}`);
}

function updateTrainingPackageProgress(value, message, status = "") {
  const percentage = Math.max(0, Math.min(100, Math.round(value)));
  $("#trainingPackageStatusText").textContent = message;
  $("#trainingPackageProgressBar").style.width = `${percentage}%`;
  $("#trainingPackageProgress").setAttribute("aria-valuenow", String(percentage));
  $("#trainingPackageStatus").classList.toggle("error", status === "error");
  $("#trainingPackageStatus").classList.toggle("success", status === "success");
}

function importTrainingPackage(file) {
  if (state.trainingPackageImporting) return;
  if (!file.name.toLowerCase().endsWith(".zip")) {
    return toast("请选择 Motion Inspector 导出的 .zip 训练恢复包。", true);
  }

  state.trainingPackageImporting = true;
  trainingPackageDropZone.classList.add("busy");
  updateTrainingPackageProgress(2, `准备导入 ${file.name}`);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/training-package/import");
  xhr.setRequestHeader("Content-Type", "application/zip");
  xhr.setRequestHeader("X-Filename", encodeURIComponent(file.name));
  xhr.upload.onprogress = (event) => {
    if (event.lengthComputable) {
      updateTrainingPackageProgress((event.loaded / event.total) * 85, "正在上传训练恢复包…");
    }
  };
  xhr.upload.onload = () => updateTrainingPackageProgress(88, "正在校验并安装新任务…");
  xhr.onload = async () => {
    try {
      const payload = JSON.parse(xhr.responseText);
      if (xhr.status >= 400) throw new Error(payload.error || "训练恢复包导入失败");
      if (!payload.job_id || !payload.imported_run_directory) {
        throw new Error("训练恢复包导入响应不完整。");
      }
      state.preferredCheckpointRunDirectory = payload.imported_run_directory;
      restoreJob(payload);
      await loadTrainingHistory();
      if ([...$("#trainingHistorySelect").options].some(
        (option) => option.value === payload.imported_run_directory,
      )) {
        $("#trainingHistorySelect").value = payload.imported_run_directory;
        updateTrainingHistorySelection();
      }
      updateTrainingPackageProgress(100, `已创建新 Run：${payload.imported_run_directory}`, "success");
      toast(`训练恢复包已导入：${payload.filename}`, false);
      $("#trainingSection").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
      updateTrainingPackageProgress(100, error.message, "error");
      toast(error.message, true);
    } finally {
      state.trainingPackageImporting = false;
      trainingPackageDropZone.classList.remove("busy");
    }
  };
  xhr.onerror = () => {
    state.trainingPackageImporting = false;
    trainingPackageDropZone.classList.remove("busy");
    updateTrainingPackageProgress(100, "无法连接本地服务。", "error");
    toast("无法连接本地服务。", true);
  };
  xhr.send(file);
}

async function openSelectedHistoricalJob() {
  const runDirectory = $("#trainingHistorySelect").value;
  const trainingRun = state.historicalJobs.find((candidate) => candidate.run_directory === runDirectory);
  const jobId = trainingRun?.job_id || "";
  if (!trainingRun?.resumable || !/^[0-9a-f]{32}$/.test(jobId)) {
    const message = trainingRun?.missing_motion_filename
      ? `请先重新上传原动作文件 ${trainingRun.missing_motion_filename}，系统会自动恢复该 Run 的续训入口。`
      : "这个训练 Run 没有关联可恢复的网页任务。";
    return toast(message, true);
  }
  const currentTrainingStatus = state.job?.training?.status;
  if (["submitting", "starting", "running", "stopping"].includes(currentTrainingStatus)) {
    const confirmed = window.confirm("切换页面当前查看的任务不会停止后台训练。确认打开其他历史任务吗？");
    if (!confirmed) return;
  }
  const openButton = $("#openTrainingHistoryButton");
  openButton.disabled = true;
  openButton.textContent = "正在打开…";
  try {
    state.preferredCheckpointRunDirectory = trainingRun.run_directory;
    const job = await fetchJob(`/api/jobs/${jobId}`);
    if (!job) throw new Error("该历史任务已不存在。")
    restoreJob(job);
    toast(`已打开历史任务：${job.filename}`, false);
    $("#trainingSection").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    toast(error.message, true);
    await loadTrainingHistory();
  } finally {
    openButton.textContent = "打开任务";
    updateTrainingHistorySelection();
  }
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "—";
  const units = ["B", "KiB", "MiB", "GiB"];
  let value = bytes, unit = 0;
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit += 1; }
  return `${value.toFixed(unit ? 1 : 0)} ${units[unit]}`;
}
function formatNumber(value) { return Number.isFinite(value) ? Number(value.toFixed(3)).toString() : "—"; }
function formatRange(min, max) { return Number.isFinite(min) && Number.isFinite(max) ? `${min.toPrecision(3)} … ${max.toPrecision(3)}` : "—"; }
function escapeHtml(value) { const div = document.createElement("div"); div.textContent = String(value); return div.innerHTML; }
function selectedTrainingDevices(select = $("#trainingDevice")) {
  return [...select.selectedOptions]
    .filter((option) => !option.disabled)
    .map((option) => option.value);
}

function suggestedRunName(filename, numEnvs, gpuCount = Math.max(1, selectedTrainingDevices().length)) {
  const stem = String(filename || "motion").replace(/\.npz$/i, "").toLowerCase();
  const safe = stem.replace(/[^a-z0-9_-]+/g, "_").replace(/^[_-]+|[_-]+$/g, "") || "motion";
  return `${safe}_local_${gpuCount}gpu_${numEnvs}`.slice(0, 72);
}

function updateTrainingSelectionSummary() {
  const gpuCount = Math.max(1, selectedTrainingDevices().length);
  $("#trainingMode").textContent = `Headless / ${gpuCount} GPU${gpuCount > 1 ? "s / DDP" : ""}`;
  const runName = $("#runName");
  if (/_local_\d+gpu_\d+$/.test(runName.value)) {
    runName.value = runName.value.replace(/_local_\d+gpu_\d+$/, `_local_${gpuCount}gpu_${$("#numEnvs").value}`);
  }
}

function populateGpuSelect(select, gpus, activeTrainingDevices, includeCpu) {
  const previousValues = new Set([...select.selectedOptions].map((option) => option.value));
  const restoredConfig = state.job?.training?.config || {};
  const restoredDevices = Array.isArray(restoredConfig.devices)
    ? restoredConfig.devices
    : restoredConfig.device ? [restoredConfig.device] : [];
  const preferredValues = new Set(restoredDevices.length ? restoredDevices : previousValues);
  select.replaceChildren();

  for (const gpu of gpus) {
    const option = document.createElement("option");
    const device = `cuda:${gpu.index}`;
    const memoryGib = Math.round(Number(gpu.memory_mib) / 1024);
    const freeMemoryGib = Number(gpu.memory_free_mib) / 1024;
    const shortName = String(gpu.name).replace(/^NVIDIA\s+/, "").replace(/^GeForce\s+/, "");
    const isTraining = activeTrainingDevices.has(device);
    option.value = device;
    option.disabled = isTraining;
    option.selected = preferredValues.has(device) && (!isTraining || restoredDevices.includes(device));
    option.textContent = `CUDA : ${gpu.index} · ${shortName} · ${memoryGib} GB · ${freeMemoryGib.toFixed(1)} GB FREE${isTraining ? " · 训练中" : ""}`;
    select.append(option);
  }

  if (includeCpu) {
    const cpuOption = document.createElement("option");
    cpuOption.value = "cpu";
    cpuOption.textContent = "CPU";
    select.append(cpuOption);
  }

  const availableSelectedOptions = [...select.selectedOptions].filter((option) => !option.disabled);
  if (!availableSelectedOptions.length && !restoredDevices.length) {
    const firstAvailableOption = [...select.options].find((option) => !option.disabled);
    if (firstAvailableOption) firstAvailableOption.selected = true;
  }
  if (!select.multiple) {
    const selectedOption = [...select.options].find((option) => option.selected && !option.disabled);
    const firstAvailableOption = [...select.options].find((option) => !option.disabled);
    select.value = (selectedOption || firstAvailableOption)?.value || "";
  } else if (select.id === "trainingDevice") {
    updateTrainingSelectionSummary();
  }
}

async function loadSystemInfo() {
  if (state.gpuMonitorLoading) return;
  state.gpuMonitorLoading = true;
  const abortController = new AbortController();
  const timeoutId = setTimeout(() => abortController.abort(), 8000);
  try {
    const response = await fetch("/api/system-info", {
      cache: "no-store",
      signal: abortController.signal,
    });
    const responseText = await response.text();
    let payload;
    try {
      payload = JSON.parse(responseText);
    } catch {
      throw new Error(`系统信息接口返回了无效响应（HTTP ${response.status}）`);
    }
    if (!response.ok) throw new Error(payload.error || "系统信息读取失败");
    const gpus = Array.isArray(payload.gpus) ? payload.gpus : [];
    const activeTrainingDevices = new Set(payload.active_training_devices || []);
    state.gpuMonitorHasData = true;
    state.gpuMonitorFailureCount = 0;
    renderGpuMonitor(gpus, Boolean(payload.stale), Boolean(payload.refreshing));
    loadActiveTrainingJobs();
    populateGpuSelect($("#trainingDevice"), gpus, activeTrainingDevices, false);
    populateGpuSelect($("#resumeTrainingDevices"), gpus, activeTrainingDevices, false);
    populateGpuSelect($("#device"), gpus, activeTrainingDevices, false);

    if (!gpus.length) {
      $("#gpuBadgeBrand").textContent = "GPU";
      $("#gpuBadgeModel").textContent = payload.refreshing ? "读取中" : "不可用";
      $("#gpuBadgeMeta").textContent = payload.refreshing
        ? "NVIDIA-SMI 后台采集中"
        : "NVIDIA-SMI 未检测到设备";
      $("#systemStateText").textContent = "ISAAC LAB / CPU";
      return;
    }

    const names = [...new Set(gpus.map((gpu) => String(gpu.name)))];
    const displayName = names.length === 1
      ? names[0].replace(/^NVIDIA\s+/, "").replace(/^GeForce\s+/, "")
      : "MIXED GPU";
    const nameParts = displayName.split(/\s+/, 2);
    const memoryValues = [...new Set(gpus.map((gpu) => Math.round(Number(gpu.memory_mib) / 1024)))];
    const indexes = gpus.map((gpu) => Number(gpu.index)).sort((a, b) => a - b);
    const indexLabel = indexes.length > 1 ? `${indexes[0]}–${indexes[indexes.length - 1]}` : `${indexes[0]}`;

    $("#gpuBadgeBrand").textContent = `${gpus.length > 1 ? `${gpus.length} × ` : ""}${nameParts[0]}`;
    $("#gpuBadgeModel").textContent = nameParts[1] || displayName;
    $("#gpuBadgeMeta").textContent = `${memoryValues.length === 1 ? `${memoryValues[0]} GB EACH` : "MIXED MEMORY"} / CUDA ${indexLabel}`;
    $("#systemStateText").textContent = `${gpus.length} GPU / ISAAC LAB`;
  } catch (error) {
    state.gpuMonitorFailureCount += 1;
    const errorMessage = error?.name === "AbortError"
      ? "显卡状态请求超时"
      : (error?.message || "显卡状态请求失败");
    $("#gpuUpdatedAt").textContent = `刷新失败 · 将自动重试（${state.gpuMonitorFailureCount}）`;
    if (state.gpuMonitorHasData) {
      $("#gpuBadgeMeta").textContent = `${errorMessage} · 保留上次数据`;
    } else {
      $("#gpuBadgeModel").textContent = "未知";
      $("#gpuBadgeMeta").textContent = errorMessage;
      $("#gpuCards").innerHTML = `<div class="gpu-monitor-empty error">${escapeHtml(errorMessage)}</div>`;
    }
  } finally {
    clearTimeout(timeoutId);
    state.gpuMonitorLoading = false;
  }
}

function renderGpuMonitor(gpus, stale = false, refreshing = false) {
  const cards = $("#gpuCards");
  const updateTime = new Date().toLocaleTimeString("zh-CN", { hour12: false });
  $("#gpuUpdatedAt").textContent = refreshing
    ? `后台更新中 · ${updateTime}`
    : stale ? `缓存数据 · ${updateTime}` : `更新于 ${updateTime}`;
  if (!gpus.length) {
    cards.innerHTML = refreshing
      ? '<div class="gpu-monitor-empty">正在后台读取 NVIDIA-SMI，页面无需等待。</div>'
      : '<div class="gpu-monitor-empty">NVIDIA-SMI 未检测到可用 GPU。</div>';
    return;
  }

  cards.innerHTML = gpus.map((gpu) => {
    const totalMemory = Number(gpu.memory_mib);
    const usedMemory = Number(gpu.memory_used_mib);
    const freeMemory = Number(gpu.memory_free_mib);
    const utilization = Number(gpu.utilization_percent);
    const memoryPercentage = Number.isFinite(totalMemory) && totalMemory > 0 && Number.isFinite(usedMemory)
      ? Math.min(100, Math.max(0, (usedMemory / totalMemory) * 100))
      : 0;
    const processes = Array.isArray(gpu.processes) ? gpu.processes : [];
    const trainingRuns = Array.isArray(gpu.training_runs) ? gpu.training_runs : [];
    const processRows = processes.length
      ? processes.map((process) => `
          <div class="gpu-process">
            <span>PID ${escapeHtml(process.pid)}</span>
            <strong title="${escapeHtml(process.name)}">${escapeHtml(shortProcessName(process.name))}</strong>
            <em>${formatMib(process.used_memory_mib)}</em>
          </div>`).join("")
      : '<div class="gpu-process empty">无计算进程</div>';
    const trainingRows = trainingRuns.map((trainingRun) => {
      const iteration = Number(trainingRun.iteration);
      const maxIterations = Number(trainingRun.max_iterations);
      const iterationLabel = Number.isFinite(iteration) && Number.isFinite(maxIterations)
        ? `ITER ${iteration} / ${maxIterations}`
        : "正在读取训练轮数";
      return `
        <div class="gpu-training-run">
          <small>RUN NAME</small>
          <strong title="${escapeHtml(trainingRun.run_name)}">${escapeHtml(trainingRun.run_name)}</strong>
          <em>${escapeHtml(iterationLabel)}</em>
        </div>`;
    }).join("");
    const shortName = String(gpu.name).replace(/^NVIDIA\s+/, "").replace(/^GeForce\s+/, "");
    return `
      <article class="gpu-card">
        <div class="gpu-card-heading">
          <div><small>GPU ${escapeHtml(gpu.index)}</small><strong>${escapeHtml(shortName)}</strong></div>
          <span>${formatMetric(gpu.temperature_celsius, "°C")}</span>
        </div>
        <div class="gpu-memory-label">
          <span>显存 ${formatMib(usedMemory)} / ${formatMib(totalMemory)}</span>
          <strong>${memoryPercentage.toFixed(1)}%</strong>
        </div>
        <div class="gpu-memory-track"><span style="width: ${memoryPercentage.toFixed(1)}%"></span></div>
        <div class="gpu-stat-grid">
          <div><small>GPU UTIL</small><strong>${formatMetric(utilization, "%")}</strong></div>
          <div><small>FREE VRAM</small><strong>${formatMib(freeMemory)}</strong></div>
          <div><small>POWER</small><strong>${formatPower(gpu.power_draw_watts, gpu.power_limit_watts)}</strong></div>
        </div>
        <div class="gpu-processes"><small>COMPUTE PROCESSES</small>${processRows}</div>
        ${trainingRows ? `<div class="gpu-training"><small>ACTIVE TRAINING</small>${trainingRows}</div>` : ""}
      </article>`;
  }).join("");
}

function formatMib(value) {
  const memoryMib = Number(value);
  if (!Number.isFinite(memoryMib)) return "—";
  return memoryMib >= 1024 ? `${(memoryMib / 1024).toFixed(1)} GiB` : `${Math.round(memoryMib)} MiB`;
}

function formatMetric(value, unit) {
  const numericValue = Number(value);
  return Number.isFinite(numericValue) ? `${Math.round(numericValue)}${unit}` : "—";
}

function formatPower(draw, limit) {
  const powerDraw = Number(draw);
  const powerLimit = Number(limit);
  if (!Number.isFinite(powerDraw)) return "—";
  return Number.isFinite(powerLimit) ? `${powerDraw.toFixed(0)} / ${powerLimit.toFixed(0)} W` : `${powerDraw.toFixed(0)} W`;
}

function shortProcessName(processName) {
  const normalizedName = String(processName || "unknown").replace(/\\/g, "/");
  return normalizedName.split("/").pop() || normalizedName;
}
function toast(message, error = false) {
  const element = $("#toast"); element.textContent = message; element.className = `toast show${error ? " error" : ""}`;
  clearTimeout(toast.timer); toast.timer = setTimeout(() => element.className = "toast", 3200);
}

updateRenderControls();
loadSystemInfo();
state.gpuMonitorTimer = setInterval(loadSystemInfo, 5000);
loadTrainingHistory();
restorePersistedJob();
