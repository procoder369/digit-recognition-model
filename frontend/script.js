/**
 * Digit Recognition Web App
 * Interactive Canvas & FastAPI Inference Client
 */

document.addEventListener("DOMContentLoaded", () => {
  // --- DOM Elements ---
  const canvas = document.getElementById("digitCanvas");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const canvasWrapper = document.querySelector(".canvas-wrapper");
  const placeholder = document.getElementById("canvasPlaceholder");
  const strokeRange = document.getElementById("strokeRange");
  const strokeVal = document.getElementById("strokeVal");
  const clearBtn = document.getElementById("clearBtn");
  const predictBtn = document.getElementById("predictBtn");
  const realtimeToggle = document.getElementById("realtimeToggle");
  const sampleBtns = document.querySelectorAll(".sample-btn");

  const healthBadge = document.getElementById("healthBadge");
  const healthText = document.getElementById("healthText");
  const predDigit = document.getElementById("predDigit");
  const predSubtext = document.getElementById("predSubtext");
  const confPct = document.getElementById("confPct");
  const grid8x8 = document.getElementById("grid8x8");
  const probList = document.getElementById("probList");
  const latencyVal = document.getElementById("latencyVal");
  const knnModeLabel = document.getElementById("knnModeLabel");

  // --- State ---
  let isDrawing = false;
  let hasDrawn = false;
  let strokeWidth = parseInt(strokeRange.value, 10);
  let debounceTimer = null;
  let lastX = 0;
  let lastY = 0;

  // --- Initialize Canvas ---
  function initCanvas() {
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#FFFFFF";
    ctx.fillStyle = "#FFFFFF";
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = strokeWidth;
  }

  // --- Initialize Diagnostics UI ---
  function initDiagnostics() {
    // Build 64-cell 8x8 grid
    grid8x8.innerHTML = "";
    for (let i = 0; i < 64; i++) {
      const cell = document.createElement("div");
      cell.className = "grid-cell";
      cell.title = `Pixel ${i} (r: ${Math.floor(i / 8)}, c: ${i % 8})`;
      grid8x8.appendChild(cell);
    }

    // Build 10 probability rows
    probList.innerHTML = "";
    for (let d = 0; d < 10; d++) {
      const row = document.createElement("div");
      row.className = "prob-row";
      row.id = `prob-row-${d}`;
      row.innerHTML = `
        <span class="prob-digit">${d}</span>
        <div class="prob-bar-bg">
          <div class="prob-bar-fill" id="prob-fill-${d}" style="width: 0%"></div>
        </div>
        <span class="prob-val" id="prob-val-${d}">0.0%</span>
      `;
      probList.appendChild(row);
    }
  }

  // --- Health Check ---
  async function checkHealth() {
    try {
      const res = await fetch("/health");
      if (res.ok) {
        const data = await res.json();
        healthBadge.className = "health-badge online";
        healthText.textContent = `FastAPI Online (${data.model_type || "KNN Ready"})`;

        // Reflect the model's real hyperparameters instead of a hardcoded guess.
        if (knnModeLabel && data.n_neighbors) {
          const weightLabel = data.weights === "distance" ? "Distance-Weighted" : "Uniform";
          knnModeLabel.textContent = `K=${data.n_neighbors} ${weightLabel}`;
        }
      } else {
        throw new Error("Health check returned status " + res.status);
      }
    } catch (err) {
      healthBadge.className = "health-badge error";
      healthText.textContent = "Backend Offline";
      console.warn("Backend health check failed:", err);
    }
  }

  // --- Drawing Handlers (Pointer Events) ---
  function getCanvasCoords(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (e.clientX - rect.x) * scaleX,
      y: (e.clientY - rect.y) * scaleY
    };
  }

  function startDrawing(e) {
    e.preventDefault();
    isDrawing = true;
    hasDrawn = true;
    placeholder.classList.add("hidden");
    canvasWrapper.classList.add("drawing");

    const { x, y } = getCanvasCoords(e);
    lastX = x;
    lastY = y;

    // Draw single point/dot if clicked
    ctx.beginPath();
    ctx.arc(x, y, strokeWidth / 2, 0, Math.PI * 2);
    ctx.fill();
  }

  function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();

    const { x, y } = getCanvasCoords(e);

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();

    lastX = x;
    lastY = y;

    if (realtimeToggle.checked) {
      triggerDebouncedPrediction(150);
    }
  }

  function stopDrawing(e) {
    if (!isDrawing) return;
    isDrawing = false;
    canvasWrapper.classList.remove("drawing");

    if (realtimeToggle.checked || hasDrawn) {
      triggerDebouncedPrediction(80);
    }
  }

  function triggerDebouncedPrediction(delay = 100) {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      predict();
    }, delay);
  }

  // --- Clear Canvas ---
  function clearCanvas() {
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    hasDrawn = false;
    placeholder.classList.remove("hidden");
    resetPredictionUI();
  }

  function resetPredictionUI() {
    predDigit.textContent = "-";
    predDigit.style.color = "#38bdf8";
    predSubtext.textContent = "Draw a digit on the canvas to begin";
    confPct.textContent = "0%";

    // Reset 8x8 grid
    const cells = grid8x8.querySelectorAll(".grid-cell");
    cells.forEach(cell => {
      cell.style.backgroundColor = "#000000";
    });

    // Reset probabilities
    for (let d = 0; d < 10; d++) {
      const row = document.getElementById(`prob-row-${d}`);
      const fill = document.getElementById(`prob-fill-${d}`);
      const val = document.getElementById(`prob-val-${d}`);
      if (row) row.classList.remove("winner");
      if (fill) fill.style.width = "0%";
      if (val) val.textContent = "0.0%";
    }
  }

  // --- Predict Digit via FastAPI Backend ---
  async function predict() {
    if (!hasDrawn) return;

    const startTime = performance.now();
    const imageData = canvas.toDataURL("image/png");

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageData })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Prediction request failed");
      }

      const result = await response.json();
      const elapsed = Math.round(performance.now() - startTime);
      latencyVal.textContent = `${elapsed} ms`;

      updateUIWithResult(result);
    } catch (err) {
      console.error("Prediction error:", err);
      predSubtext.textContent = `Error: ${err.message}`;
    }
  }

  // --- Update UI with Model Output ---
  function updateUIWithResult(result) {
    const { prediction, confidence, probabilities, preprocessed_grid } = result;

    if (prediction === -1) {
      resetPredictionUI();
      return;
    }

    // 1. Hero Digit
    predDigit.textContent = prediction;
    const confidencePct = Math.round(confidence * 100);
    confPct.textContent = `${confidencePct}%`;
    predSubtext.textContent = `Predicted: ${prediction} with ${confidencePct}% confidence`;

    // 2. Update 8×8 Downsampled Vision Grid
    if (preprocessed_grid && preprocessed_grid.length === 8) {
      const cells = grid8x8.querySelectorAll(".grid-cell");
      let cellIdx = 0;
      for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
          const val = preprocessed_grid[r][c]; // 0.0 to 16.0
          const intensity = Math.min(255, Math.round((val / 16.0) * 255));
          if (cells[cellIdx]) {
            cells[cellIdx].style.backgroundColor = `rgb(${intensity}, ${intensity}, ${intensity})`;
            cells[cellIdx].title = `Pos (${r},${c}): ${val} / 16`;
          }
          cellIdx++;
        }
      }
    }

    // 3. Update Probabilities Bar Chart
    if (probabilities && probabilities.length === 10) {
      probabilities.forEach((prob, digit) => {
        const row = document.getElementById(`prob-row-${digit}`);
        const fill = document.getElementById(`prob-fill-${digit}`);
        const val = document.getElementById(`prob-val-${digit}`);

        const pct = (prob * 100).toFixed(1);
        if (fill) fill.style.width = `${pct}%`;
        if (val) val.textContent = `${pct}%`;

        if (row) {
          if (digit === prediction) {
            row.classList.add("winner");
          } else {
            row.classList.remove("winner");
          }
        }
      });
    }
  }

  // --- Preset Digit Drawings (Vector Strokes) ---
  const DIGIT_PRESETS = {
    0: (ctx) => {
      ctx.beginPath();
      ctx.ellipse(140, 140, 50, 75, 0, 0, Math.PI * 2);
      ctx.stroke();
    },
    1: (ctx) => {
      ctx.beginPath();
      ctx.moveTo(115, 80);
      ctx.lineTo(140, 60);
      ctx.lineTo(140, 220);
      ctx.moveTo(105, 220);
      ctx.lineTo(175, 220);
      ctx.stroke();
    },
    2: (ctx) => {
      ctx.beginPath();
      ctx.arc(140, 100, 45, Math.PI, 0, false);
      ctx.lineTo(95, 215);
      ctx.lineTo(185, 215);
      ctx.stroke();
    },
    3: (ctx) => {
      ctx.beginPath();
      ctx.arc(140, 100, 40, -Math.PI / 2, Math.PI / 2, false);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(140, 175, 45, -Math.PI / 2, Math.PI / 2, false);
      ctx.stroke();
    },
    4: (ctx) => {
      ctx.beginPath();
      ctx.moveTo(165, 60);
      ctx.lineTo(95, 170);
      ctx.lineTo(185, 170);
      ctx.moveTo(165, 60);
      ctx.lineTo(165, 220);
      ctx.stroke();
    },
    5: (ctx) => {
      ctx.beginPath();
      ctx.moveTo(180, 65);
      ctx.lineTo(105, 65);
      ctx.lineTo(100, 125);
      ctx.arc(140, 165, 45, -Math.PI / 1.8, Math.PI / 1.8, false);
      ctx.stroke();
    },
    6: (ctx) => {
      ctx.beginPath();
      ctx.moveTo(170, 75);
      ctx.arc(140, 160, 50, 0, Math.PI * 2);
      ctx.moveTo(170, 75);
      ctx.quadraticCurveTo(100, 90, 95, 170);
      ctx.stroke();
    },
    7: (ctx) => {
      ctx.beginPath();
      ctx.moveTo(95, 70);
      ctx.lineTo(185, 70);
      ctx.lineTo(115, 220);
      ctx.stroke();
    },
    8: (ctx) => {
      ctx.beginPath();
      ctx.arc(140, 105, 38, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(140, 175, 48, 0, Math.PI * 2);
      ctx.stroke();
    },
    9: (ctx) => {
      ctx.beginPath();
      ctx.arc(140, 115, 45, 0, Math.PI * 2);
      ctx.moveTo(185, 115);
      ctx.quadraticCurveTo(185, 190, 115, 220);
      ctx.stroke();
    }
  };

  function loadSampleDigit(digit) {
    clearCanvas();
    hasDrawn = true;
    placeholder.classList.add("hidden");

    ctx.save();
    ctx.lineWidth = strokeWidth + 4;
    ctx.strokeStyle = "#FFFFFF";
    ctx.fillStyle = "#FFFFFF";

    if (DIGIT_PRESETS[digit]) {
      DIGIT_PRESETS[digit](ctx);
    }
    ctx.restore();

    predict();
  }

  // --- Event Listeners ---
  canvas.addEventListener("pointerdown", startDrawing);
  window.addEventListener("pointermove", draw);
  window.addEventListener("pointerup", stopDrawing);
  window.addEventListener("pointercancel", stopDrawing);

  clearBtn.addEventListener("click", clearCanvas);
  predictBtn.addEventListener("click", predict);

  strokeRange.addEventListener("input", (e) => {
    strokeWidth = parseInt(e.target.value, 10);
    strokeVal.textContent = `${strokeWidth}px`;
    ctx.lineWidth = strokeWidth;
  });

  sampleBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const digit = parseInt(btn.dataset.digit, 10);
      loadSampleDigit(digit);
    });
  });

  // Keyboard shortcut: Ctrl + Enter to Predict, Escape to Clear
  window.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      predict();
    } else if (e.key === "Escape") {
      clearCanvas();
    }
  });

  // Initialize
  initCanvas();
  initDiagnostics();
  checkHealth();
  setInterval(checkHealth, 10000);
});
