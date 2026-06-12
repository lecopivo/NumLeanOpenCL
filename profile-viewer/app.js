const rawInput = document.getElementById("rawInput");
const fileInput = document.getElementById("fileInput");
const parseButton = document.getElementById("parse");
const loadSampleButton = document.getElementById("loadSample");
const clearButton = document.getElementById("clear");
const kindFilter = document.getElementById("kindFilter");
const labelFilter = document.getElementById("labelFilter");
const compressGaps = document.getElementById("compressGaps");
const resetZoom = document.getElementById("resetZoom");
const zoomInfo = document.getElementById("zoomInfo");
const timeline = document.getElementById("timeline");
const table = document.getElementById("eventsTable");
const stats = document.getElementById("stats");

const columns = [
  "idx",
  "kind",
  "label",
  "work_items",
  "bytes",
  "queued_us",
  "submit_us",
  "start_us",
  "end_us",
  "queue_to_submit_us",
  "submit_to_start_us",
  "duration_us",
  "host_start_us",
  "host_end_us",
  "host_duration_us",
];

const numericColumns = new Set(columns.filter((c) => c !== "label" && c !== "kind"));
let rows = [];
let sortState = { column: "idx", direction: "asc" };
let zoomDomain = null;
let currentTimelineScale = null;
let brushStartPx = null;
let brushChart = null;
let brushTrack = null;

const sample = `profile workload: n=65536, salt=3, output_size=65536
sample: y[0]=11.128922, y[n/2]=13732.791016, y[n-1]=8517.594727
OpenCL profile enabled: 1
OpenCL profile record calls: 31
OpenCL profile null enqueue events: 0
OpenCL profile events: 6
idx,kind,label,work_items,bytes,queued_us,submit_us,start_us,end_us,queue_to_submit_us,submit_to_start_us,duration_us,host_start_us,host_end_us,host_duration_us
0,device,profile/start,0,0,0.000,426.594,3729.251,3729.928,426.594,3302.657,0.677,0.000,0.000,0.000
1,host,compile/blas1,0,0,12000.000,12000.000,12000.000,52000.000,0.000,0.000,40000.000,12000.000,52000.000,40000.000
2,device,write/ofArray,0,262144,20021.845,20023.001,20258.875,20392.677,1.156,235.874,133.802,0.000,0.000,0.000
3,device,kernel/scal,65536,0,40082.074,40091.334,40257.896,40263.312,9.260,166.562,5.416,0.000,0.000,0.000
4,host,compile/mapUnsafe,0,0,41000.000,41000.000,41000.000,47000.000,0.000,0.000,6000.000,41000.000,47000.000,6000.000
5,device,kernel/mapUnsafe,65536,0,48066.876,48071.855,48256.699,48279.251,4.979,184.844,22.552,0.000,0.000,0.000
6,device,read/toArray,0,262144,51595.047,51595.396,51656.487,51852.844,0.349,61.091,196.357,0.000,0.000,0.000`;

function parseCsvLine(line) {
  const parts = [];
  let cur = "";
  let quoted = false;
  for (const ch of line) {
    if (ch === '"') {
      quoted = !quoted;
    } else if (ch === "," && !quoted) {
      parts.push(cur);
      cur = "";
    } else {
      cur += ch;
    }
  }
  parts.push(cur);
  return parts;
}

function parseProfile(text) {
  const lines = text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  const headerIndex = lines.findIndex((line) => line.startsWith("idx,label,") || line.startsWith("idx,kind,label,"));
  if (headerIndex < 0) {
    throw new Error("Could not find profiler CSV header starting with 'idx,label,'.");
  }

  const header = parseCsvLine(lines[headerIndex]);
  const required = ["idx", "label", "work_items", "bytes", "queued_us", "submit_us", "start_us", "end_us", "queue_to_submit_us", "submit_to_start_us", "duration_us"];
  const missing = required.filter((column) => !header.includes(column));
  if (missing.length) {
    throw new Error(`Missing expected columns: ${missing.join(", ")}`);
  }

  return lines.slice(headerIndex + 1).map((line) => {
    const values = parseCsvLine(line);
    const row = {};
    header.forEach((column, index) => {
      row[column] = numericColumns.has(column) ? Number(values[index]) : values[index];
    });
    if (!row.kind) row.kind = "device";
    for (const column of columns) {
      if (numericColumns.has(column) && !Number.isFinite(row[column])) row[column] = 0;
      if (!numericColumns.has(column) && row[column] == null) row[column] = "";
    }
    return row;
  }).filter((row) => Number.isFinite(row.idx));
}

function filteredRows() {
  const kind = kindFilter.value;
  const labelNeedle = labelFilter.value.trim().toLowerCase();
  return rows.filter((row) => {
    if (kind !== "all" && row.kind !== kind) return false;
    if (labelNeedle && !row.label.toLowerCase().includes(labelNeedle)) return false;
    return true;
  });
}

function formatNumber(value, digits = 3) {
  if (!Number.isFinite(value)) return "";
  return value.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function renderStats() {
  const shown = filteredRows();
  if (!shown.length) {
    stats.innerHTML = "";
    return;
  }
  const totalSpan = Math.max(...shown.map((r) => r.end_us)) - Math.min(...shown.map((r) => r.queued_us));
  const totalRun = shown.reduce((acc, row) => acc + row.duration_us, 0);
  const maxRun = shown.reduce((best, row) => row.duration_us > best.duration_us ? row : best, shown[0]);
  stats.innerHTML = `
    <div class="stat"><b>${shown.length}</b><span>visible events</span></div>
    <div class="stat"><b>${formatNumber(totalSpan)}</b><span>timeline us</span></div>
    <div class="stat"><b>${formatNumber(totalRun)}</b><span>total run us</span></div>
    <div class="stat"><b>${formatNumber(maxRun.duration_us)}</b><span>slowest: ${maxRun.label}</span></div>
  `;
}

function renderTimeline() {
  const shown = filteredRows().sort((a, b) => a.idx - b.idx);
  if (!shown.length) {
    timeline.className = "timeline empty";
    timeline.textContent = "No profile loaded.";
    zoomInfo.textContent = "Drag across the timeline to zoom into a subinterval.";
    return;
  }

  timeline.className = "timeline";
  const transformedRows = compressGaps.checked ? compressIdleGaps(shown) : shown.map((row) => ({ ...row }));
  const fullMin = Math.min(...transformedRows.map((r) => r.queued_us));
  const fullMax = Math.max(...transformedRows.map((r) => r.end_us));
  if (zoomDomain && (zoomDomain[1] <= fullMin || zoomDomain[0] >= fullMax)) zoomDomain = null;

  const minQueued = zoomDomain ? Math.max(fullMin, zoomDomain[0]) : fullMin;
  const maxEnd = zoomDomain ? Math.min(fullMax, zoomDomain[1]) : fullMax;
  const span = Math.max(1, maxEnd - minQueued);
  const timelineRows = transformedRows.filter((row) => row.end_us >= minQueued && row.queued_us <= maxEnd);
  currentTimelineScale = { min: minQueued, max: maxEnd, span };
  zoomInfo.textContent = zoomDomain
    ? `Zoom: ${formatNumber(minQueued)} us to ${formatNumber(maxEnd)} us. Drag again to zoom further, or reset.`
    : "Drag across the timeline to zoom into a subinterval.";

  timeline.innerHTML = `
    <div class="timeline-row timeline-axis-row">
      <div class="event-label">time axis</div>
      <div class="timeline-axis">${renderAxisTicks(minQueued, maxEnd)}</div>
      <div class="duration">${formatNumber(span)} us</div>
    </div>
    <div class="timeline-chart">
      <div class="brush"></div>
      ${timelineRows.map((row) => {
    const q = ((row.queued_us - minQueued) / span) * 100;
    const submit = ((row.submit_us - minQueued) / span) * 100;
    const start = ((row.start_us - minQueued) / span) * 100;
    const end = ((row.end_us - minQueued) / span) * 100;
    const queueW = Math.max(0.05, submit - q);
    const waitW = Math.max(0.05, start - submit);
    const runW = Math.max(0.05, end - start);
    const title = `${row.label}\nqueued ${row.queued_us}us\nsubmit ${row.submit_us}us\nstart ${row.start_us}us\nend ${row.end_us}us\nduration ${row.duration_us}us`;
    const transferClass = isTransfer(row) ? "transfer" : "";
    return `
      <div class="timeline-row" title="${escapeHtml(title)}">
        <div class="event-label">${row.idx}. [${escapeHtml(row.kind)}] ${escapeHtml(row.label)}</div>
        <div class="track">
          ${row.kind === "host"
            ? `<span class="seg host" style="left:${q}%;width:${Math.max(0.05, end - q)}%"></span>`
            : `<span class="seg queue" style="left:${q}%;width:${queueW}%"></span><span class="seg wait" style="left:${submit}%;width:${waitW}%"></span><span class="seg run ${transferClass}" style="left:${start}%;width:${runW}%"></span>`}
        </div>
        <div class="duration">${formatNumber(row.duration_us)} us</div>
      </div>
    `;
  }).join("")}
    </div>
  `;
}

function renderAxisTicks(min, max) {
  const span = Math.max(1, max - min);
  const ticks = 6;
  const out = [];
  for (let i = 0; i <= ticks; i++) {
    const pct = (i / ticks) * 100;
    const value = min + span * (i / ticks);
    out.push(`<span class="tick" style="left:${pct}%">${formatNumber(value)} us</span>`);
  }
  return out.join("");
}

function isTransfer(row) {
  return row.label.startsWith("write/") || row.label.startsWith("read/") || row.bytes > 0;
}

function compressIdleGaps(inputRows) {
  const maxGap = 1500;
  let removed = 0;
  let previousEnd = null;
  return inputRows.map((row) => {
    if (previousEnd != null) {
      const gap = row.queued_us - previousEnd;
      if (gap > maxGap) removed += gap - maxGap;
    }
    previousEnd = Math.max(previousEnd ?? row.end_us, row.end_us);
    return {
      ...row,
      queued_us: row.queued_us - removed,
      submit_us: row.submit_us - removed,
      start_us: row.start_us - removed,
      end_us: row.end_us - removed,
    };
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function sortedRows() {
  const { column, direction } = sortState;
  const sign = direction === "asc" ? 1 : -1;
  return [...filteredRows()].sort((a, b) => {
    if (numericColumns.has(column)) {
      return sign * ((a[column] ?? 0) - (b[column] ?? 0));
    }
    return sign * String(a[column] ?? "").localeCompare(String(b[column] ?? ""));
  });
}

function renderTable() {
  const thead = table.querySelector("thead");
  const tbody = table.querySelector("tbody");
  thead.innerHTML = `<tr>${columns.map((column) => {
    const arrow = sortState.column === column ? (sortState.direction === "asc" ? " ▲" : " ▼") : "";
    return `<th data-column="${column}">${column}${arrow}</th>`;
  }).join("")}</tr>`;

  tbody.innerHTML = sortedRows().map((row) => `
    <tr>
      ${columns.map((column) => {
        const value = numericColumns.has(column) ? formatNumber(row[column]) : escapeHtml(row[column]);
        const cls = numericColumns.has(column) ? "number" : "";
        return `<td class="${cls}">${value}</td>`;
      }).join("")}
    </tr>
  `).join("");
}

function renderAll() {
  renderStats();
  renderTimeline();
  renderTable();
}

function parseAndRender() {
  try {
    rows = parseProfile(rawInput.value);
    sortState = { column: "idx", direction: "asc" };
    zoomDomain = null;
    renderAll();
  } catch (err) {
    timeline.className = "timeline empty";
    timeline.innerHTML = `<span class="error">${escapeHtml(err.message)}</span>`;
    rows = [];
    renderStats();
    renderTable();
  }
}

parseButton.addEventListener("click", parseAndRender);

loadSampleButton.addEventListener("click", () => {
  rawInput.value = sample;
  parseAndRender();
});

clearButton.addEventListener("click", () => {
  rawInput.value = "";
  rows = [];
  zoomDomain = null;
  renderAll();
});

fileInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  rawInput.value = await file.text();
  parseAndRender();
});

kindFilter.addEventListener("change", renderAll);
labelFilter.addEventListener("input", renderAll);
compressGaps.addEventListener("change", () => {
  zoomDomain = null;
  renderTimeline();
});
resetZoom.addEventListener("click", () => {
  zoomDomain = null;
  renderTimeline();
});

timeline.addEventListener("mousedown", (event) => {
  const chart = event.target.closest(".timeline-chart");
  if (!chart || !currentTimelineScale) return;
  const track = chart.querySelector(".track");
  if (!track) return;
  brushChart = chart;
  brushTrack = track.getBoundingClientRect();
  brushStartPx = clamp(event.clientX, brushTrack.left, brushTrack.right);
  updateBrush(event.clientX);
  event.preventDefault();
});

timeline.addEventListener("dblclick", (event) => {
  if (!event.target.closest(".timeline-chart") && !event.target.closest(".timeline-axis")) return;
  zoomDomain = null;
  renderTimeline();
});

document.addEventListener("mousemove", (event) => {
  if (!brushChart) return;
  updateBrush(event.clientX);
});

document.addEventListener("mouseup", (event) => {
  if (!brushChart || !currentTimelineScale || !brushTrack) return;
  const endPx = clamp(event.clientX, brushTrack.left, brushTrack.right);
  const startPct = (Math.min(brushStartPx, endPx) - brushTrack.left) / brushTrack.width;
  const endPct = (Math.max(brushStartPx, endPx) - brushTrack.left) / brushTrack.width;
  const brush = brushChart.querySelector(".brush");
  if (brush) brush.style.display = "none";

  if (endPct - startPct > 0.01) {
    const { min, span } = currentTimelineScale;
    zoomDomain = [min + startPct * span, min + endPct * span];
    renderTimeline();
  }
  brushStartPx = null;
  brushChart = null;
  brushTrack = null;
});

function updateBrush(clientX) {
  if (!brushChart || brushStartPx == null || !brushTrack) return;
  const brush = brushChart.querySelector(".brush");
  if (!brush) return;
  const endPx = clamp(clientX, brushTrack.left, brushTrack.right);
  const chartRect = brushChart.getBoundingClientRect();
  const left = Math.min(brushStartPx, endPx) - chartRect.left;
  const width = Math.abs(endPx - brushStartPx);
  brush.style.display = "block";
  brush.style.left = `${left}px`;
  brush.style.width = `${width}px`;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

table.addEventListener("click", (event) => {
  const th = event.target.closest("th");
  if (!th) return;
  const column = th.dataset.column;
  if (sortState.column === column) {
    sortState.direction = sortState.direction === "asc" ? "desc" : "asc";
  } else {
    sortState = { column, direction: numericColumns.has(column) ? "desc" : "asc" };
  }
  renderTable();
});

renderAll();
