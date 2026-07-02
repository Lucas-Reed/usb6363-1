"""双峰锁定最小波形查看器。

运行方式：
    1. 先启动底层 USB-6363 API：
       python usb6363_server.py

    2. 再启动本查看器：
       python two_peak_viewer.py

    3. 浏览器打开：
       http://127.0.0.1:8766

重要边界：
    本文件不直接 import nidaqmx。
    它只通过 usb6363_client.Usb6363Client 调用底层 API。

当前目标：
    - 采集并显示任意 AI 通道的一帧同步波形。
    - 在第一路波形上手动选择 P1/P2。
    - 根据当前参数测量峰高或峰面积。
    - 保存当前帧样本，供后面讨论自动寻峰算法。
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from two_peak.config import TwoPeakSettings
from two_peak.signal import locate_and_measure_two_peaks
from usb6363_client import DEFAULT_BASE_URL, Usb6363Client


# 查看器默认端口。底层 usb6363_server.py 默认使用 8765，所以这里用 8766。
DEFAULT_VIEWER_HOST = "127.0.0.1"
DEFAULT_VIEWER_PORT = 8766

# 样本默认保存目录。一个样本包含 .npy 原始波形和 .json 元数据。
DEFAULT_SAMPLE_DIR = Path("data") / "two_peak_samples"


class ViewerState:
    """查看器运行状态。

    这里保存“最近采到的一帧”，这样 WebUI 点击保存时，不需要重新采集。
    这只是查看器自己的内存状态，不是采集卡状态。
    """

    def __init__(self, api_base_url: str, sample_dir: Path) -> None:
        self.daq = Usb6363Client(base_url=api_base_url)
        self.sample_dir = sample_dir
        self.settings = TwoPeakSettings.defaults()
        self.latest_frame: dict[str, Any] | None = None
        self.latest_measurement: dict[str, Any] | None = None


def make_handler(state: ViewerState):
    """创建 HTTP 请求处理类。

    Python 标准库 HTTPServer 需要传入“类”，所以用函数把 state 包进去。
    """

    class TwoPeakViewerHandler(BaseHTTPRequestHandler):
        server_version = "TwoPeakViewer/0.1"

        def do_GET(self) -> None:
            """处理页面和只读 API。"""

            try:
                if self.path == "/" or self.path.startswith("/?"):
                    self._send_html(HTML_PAGE)
                elif self.path == "/api/defaults":
                    self._send_json(state.settings.to_web_parameters())
                elif self.path == "/api/latest":
                    self._send_json(
                        {
                            "has_frame": state.latest_frame is not None,
                            "frame": _frame_summary(state.latest_frame),
                            "measurement": state.latest_measurement,
                        }
                    )
                else:
                    self._send_error(HTTPStatus.NOT_FOUND, "Unknown route")
            except Exception as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, str(exc))

        def do_POST(self) -> None:
            """处理会改变查看器状态的 API。"""

            try:
                if self.path == "/api/capture":
                    body = self._read_json()
                    frame = _capture_frame(state, body)
                    state.latest_frame = frame
                    state.latest_measurement = None
                    self._send_json(frame)
                elif self.path == "/api/measure":
                    body = self._read_json()
                    measurement = _measure_latest_frame(state, body)
                    state.latest_measurement = measurement
                    self._send_json(measurement)
                elif self.path == "/api/save":
                    body = self._read_json()
                    saved = _save_latest_frame(state, body)
                    self._send_json(saved)
                else:
                    self._send_error(HTTPStatus.NOT_FOUND, "Unknown route")
            except Exception as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, str(exc))

        def log_message(self, format: str, *args: Any) -> None:
            """打印请求日志，便于调试浏览器发了什么请求。"""

            print(f"{self.address_string()} - {format % args}")

        def _read_json(self) -> dict[str, Any]:
            """读取 POST 请求里的 JSON。"""

            length = int(self.headers.get("Content-Length", "0"))
            if length == 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8")
            return json.loads(raw)

        def _send_html(self, html: str) -> None:
            """返回 HTML 页面。"""

            payload = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, data: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            """返回 JSON。"""

            payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_error(self, status: HTTPStatus, message: str) -> None:
            """用统一格式返回错误。"""

            self._send_json({"ok": False, "error": message}, status=status)

    return TwoPeakViewerHandler


def _parse_channels(value: Any) -> list[str]:
    """把 WebUI 传来的通道参数整理成列表。

    支持两种写法：
        ["ai0", "ai1"]
        "ai0, ai1"
    """

    if isinstance(value, list):
        channels = [str(item).strip() for item in value]
    elif isinstance(value, str):
        channels = [item.strip() for item in value.split(",")]
    else:
        raise ValueError("channels must be a list or comma-separated string")

    channels = [channel for channel in channels if channel]
    if not channels:
        raise ValueError("channels must not be empty")
    return channels


def _capture_frame(state: ViewerState, body: dict[str, Any]) -> dict[str, Any]:
    """调用底层 API 同步采集一帧。"""

    channels = _parse_channels(body.get("channels", ["ai0", "ai1"]))
    samples = int(body.get("samples", 5000))
    rate = float(body.get("rate", 50_000.0))
    terminal_config = str(body.get("terminal_config", "DIFF"))
    min_val = float(body.get("min_val", -5.0))
    max_val = float(body.get("max_val", 5.0))
    timeout = float(body.get("timeout", 10.0))

    frame = state.daq.capture_ai_frame(
        channels=channels,
        samples=samples,
        rate=rate,
        terminal_config=terminal_config,
        min_val=min_val,
        max_val=max_val,
        timeout=timeout,
    )
    frame["captured_by"] = "two_peak_viewer"
    frame["viewer_received_at"] = time.time()
    return frame


def _measure_latest_frame(state: ViewerState, body: dict[str, Any]) -> dict[str, Any]:
    """测量最近一帧里的 P1/P2。

    当前版本默认在第一路 AI 波形上测峰，因为旧双峰程序里 AI0 是 FP 透射信号。
    后面 WebUI 可以再加“选择哪一路作为寻峰信号”。
    """

    frame = state.latest_frame
    if frame is None:
        raise RuntimeError("No frame has been captured yet")

    values = np.asarray(frame["values"], dtype=float)
    if values.ndim != 2 or values.shape[0] < 1:
        raise RuntimeError("latest frame has no AI waveform")

    peak_indices = body.get("peak_indices")
    if not isinstance(peak_indices, list) or len(peak_indices) != 2:
        raise ValueError("peak_indices must be a two-item list")

    smooth_window = int(body.get("smooth_window", 25))
    search_window_half = int(body.get("search_window_half", 20))
    measure_half = int(body.get("measure_half", 5))
    peak_mode = str(body.get("peak_mode", "height"))
    analysis_channel_index = int(body.get("analysis_channel_index", 0))
    if analysis_channel_index < 0 or analysis_channel_index >= values.shape[0]:
        raise ValueError("analysis_channel_index is out of range")

    _, measurements = locate_and_measure_two_peaks(
        ai0=values[analysis_channel_index],
        peak_indices=[int(peak_indices[0]), int(peak_indices[1])],
        smooth_window=smooth_window,
        search_window_half=search_window_half,
        measure_half=measure_half,
        mode=peak_mode,
    )

    return {
        "analysis_channel_index": analysis_channel_index,
        "analysis_channel": frame["channels"][analysis_channel_index],
        "peak_indices_input": [int(peak_indices[0]), int(peak_indices[1])],
        "smooth_window": smooth_window,
        "search_window_half": search_window_half,
        "measure_half": measure_half,
        "peak_mode": peak_mode,
        "measurements": [asdict(item) for item in measurements],
    }


def _save_latest_frame(state: ViewerState, body: dict[str, Any]) -> dict[str, Any]:
    """保存最近一帧原始数据和元数据。"""

    frame = state.latest_frame
    if frame is None:
        raise RuntimeError("No frame has been captured yet")

    sample_dir = state.sample_dir
    sample_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = str(body.get("label", "")).strip()
    safe_label = "".join(ch for ch in label if ch.isalnum() or ch in ("-", "_"))
    stem = f"{timestamp}_{safe_label}" if safe_label else timestamp

    npy_path = sample_dir / f"{stem}.npy"
    json_path = sample_dir / f"{stem}.json"

    values = np.asarray(frame["values"], dtype=np.float64)
    np.save(npy_path, values)

    metadata = {
        "saved_at": time.time(),
        "npy_file": str(npy_path.resolve()),
        "metadata_file": str(json_path.resolve()),
        "shape": list(values.shape),
        "frame": _frame_summary(frame),
        "measurement": state.latest_measurement,
        "web_parameters": body.get("parameters", {}),
        "factory_defaults": state.settings.to_web_parameters(),
        "format": "numpy .npy, shape=(channel_count, samples_per_channel)",
    }
    json_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return metadata


def _frame_summary(frame: dict[str, Any] | None) -> dict[str, Any] | None:
    """返回不包含巨大 values 的帧摘要。"""

    if frame is None:
        return None
    return {
        "device": frame.get("device"),
        "channels": frame.get("channels"),
        "channel_count": frame.get("channel_count"),
        "samples_per_channel": frame.get("samples_per_channel"),
        "rate_per_channel": frame.get("rate_per_channel"),
        "aggregate_rate": frame.get("aggregate_rate"),
        "terminal_config": frame.get("terminal_config"),
        "min_val": frame.get("min_val"),
        "max_val": frame.get("max_val"),
        "duration_seconds": frame.get("duration_seconds"),
        "started_at": frame.get("started_at"),
        "finished_at": frame.get("finished_at"),
    }


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>双峰波形查看器</title>
<style>
  :root {
    --bg: #f4f6f8;
    --panel: #ffffff;
    --line: #d7dde5;
    --text: #17202a;
    --muted: #697586;
    --blue: #1f6feb;
    --green: #16833a;
    --red: #c62828;
    --amber: #9a6700;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: "Segoe UI", Arial, sans-serif;
    height: 100vh;
    display: grid;
    grid-template-columns: 360px 1fr;
    overflow: hidden;
  }
  aside {
    background: var(--panel);
    border-right: 1px solid var(--line);
    padding: 12px;
    overflow: auto;
  }
  main {
    min-width: 0;
    display: grid;
    grid-template-rows: auto 1fr auto;
    gap: 10px;
    padding: 12px;
  }
  h1 {
    font-size: 18px;
    margin: 0 0 12px;
  }
  h2 {
    font-size: 14px;
    margin: 18px 0 8px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--line);
  }
  label {
    display: block;
    font-size: 12px;
    color: var(--muted);
    margin: 8px 0 4px;
  }
  input, select, button {
    width: 100%;
    font: inherit;
    font-size: 13px;
  }
  input, select {
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 7px 8px;
    background: #fff;
    color: var(--text);
  }
  button {
    border: 0;
    border-radius: 6px;
    padding: 8px 10px;
    background: var(--blue);
    color: white;
    font-weight: 650;
    cursor: pointer;
  }
  button.secondary { background: #59636f; }
  button.green { background: var(--green); }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  .grid2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }
  .actions {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-top: 12px;
  }
  .status {
    border: 1px solid var(--line);
    background: var(--panel);
    border-radius: 8px;
    padding: 9px 10px;
    font-size: 13px;
    color: var(--muted);
  }
  .status.error { color: var(--red); }
  .status.ok { color: var(--green); }
  .plots {
    display: grid;
    gap: 10px;
    min-height: 0;
    overflow: auto;
  }
  .plot {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 8px;
    min-height: 230px;
    position: relative;
    overflow: hidden;
  }
  .plot-title {
    position: absolute;
    left: 10px;
    top: 8px;
    background: rgba(255,255,255,0.88);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 3px 7px;
    font-size: 12px;
    color: var(--muted);
    z-index: 2;
  }
  canvas {
    width: 100%;
    height: 100%;
    display: block;
    cursor: crosshair;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }
  th, td {
    text-align: right;
    border-bottom: 1px solid var(--line);
    padding: 6px 4px;
  }
  th:first-child, td:first-child { text-align: left; }
  .footer {
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: center;
    gap: 8px;
  }
  .mono {
    font-family: Consolas, "SFMono-Regular", monospace;
  }
</style>
</head>
<body>
<aside>
  <h1>双峰波形查看器</h1>

  <h2>采集</h2>
  <label>AI 通道，逗号分隔</label>
  <input id="channels" value="ai0, ai1">
  <div class="grid2">
    <div>
      <label>采样率 / Hz</label>
      <input id="rate" type="number" value="50000" step="1000">
    </div>
    <div>
      <label>每通道点数</label>
      <input id="samples" type="number" value="5000" step="100">
    </div>
  </div>
  <div class="grid2">
    <div>
      <label>接线方式</label>
      <select id="terminal_config">
        <option value="DIFF">DIFF</option>
        <option value="RSE">RSE</option>
        <option value="NRSE">NRSE</option>
      </select>
    </div>
    <div>
      <label>超时 / s</label>
      <input id="timeout" type="number" value="10" step="1">
    </div>
  </div>
  <div class="grid2">
    <div>
      <label>AI 最小值 / V</label>
      <input id="min_val" type="number" value="-5" step="0.5">
    </div>
    <div>
      <label>AI 最大值 / V</label>
      <input id="max_val" type="number" value="5" step="0.5">
    </div>
  </div>
  <div class="actions">
    <button id="captureBtn" onclick="captureFrame()">采集一帧</button>
    <button class="green" id="saveBtn" onclick="saveFrame()" disabled>保存样本</button>
  </div>

  <h2>手动选峰</h2>
  <label>点击波形时设置</label>
  <select id="pick_target">
    <option value="0">P1</option>
    <option value="1">P2</option>
  </select>
  <div class="grid2">
    <div>
      <label>P1 索引</label>
      <input id="peak0" type="number" value="642" step="1">
    </div>
    <div>
      <label>P2 索引</label>
      <input id="peak1" type="number" value="754" step="1">
    </div>
  </div>

  <h2>测峰参数</h2>
  <div class="grid2">
    <div>
      <label>分析通道序号</label>
      <input id="analysis_channel_index" type="number" value="0" step="1">
    </div>
    <div>
      <label>测峰模式</label>
      <select id="peak_mode">
        <option value="height">height</option>
        <option value="area">area</option>
      </select>
    </div>
  </div>
  <div class="grid2">
    <div>
      <label>平滑窗口</label>
      <input id="smooth_window" type="number" value="25" step="2">
    </div>
    <div>
      <label>搜索半宽</label>
      <input id="search_window_half" type="number" value="20" step="1">
    </div>
  </div>
  <label>测量半宽</label>
  <input id="measure_half" type="number" value="5" step="1">
  <div class="actions">
    <button class="secondary" onclick="measurePeaks()">重新测峰</button>
    <button class="secondary" onclick="clearPeaks()">清空标记</button>
  </div>

  <h2>结果</h2>
  <table>
    <thead><tr><th>峰</th><th>索引</th><th>测量值</th><th>窗口</th></tr></thead>
    <tbody id="peakRows">
      <tr><td>P1</td><td>--</td><td>--</td><td>--</td></tr>
      <tr><td>P2</td><td>--</td><td>--</td><td>--</td></tr>
    </tbody>
  </table>
</aside>

<main>
  <div id="status" class="status">查看器已打开。请先确认底层 API 服务正在运行。</div>
  <div id="plots" class="plots"></div>
  <div class="footer">
    <div id="frameInfo" class="status">尚未采集。</div>
    <div class="mono" id="cursorInfo"></div>
  </div>
</main>

<script>
let latestFrame = null;
let latestMeasurement = null;
let plotCanvases = [];

function getCaptureParams() {
  return {
    channels: document.getElementById('channels').value,
    rate: Number(document.getElementById('rate').value),
    samples: Number(document.getElementById('samples').value),
    terminal_config: document.getElementById('terminal_config').value,
    min_val: Number(document.getElementById('min_val').value),
    max_val: Number(document.getElementById('max_val').value),
    timeout: Number(document.getElementById('timeout').value),
  };
}

function getMeasureParams() {
  return {
    peak_indices: [
      Number(document.getElementById('peak0').value),
      Number(document.getElementById('peak1').value),
    ],
    analysis_channel_index: Number(document.getElementById('analysis_channel_index').value),
    smooth_window: Number(document.getElementById('smooth_window').value),
    search_window_half: Number(document.getElementById('search_window_half').value),
    measure_half: Number(document.getElementById('measure_half').value),
    peak_mode: document.getElementById('peak_mode').value,
  };
}

function setStatus(text, kind) {
  const el = document.getElementById('status');
  el.textContent = text;
  el.className = 'status' + (kind ? ' ' + kind : '');
}

async function postJson(path, body) {
  const response = await fetch(path, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok || data.ok === false) {
    throw new Error(data.error || '请求失败');
  }
  return data;
}

async function captureFrame() {
  const btn = document.getElementById('captureBtn');
  btn.disabled = true;
  setStatus('正在采集一帧...', '');
  try {
    latestFrame = await postJson('/api/capture', getCaptureParams());
    latestMeasurement = null;
    document.getElementById('saveBtn').disabled = false;
    renderPlots();
    updateFrameInfo();
    setStatus('采集完成。', 'ok');
    await measurePeaks();
  } catch (err) {
    setStatus(String(err.message || err), 'error');
  } finally {
    btn.disabled = false;
  }
}

async function measurePeaks() {
  if (!latestFrame) {
    setStatus('还没有采集波形。', 'error');
    return;
  }
  try {
    latestMeasurement = await postJson('/api/measure', getMeasureParams());
    updatePeakTable();
    renderPlots();
    setStatus('测峰完成。', 'ok');
  } catch (err) {
    setStatus(String(err.message || err), 'error');
  }
}

async function saveFrame() {
  if (!latestFrame) {
    setStatus('还没有采集波形。', 'error');
    return;
  }
  try {
    const saved = await postJson('/api/save', {
      parameters: {...getCaptureParams(), ...getMeasureParams()},
    });
    setStatus('样本已保存：' + saved.npy_file, 'ok');
  } catch (err) {
    setStatus(String(err.message || err), 'error');
  }
}

function clearPeaks() {
  latestMeasurement = null;
  updatePeakTable();
  renderPlots();
}

function renderPlots() {
  const container = document.getElementById('plots');
  container.innerHTML = '';
  plotCanvases = [];
  if (!latestFrame || !latestFrame.values) return;

  latestFrame.values.forEach((data, index) => {
    const box = document.createElement('div');
    box.className = 'plot';
    const title = document.createElement('div');
    title.className = 'plot-title';
    title.textContent = `${latestFrame.channels[index]}  ${data.length} 点`;
    const canvas = document.createElement('canvas');
    canvas.addEventListener('click', event => handlePlotClick(event, index));
    canvas.addEventListener('mousemove', event => updateCursor(event, index));
    box.appendChild(title);
    box.appendChild(canvas);
    container.appendChild(box);
    plotCanvases.push({canvas, data, index});
  });

  resizeAndDraw();
}

function resizeAndDraw() {
  for (const item of plotCanvases) {
    const rect = item.canvas.parentElement.getBoundingClientRect();
    item.canvas.width = Math.max(300, Math.floor(rect.width * devicePixelRatio));
    item.canvas.height = Math.max(180, Math.floor(rect.height * devicePixelRatio));
    drawWaveform(item.canvas, item.data, item.index);
  }
}

function drawWaveform(canvas, data, channelIndex) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, w, h);

  if (!data || data.length < 2) return;
  let min = data[0];
  let max = data[0];
  for (const value of data) {
    if (value < min) min = value;
    if (value > max) max = value;
  }
  const span = Math.max(1e-12, max - min);
  const padX = 44 * devicePixelRatio;
  const padY = 28 * devicePixelRatio;
  const x0 = padX;
  const y0 = padY;
  const plotW = w - 2 * padX;
  const plotH = h - 2 * padY;

  ctx.strokeStyle = '#d7dde5';
  ctx.lineWidth = 1 * devicePixelRatio;
  ctx.strokeRect(x0, y0, plotW, plotH);

  ctx.beginPath();
  data.forEach((value, i) => {
    const x = x0 + (i / (data.length - 1)) * plotW;
    const y = y0 + (1 - (value - min) / span) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = channelIndex === 0 ? '#1f6feb' : '#16833a';
  ctx.lineWidth = 1.4 * devicePixelRatio;
  ctx.stroke();

  drawMarkers(ctx, data, channelIndex, x0, y0, plotW, plotH, min, span);

  ctx.fillStyle = '#697586';
  ctx.font = `${11 * devicePixelRatio}px Consolas`;
  ctx.fillText(`min ${min.toFixed(5)} V`, x0, h - 8 * devicePixelRatio);
  ctx.fillText(`max ${max.toFixed(5)} V`, x0 + 150 * devicePixelRatio, h - 8 * devicePixelRatio);
}

function drawMarkers(ctx, data, channelIndex, x0, y0, plotW, plotH, min, span) {
  const analysisIndex = Number(document.getElementById('analysis_channel_index').value);
  if (channelIndex !== analysisIndex) return;

  const peaks = [
    Number(document.getElementById('peak0').value),
    Number(document.getElementById('peak1').value),
  ];
  peaks.forEach((idx, i) => {
    if (!Number.isFinite(idx) || idx < 0 || idx >= data.length) return;
    const x = x0 + (idx / (data.length - 1)) * plotW;
    ctx.strokeStyle = i === 0 ? '#c62828' : '#9a6700';
    ctx.lineWidth = 1.3 * devicePixelRatio;
    ctx.beginPath();
    ctx.moveTo(x, y0);
    ctx.lineTo(x, y0 + plotH);
    ctx.stroke();
    ctx.fillStyle = i === 0 ? '#c62828' : '#9a6700';
    ctx.font = `${12 * devicePixelRatio}px Segoe UI`;
    ctx.fillText(i === 0 ? 'P1' : 'P2', x + 4 * devicePixelRatio, y0 + 16 * devicePixelRatio);
  });

  if (latestMeasurement && latestMeasurement.measurements) {
    latestMeasurement.measurements.forEach((m, i) => {
      const idx = m.index;
      if (idx < 0 || idx >= data.length) return;
      const x = x0 + (idx / (data.length - 1)) * plotW;
      const y = y0 + (1 - (data[idx] - min) / span) * plotH;
      ctx.fillStyle = i === 0 ? '#c62828' : '#9a6700';
      ctx.beginPath();
      ctx.arc(x, y, 4 * devicePixelRatio, 0, Math.PI * 2);
      ctx.fill();
    });
  }
}

function handlePlotClick(event, channelIndex) {
  const analysisIndex = Number(document.getElementById('analysis_channel_index').value);
  if (channelIndex !== analysisIndex || !latestFrame) return;
  const item = plotCanvases[channelIndex];
  const idx = eventToSampleIndex(event, item.canvas, item.data.length);
  const target = document.getElementById('pick_target').value;
  document.getElementById(target === '0' ? 'peak0' : 'peak1').value = idx;
  document.getElementById('pick_target').value = target === '0' ? '1' : '0';
  measurePeaks();
}

function updateCursor(event, channelIndex) {
  const item = plotCanvases[channelIndex];
  const idx = eventToSampleIndex(event, item.canvas, item.data.length);
  const value = item.data[idx];
  document.getElementById('cursorInfo').textContent =
    `${latestFrame.channels[channelIndex]}  index=${idx}  value=${value.toFixed(6)} V`;
}

function eventToSampleIndex(event, canvas, length) {
  const rect = canvas.getBoundingClientRect();
  const x = Math.min(Math.max(event.clientX - rect.left, 0), rect.width);
  return Math.round((x / Math.max(1, rect.width)) * (length - 1));
}

function updatePeakTable() {
  const rows = document.getElementById('peakRows');
  if (!latestMeasurement || !latestMeasurement.measurements) {
    rows.innerHTML = '<tr><td>P1</td><td>--</td><td>--</td><td>--</td></tr><tr><td>P2</td><td>--</td><td>--</td><td>--</td></tr>';
    return;
  }
  rows.innerHTML = latestMeasurement.measurements.map((m, i) => `
    <tr>
      <td>${i === 0 ? 'P1' : 'P2'}</td>
      <td>${m.index}</td>
      <td>${Number(m.value).toExponential(6)}</td>
      <td>${m.window_left}-${m.window_right}</td>
    </tr>
  `).join('');
}

function updateFrameInfo() {
  if (!latestFrame) return;
  document.getElementById('frameInfo').textContent =
    `channels=${latestFrame.channels.join(', ')}  rate=${latestFrame.rate_per_channel} Hz  samples=${latestFrame.samples_per_channel}`;
}

window.addEventListener('resize', resizeAndDraw);
updatePeakTable();
</script>
</body>
</html>
"""


def main() -> int:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="Run the two-peak waveform viewer.")
    parser.add_argument("--host", default=DEFAULT_VIEWER_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_VIEWER_PORT)
    parser.add_argument("--api-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--sample-dir", default=str(DEFAULT_SAMPLE_DIR))
    args = parser.parse_args()

    state = ViewerState(
        api_base_url=args.api_base_url,
        sample_dir=Path(args.sample_dir),
    )
    handler = make_handler(state)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    print(f"Two-peak waveform viewer running at http://{args.host}:{args.port}")
    print(f"Using USB-6363 API at {args.api_base_url}")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping viewer.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
