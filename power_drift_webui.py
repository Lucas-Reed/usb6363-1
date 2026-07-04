"""光电探测器功率慢漂 WebUI。

这个文件提供一个很小的本地网页，用来长期监测光电探测器输出。

运行方式：
    python power_drift_webui.py

然后打开：
    http://127.0.0.1:8767

设计边界：
- 本文件不 import nidaqmx。
- 本文件不直接控制 USB-6363。
- 真正读硬件仍然通过 Usb6363Client -> 8765 底层 API。
- 采集/统计逻辑复用 power_drift_monitor.py，避免命令行版和 WebUI 版各写一套。
"""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from power_drift_monitor import DEFAULT_OUTPUT_DIR
from power_drift_monitor import PowerDriftMonitor
from power_drift_monitor import PowerDriftPoint
from power_drift_monitor import PowerDriftSettings
from usb6363_client import DEFAULT_BASE_URL


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8767


class PowerDriftWebState:
    """WebUI 的运行状态。

    浏览器会不断请求 /api/status。这个对象负责保存：
    - 当前是否正在记录；
    - CSV 文件路径；
    - 最近一些点，用于前端画趋势图；
    - 后台线程中的错误信息。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._running = False
        self._error: str | None = None
        self._csv_path: Path | None = None
        self._metadata_path: Path | None = None
        self._settings: PowerDriftSettings | None = None
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._rows_written = 0
        self._latest_point: PowerDriftPoint | None = None
        self._recent_points: deque[PowerDriftPoint] = deque(maxlen=1000)

    def start(self, settings: PowerDriftSettings) -> dict[str, Any]:
        """启动后台功率慢漂记录线程。"""

        with self._lock:
            if self._running:
                raise RuntimeError("功率慢漂记录已经在运行")

            settings.output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = settings.output_dir / f"power_drift_web_{timestamp}.csv"
            metadata_path = settings.output_dir / f"power_drift_web_{timestamp}.json"

            stop_event = threading.Event()
            thread = threading.Thread(
                target=self._worker,
                args=(settings, csv_path, metadata_path, stop_event),
                daemon=True,
                name="power-drift-web-monitor",
            )

            self._thread = thread
            self._stop_event = stop_event
            self._running = True
            self._error = None
            self._csv_path = csv_path
            self._metadata_path = metadata_path
            self._settings = settings
            self._started_at = time.time()
            self._finished_at = None
            self._rows_written = 0
            self._latest_point = None
            self._recent_points.clear()
            thread.start()

        return self.status()

    def stop(self) -> dict[str, Any]:
        """停止后台记录。

        如果底层正在进行一次 read_ai，线程可能要等这次读取完成后才会停下。
        """

        with self._lock:
            stop_event = self._stop_event
            thread = self._thread
            if stop_event is not None:
                stop_event.set()

        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)

        with self._lock:
            if self._thread is thread and (thread is None or not thread.is_alive()):
                self._running = False
                self._thread = None
                self._stop_event = None
                self._finished_at = time.time()

        return self.status()

    def status(self) -> dict[str, Any]:
        """返回前端需要显示的状态。"""

        with self._lock:
            return {
                "running": self._running,
                "error": self._error,
                "csv_file": str(self._csv_path.resolve()) if self._csv_path else None,
                "metadata_file": str(self._metadata_path.resolve()) if self._metadata_path else None,
                "started_at": self._started_at,
                "finished_at": self._finished_at,
                "rows_written": self._rows_written,
                "settings": _settings_for_json(self._settings) if self._settings else None,
                "latest_point": asdict(self._latest_point) if self._latest_point else None,
                "recent_points": [asdict(point) for point in self._recent_points],
            }

    def latest_csv_path(self) -> Path:
        """返回当前或最近一次记录的 CSV 路径，用于下载。"""

        with self._lock:
            if self._csv_path is None:
                raise FileNotFoundError("还没有 CSV 文件")
            return self._csv_path

    def _worker(
        self,
        settings: PowerDriftSettings,
        csv_path: Path,
        metadata_path: Path,
        stop_event: threading.Event,
    ) -> None:
        """后台记录线程主体。"""

        monitor = PowerDriftMonitor(settings)
        fieldnames = list(PowerDriftPoint.__dataclass_fields__)
        start_time = time.time()
        next_start_time = start_time
        row_index = 0

        try:
            # 启动前检查硬件状态，避免和双峰连续采集等任务互相抢 AI。
            monitor.check_hardware_idle()

            metadata_path.write_text(
                json.dumps(
                    {
                        "started_at": datetime.fromtimestamp(start_time).isoformat(timespec="seconds"),
                        "csv_file": str(csv_path.resolve()),
                        "settings": _settings_for_json(settings),
                        "note": "power_estimate = (mean_v - zero_voltage) * power_per_volt",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            with csv_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()

                while not stop_event.is_set():
                    now = time.time()
                    if settings.duration is not None and now - start_time >= settings.duration:
                        break

                    sleep_s = next_start_time - now
                    if sleep_s > 0:
                        stop_event.wait(timeout=sleep_s)
                        if stop_event.is_set():
                            break

                    row_index += 1
                    point = monitor.read_one_point(row_index=row_index, start_time=start_time)
                    writer.writerow(asdict(point))
                    file.flush()

                    with self._lock:
                        self._rows_written += 1
                        self._latest_point = point
                        self._recent_points.append(point)
                        self._error = None

                    next_start_time += settings.interval

        except Exception as exc:
            with self._lock:
                self._error = str(exc)
        finally:
            with self._lock:
                if self._thread is threading.current_thread():
                    self._running = False
                    self._thread = None
                    self._stop_event = None
                    self._finished_at = time.time()


def make_handler(state: PowerDriftWebState):
    """创建 HTTP 请求处理类。"""

    class PowerDriftHandler(BaseHTTPRequestHandler):
        server_version = "PowerDriftWebUI/0.1"

        def do_GET(self) -> None:
            """处理页面、状态查询和 CSV 下载。"""

            try:
                parsed = urlparse(self.path)
                if parsed.path == "/" or parsed.path == "/index.html":
                    self._send_html(HTML_PAGE)
                elif parsed.path == "/api/status":
                    self._send_json(state.status())
                elif parsed.path == "/api/download":
                    self._send_csv_file(state.latest_csv_path())
                else:
                    self._send_error(HTTPStatus.NOT_FOUND, "Unknown route")
            except Exception as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, str(exc))

        def do_POST(self) -> None:
            """处理开始和停止记录。"""

            try:
                parsed = urlparse(self.path)
                if parsed.path == "/api/start":
                    settings = _settings_from_body(self._read_json())
                    self._send_json(state.start(settings))
                elif parsed.path == "/api/stop":
                    self._send_json(state.stop())
                else:
                    self._send_error(HTTPStatus.NOT_FOUND, "Unknown route")
            except Exception as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, str(exc))

        def log_message(self, format: str, *args: Any) -> None:
            """打印请求日志，方便调试。"""

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

        def _send_csv_file(self, path: Path) -> None:
            """把 CSV 文件作为附件返回给浏览器下载。"""

            if not path.exists():
                raise FileNotFoundError(str(path))
            payload = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_error(self, status: HTTPStatus, message: str) -> None:
            """用统一 JSON 格式返回错误。"""

            self._send_json({"ok": False, "error": message}, status=status)

    return PowerDriftHandler


def _settings_from_body(body: dict[str, Any]) -> PowerDriftSettings:
    """把前端 JSON 转成 PowerDriftSettings。"""

    settings = PowerDriftSettings(
        channel=str(body.get("channel", "ai2")),
        data_source=str(body.get("data_source", "direct_read")),
        interval=float(body.get("interval", 1.0)),
        samples=int(body.get("samples", 1000)),
        rate=float(body.get("rate", 1000.0)),
        terminal_config=str(body.get("terminal_config", "RSE")),
        min_val=float(body.get("min_val", -10.0)),
        max_val=float(body.get("max_val", 10.0)),
        timeout=float(body.get("timeout", 10.0)),
        duration=_optional_float(body.get("duration")),
        output_dir=Path(str(body.get("output_dir", DEFAULT_OUTPUT_DIR))),
        api_base_url=str(body.get("api_base_url", DEFAULT_BASE_URL)),
        power_per_volt=float(body.get("power_per_volt", 1.0)),
        zero_voltage=float(body.get("zero_voltage", 0.0)),
        allow_busy_ai=_bool_value(body.get("allow_busy_ai", False)),
    )
    _validate_settings(settings)
    return settings


def _validate_settings(settings: PowerDriftSettings) -> None:
    """检查前端传入的参数是否合理。"""

    if settings.interval <= 0:
        raise ValueError("interval must be > 0")
    if settings.samples < 1:
        raise ValueError("samples must be >= 1")
    if settings.rate <= 0:
        raise ValueError("rate must be > 0")
    if settings.timeout <= 0:
        raise ValueError("timeout must be > 0")
    if settings.duration is not None and settings.duration <= 0:
        raise ValueError("duration must be > 0")
    if settings.terminal_config not in ("RSE", "DIFF", "NRSE"):
        raise ValueError("terminal_config must be RSE, DIFF, or NRSE")
    if settings.data_source not in ("direct_read", "unified_stream"):
        raise ValueError("data_source must be direct_read or unified_stream")


def _optional_float(value: Any) -> float | None:
    """把空字符串转换成 None，否则转换成 float。"""

    if value in (None, ""):
        return None
    return float(value)


def _bool_value(value: Any) -> bool:
    """把 JSON 里的布尔值安全转换成 Python bool。"""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    return False


def _settings_for_json(settings: PowerDriftSettings | None) -> dict[str, Any] | None:
    """把设置转换成 JSON 友好的普通字典。"""

    if settings is None:
        return None
    data = asdict(settings)
    data["output_dir"] = str(settings.output_dir)
    return data


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>功率慢漂监测</title>
<style>
  :root {
    --bg: #f5f7f9;
    --panel: #ffffff;
    --line: #d7dde5;
    --text: #18202a;
    --muted: #657386;
    --blue: #1f6feb;
    --green: #16833a;
    --red: #c62828;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    height: 100vh;
    display: grid;
    grid-template-columns: 340px 1fr;
    background: var(--bg);
    color: var(--text);
    font-family: "Segoe UI", Arial, sans-serif;
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
    min-height: 0;
    display: grid;
    grid-template-rows: auto auto 1fr auto;
    gap: 10px;
    padding: 12px;
  }
  h1 { font-size: 18px; margin: 0 0 12px; }
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
  input[type="checkbox"] {
    width: auto;
    margin-right: 7px;
  }
  button {
    border: 0;
    border-radius: 6px;
    padding: 8px 10px;
    background: var(--blue);
    color: #fff;
    font-weight: 650;
    cursor: pointer;
  }
  button.green { background: var(--green); }
  button.secondary { background: #59636f; }
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
  .inline {
    display: flex;
    align-items: center;
    gap: 4px;
    margin-top: 10px;
    color: var(--text);
    font-size: 13px;
  }
  .status {
    border: 1px solid var(--line);
    background: var(--panel);
    border-radius: 8px;
    padding: 9px 10px;
    font-size: 13px;
    color: var(--muted);
  }
  .status.ok { color: var(--green); }
  .status.error { color: var(--red); }
  .metrics {
    display: grid;
    grid-template-columns: repeat(4, minmax(120px, 1fr));
    gap: 8px;
  }
  .metric {
    border: 1px solid var(--line);
    background: var(--panel);
    border-radius: 8px;
    padding: 10px;
    min-width: 0;
  }
  .metric label {
    margin: 0 0 5px;
    font-size: 11px;
  }
  .metric div {
    font-family: Consolas, "SFMono-Regular", monospace;
    font-size: 15px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .plot {
    min-height: 0;
    border: 1px solid var(--line);
    background: var(--panel);
    border-radius: 8px;
    overflow: hidden;
  }
  canvas {
    width: 100%;
    height: 100%;
    display: block;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    background: var(--panel);
    border: 1px solid var(--line);
  }
  th, td {
    border-bottom: 1px solid var(--line);
    padding: 6px 8px;
    text-align: right;
  }
  th:first-child, td:first-child { text-align: left; }
  .mono { font-family: Consolas, "SFMono-Regular", monospace; }
  @media (max-width: 900px) {
    body { grid-template-columns: 1fr; grid-template-rows: auto 1fr; }
    aside { border-right: 0; border-bottom: 1px solid var(--line); max-height: 45vh; }
    .metrics { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
  }
</style>
</head>
<body>
<aside>
  <h1>功率慢漂监测</h1>

  <h2>采集</h2>
  <label>数据来源</label>
  <select id="data_source">
    <option value="direct_read">direct_read：单独慢漂监测，独占读取 AI</option>
    <option value="unified_stream">unified_stream：读取已经运行的统一 AI 流</option>
  </select>
  <div class="grid2">
    <div>
      <label>AI 通道</label>
      <input id="channel" value="ai2">
    </div>
    <div>
      <label>接线方式</label>
      <select id="terminal_config">
        <option value="RSE">RSE</option>
        <option value="DIFF">DIFF</option>
        <option value="NRSE">NRSE</option>
      </select>
    </div>
  </div>
  <div class="grid2">
    <div>
      <label>记录间隔 / s</label>
      <input id="interval" type="number" value="1" min="0.01" step="0.5">
    </div>
    <div>
      <label>总时长 / s</label>
      <input id="duration" type="number" value="" min="0" step="60" placeholder="空=一直记录">
    </div>
  </div>
  <div class="grid2">
    <div>
      <label>每次点数</label>
      <input id="samples" type="number" value="1000" min="1" step="100">
    </div>
    <div>
      <label>采样率 / Hz</label>
      <input id="rate" type="number" value="1000" min="1" step="100">
    </div>
  </div>
  <div class="grid2">
    <div>
      <label>最小电压 / V</label>
      <input id="min_val" type="number" value="-10" step="0.5">
    </div>
    <div>
      <label>最大电压 / V</label>
      <input id="max_val" type="number" value="10" step="0.5">
    </div>
  </div>
  <label>读取超时 / s</label>
  <input id="timeout" type="number" value="10" min="0.1" step="1">

  <h2>换算</h2>
  <div class="grid2">
    <div>
      <label>功率/电压系数</label>
      <input id="power_per_volt" type="number" value="1" step="0.001">
    </div>
    <div>
      <label>零功率电压 / V</label>
      <input id="zero_voltage" type="number" value="0" step="0.001">
    </div>
  </div>
  <label>输出目录</label>
  <input id="output_dir" value="data/power_drift">
  <label>底层 API</label>
  <input id="api_base_url" value="http://127.0.0.1:8765">
  <label class="inline"><input id="allow_busy_ai" type="checkbox"> 允许 AI 忙时继续</label>

  <div class="actions">
    <button class="green" id="startBtn" onclick="startMonitor()">开始记录</button>
    <button class="secondary" id="stopBtn" onclick="stopMonitor()" disabled>停止记录</button>
  </div>
  <button class="secondary" id="downloadBtn" onclick="downloadCsv()" disabled style="margin-top:8px;">导出 CSV</button>
</aside>

<main>
  <div id="status" class="status">WebUI 已打开。请确认 8765 底层 API 服务正在运行。</div>

  <div class="metrics">
    <div class="metric"><label>状态</label><div id="m_running">停止</div></div>
    <div class="metric"><label>均值 / V</label><div id="m_mean">--</div></div>
    <div class="metric"><label>相对标准差</label><div id="m_rel">--</div></div>
    <div class="metric"><label>已写行数</label><div id="m_rows">0</div></div>
  </div>

  <div class="plot">
    <canvas id="trendCanvas"></canvas>
  </div>

  <table>
    <tbody id="detailRows">
      <tr><td>CSV</td><td>--</td></tr>
      <tr><td>最新时间</td><td>--</td></tr>
      <tr><td>标准差 / V</td><td>--</td></tr>
      <tr><td>峰峰值 / V</td><td>--</td></tr>
      <tr><td>估计功率</td><td>--</td></tr>
    </tbody>
  </table>
</main>

<script>
let latestStatus = null;
let pollTimer = null;

function getSettings() {
  return {
    channel: document.getElementById('channel').value,
    data_source: document.getElementById('data_source').value,
    interval: Number(document.getElementById('interval').value),
    samples: Number(document.getElementById('samples').value),
    rate: Number(document.getElementById('rate').value),
    terminal_config: document.getElementById('terminal_config').value,
    min_val: Number(document.getElementById('min_val').value),
    max_val: Number(document.getElementById('max_val').value),
    timeout: Number(document.getElementById('timeout').value),
    duration: document.getElementById('duration').value,
    output_dir: document.getElementById('output_dir').value,
    api_base_url: document.getElementById('api_base_url').value,
    power_per_volt: Number(document.getElementById('power_per_volt').value),
    zero_voltage: Number(document.getElementById('zero_voltage').value),
    allow_busy_ai: document.getElementById('allow_busy_ai').checked,
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

async function startMonitor() {
  try {
    const status = await postJson('/api/start', getSettings());
    updateStatus(status);
    setStatus('功率慢漂记录已启动。', 'ok');
  } catch (err) {
    setStatus(String(err.message || err), 'error');
    await refreshStatus(false);
  }
}

async function stopMonitor() {
  try {
    const status = await postJson('/api/stop', {});
    updateStatus(status);
    setStatus('功率慢漂记录已停止。', 'ok');
  } catch (err) {
    setStatus(String(err.message || err), 'error');
  }
}

function downloadCsv() {
  window.location.href = '/api/download';
}

async function refreshStatus(showError = true) {
  try {
    const response = await fetch('/api/status');
    const status = await response.json();
    if (!response.ok || status.ok === false) {
      throw new Error(status.error || '查询状态失败');
    }
    updateStatus(status);
  } catch (err) {
    if (showError) {
      setStatus(String(err.message || err), 'error');
    }
  }
}

function updateStatus(status) {
  latestStatus = status;
  const running = Boolean(status.running);
  const point = status.latest_point;
  const error = status.error ? `错误：${status.error}` : '';

  document.getElementById('startBtn').disabled = running;
  document.getElementById('stopBtn').disabled = !running;
  document.getElementById('downloadBtn').disabled = !status.csv_file;
  document.getElementById('m_running').textContent = error || (running ? '记录中' : '停止');
  document.getElementById('m_rows').textContent = status.rows_written || 0;

  if (point) {
    document.getElementById('m_mean').textContent = Number(point.mean_v).toExponential(6);
    document.getElementById('m_rel').textContent = point.rel_std_percent === null
      ? '--'
      : Number(point.rel_std_percent).toFixed(4) + '%';
  } else {
    document.getElementById('m_mean').textContent = '--';
    document.getElementById('m_rel').textContent = '--';
  }

  updateDetails(status);
  drawTrend();
}

function updateDetails(status) {
  const point = status.latest_point;
  const rows = document.getElementById('detailRows');
  if (!point) {
    rows.innerHTML = `
      <tr><td>CSV</td><td>${status.csv_file || '--'}</td></tr>
      <tr><td>最新时间</td><td>--</td></tr>
      <tr><td>标准差 / V</td><td>--</td></tr>
      <tr><td>峰峰值 / V</td><td>--</td></tr>
      <tr><td>估计功率</td><td>--</td></tr>
    `;
    return;
  }

  rows.innerHTML = `
    <tr><td>CSV</td><td class="mono">${status.csv_file || '--'}</td></tr>
    <tr><td>最新时间</td><td>${point.iso_time}</td></tr>
    <tr><td>标准差 / V</td><td>${Number(point.std_v).toExponential(6)}</td></tr>
    <tr><td>峰峰值 / V</td><td>${Number(point.peak_to_peak_v).toExponential(6)}</td></tr>
    <tr><td>估计功率</td><td>${Number(point.power_estimate).toExponential(6)}</td></tr>
  `;
}

function drawTrend() {
  const canvas = document.getElementById('trendCanvas');
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = Math.max(400, Math.floor(rect.width * devicePixelRatio));
  canvas.height = Math.max(260, Math.floor(rect.height * devicePixelRatio));
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, w, h);

  const points = latestStatus && latestStatus.recent_points ? latestStatus.recent_points : [];
  const padX = 58 * devicePixelRatio;
  const padY = 34 * devicePixelRatio;
  const x0 = padX;
  const y0 = padY;
  const plotW = w - 2 * padX;
  const plotH = h - 2 * padY;

  ctx.strokeStyle = '#d7dde5';
  ctx.lineWidth = 1 * devicePixelRatio;
  ctx.strokeRect(x0, y0, plotW, plotH);

  ctx.fillStyle = '#657386';
  ctx.font = `${12 * devicePixelRatio}px Consolas`;
  ctx.fillText('power_estimate / mean voltage trend', x0, 22 * devicePixelRatio);

  if (points.length < 2) {
    ctx.fillText('等待数据...', x0 + 12 * devicePixelRatio, y0 + 28 * devicePixelRatio);
    return;
  }

  const xs = points.map(p => Number(p.elapsed_s));
  const ys = points.map(p => Number(p.power_estimate));
  const minX = xs[0];
  const maxX = xs[xs.length - 1];
  let minY = Math.min(...ys);
  let maxY = Math.max(...ys);
  if (Math.abs(maxY - minY) < 1e-15) {
    minY -= 1;
    maxY += 1;
  }
  const yPad = (maxY - minY) * 0.08;
  minY -= yPad;
  maxY += yPad;

  ctx.beginPath();
  points.forEach((p, i) => {
    const x = x0 + ((Number(p.elapsed_s) - minX) / Math.max(1e-12, maxX - minX)) * plotW;
    const y = y0 + (1 - (Number(p.power_estimate) - minY) / Math.max(1e-30, maxY - minY)) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = '#1f6feb';
  ctx.lineWidth = 1.6 * devicePixelRatio;
  ctx.stroke();

  ctx.fillStyle = '#657386';
  ctx.fillText(`t ${minX.toFixed(1)}-${maxX.toFixed(1)} s`, x0, h - 10 * devicePixelRatio);
  ctx.fillText(`y ${minY.toExponential(3)}-${maxY.toExponential(3)}`, x0 + 180 * devicePixelRatio, h - 10 * devicePixelRatio);
}

window.addEventListener('resize', drawTrend);
refreshStatus(false);
pollTimer = setInterval(() => refreshStatus(false), 1000);
</script>
</body>
</html>
"""


def main() -> int:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="Run the photodetector power drift WebUI.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    state = PowerDriftWebState()
    handler = make_handler(state)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    print(f"Power drift WebUI running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping power drift WebUI.")
    finally:
        state.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
