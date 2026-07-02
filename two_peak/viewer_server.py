"""双峰波形查看器的 HTTP 服务。"""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

from two_peak.viewer_capture import (
    capture_frame,
    frame_summary,
    get_area_trend_status,
    get_frame_stream_latest,
    get_frame_stream_status,
    list_saved_frames,
    load_saved_frame,
    measure_latest_frame,
    save_latest_frame,
    start_area_trend,
    start_frame_stream,
    stop_area_trend,
    stop_frame_stream,
)
from two_peak.viewer_state import ViewerState


STATIC_DIR = Path(__file__).with_name("static")


def load_viewer_html() -> str:
    """读取前端页面。"""

    return (STATIC_DIR / "viewer.html").read_text(encoding="utf-8")


def make_handler(state: ViewerState):
    """创建 HTTP 请求处理类。"""

    class TwoPeakViewerHandler(BaseHTTPRequestHandler):
        server_version = "TwoPeakViewer/0.2"

        def do_GET(self) -> None:
            """处理页面和只读 API。"""

            try:
                if self.path == "/" or self.path.startswith("/?"):
                    self._send_html(load_viewer_html())
                elif self.path == "/api/defaults":
                    self._send_json(state.active_web_defaults())
                elif self.path == "/api/latest":
                    self._send_json(
                        {
                            "has_frame": state.latest_frame is not None,
                            "frame": frame_summary(state.latest_frame),
                            "measurement": state.latest_measurement,
                        }
                    )
                elif self.path == "/api/stream/status":
                    self._send_json(get_frame_stream_status(state))
                elif self.path == "/api/stream/latest":
                    self._send_json(get_frame_stream_latest(state))
                elif self.path == "/api/trend/status":
                    self._send_json(get_area_trend_status(state))
                elif self.path == "/api/samples":
                    self._send_json(list_saved_frames(state))
                else:
                    self._send_error(HTTPStatus.NOT_FOUND, "Unknown route")
            except Exception as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, str(exc))

        def do_POST(self) -> None:
            """处理会改变查看器状态的 API。"""

            try:
                if self.path == "/api/capture":
                    body = self._read_json()
                    frame = capture_frame(state, body)
                    state.latest_frame = frame
                    state.latest_measurement = None
                    self._send_json(frame)
                elif self.path == "/api/stream/start":
                    body = self._read_json()
                    self._send_json(start_frame_stream(state, body))
                elif self.path == "/api/stream/stop":
                    self._send_json(stop_frame_stream(state))
                elif self.path == "/api/trend/start":
                    body = self._read_json()
                    self._send_json(start_area_trend(state, body))
                elif self.path == "/api/trend/stop":
                    self._send_json(stop_area_trend(state))
                elif self.path == "/api/measure":
                    body = self._read_json()
                    measurement = measure_latest_frame(state, body)
                    state.latest_measurement = measurement
                    self._send_json(measurement)
                elif self.path == "/api/save":
                    body = self._read_json()
                    saved = save_latest_frame(state, body)
                    self._send_json(saved)
                elif self.path == "/api/load":
                    body = self._read_json()
                    loaded = load_saved_frame(state, body)
                    self._send_json(loaded)
                elif self.path == "/api/defaults/save":
                    body = self._read_json()
                    defaults = state.save_user_defaults(body.get("parameters", body))
                    self._send_json(defaults)
                elif self.path == "/api/defaults/reset":
                    defaults = state.reset_user_defaults()
                    self._send_json(defaults)
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
