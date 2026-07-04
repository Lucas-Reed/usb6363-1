# USB-6363 本地统一采集服务

这个项目的目标是：整台电脑上只有一个底层服务直接访问 NI USB-6363，其他 Python 程序和 WebUI 都通过 HTTP API 调用它，避免多个程序同时抢 AI 采集硬件。

当前推荐主线是 **unified AI stream**：

```text
usb6363_server.py
    直接管理 USB-6363

ai_stream_console.py
    启动/停止统一 AI 流，决定底层真实采哪些 AI 通道

two_peak_viewer.py / power_drift_webui.py / 其他子程序
    只读取统一 AI 流，不自己打开新的 AI task
```

## 正常启动顺序

1. 检查硬件是否能被 NI-DAQmx 看到：

```powershell
python minimal_dev2_check.py
```

2. 启动底层 API 服务：

```powershell
python usb6363_server.py
```

默认地址：

```text
http://127.0.0.1:8765
```

3. 启动统一 AI 控制台：

```powershell
python ai_stream_console.py
```

浏览器打开：

```text
http://127.0.0.1:8768
```

在这里设置：

```text
channels = ai0,ai1,ai2
rate
samples_per_frame
terminal_config
min_val / max_val
PFI trigger
resync_every_frames
```

然后启动 unified stream。

4. 启动双峰波形查看器：

```powershell
python two_peak_viewer.py
```

浏览器打开：

```text
http://127.0.0.1:8766
```

双峰查看器现在只读取 unified stream。它里面的 `channels` 输入框只表示“显示/分析哪些通道”，不会改变底层统一流真实采集的通道。

5. 启动功率慢漂 WebUI：

```powershell
python power_drift_webui.py
```

浏览器打开：

```text
http://127.0.0.1:8767
```

功率慢漂默认使用 `unified_stream`。如果统一流包含 `ai2`，功率慢漂就可以读取 `ai2` 的统计量并写 CSV。

## 子程序推荐调用方式

新写的 Python 子程序应该只消费 unified stream：

```python
from usb6363_client import Usb6363Client

daq = Usb6363Client()

status = daq.get_unified_ai_stream_status()
if not status["running"]:
    raise RuntimeError("请先用 ai_stream_console.py 启动统一 AI 流")

latest = daq.get_unified_ai_latest("ai0")
stats = daq.get_unified_ai_stats("ai1", max_samples=1000)
buffer = daq.get_unified_ai_buffer("ai2", max_samples=100)
```

完整示例见：

```text
example_child_program.py
```

## 推荐 AI API

```text
POST /api/ai/unified/start
POST /api/ai/unified/stop
GET  /api/ai/unified/status
GET  /api/ai/unified/latest_frame
GET  /api/ai/unified/latest?channel=ai0
GET  /api/ai/unified/buffer?channel=ai0&max_samples=1000
GET  /api/ai/unified/stats?channel=ai0&max_samples=10000
```

重要边界：

```text
只有统一 AI 控制台等底层控制入口应该启动/停止 unified stream。
双峰、功率慢漂、普通子程序只读取 unified stream。
```

## Legacy AI API

下面这些旧接口暂时保留，只用于旧脚本兼容、单独调试和逐步迁移：

```text
GET  /api/ai/read
GET  /api/ai/status
GET  /api/ai/latest
GET  /api/ai/buffer
GET  /api/ai/stats
POST /api/ai/subscribe
POST /api/ai/unsubscribe
POST /api/ai/set_channels
POST /api/ai/clear
POST /api/ai/record_to_file
POST /api/ai/capture_frame
POST /api/ai/frame_stream/start
POST /api/ai/frame_stream/stop
GET  /api/ai/frame_stream/status
GET  /api/ai/frame_stream/latest
```

这些接口可能直接或间接打开旧 AI task。新功能不要再基于它们开发。

## AO 和 PFI

AO/PFI 不属于 AI 采集 task，可以继续通过底层 API 使用：

```text
POST /api/ao/write
GET  /api/pfi/read?line=PFI0
POST /api/pfi/write
GET  /api/pfi/count?line=PFI0&counter=ctr0&seconds=1.0&edge=RISING
```

检查 PFI 上升沿：

```powershell
python pfi_rising_counter.py --line PFI0 --counter ctr0 --seconds 1
```

注意：AO/PFI 写操作会真实改变硬件端子电压/电平，接外部设备时请先确认安全。

## 代码结构

```text
usb6363/nidaqmx_driver.py
    唯一直接 import / 调用 NI-DAQmx 的内部驱动层。

usb6363_core.py
    设备信息、通道校验、统一 AI 流、AO、PFI 等核心逻辑。

usb6363_server.py
    HTTP API 服务。

usb6363_client.py
    给其他 Python 子程序调用的客户端。

ai_stream_console.py
    统一 AI 流控制台，负责启动/停止 unified stream。

two_peak/
    双峰波形查看器和后续锁定算法。现在只读取 unified stream。

power_drift_monitor.py / power_drift_webui.py
    光电探测器功率慢漂监测。默认读取 unified stream。

legacy/
    旧双峰程序参考快照，只作需求和算法参考。
```
