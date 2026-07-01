# USB-6363 本地 API

这个项目的目标是：让整台电脑上只有一个底层服务直接访问 `Dev2`，
其他 Python 子程序都通过 API 调用它，避免多个程序同时抢 USB-6363。

## 1. 快速检查硬件

```powershell
python minimal_dev2_check.py
```

## 2. 启动底层 API 服务

保持这个窗口一直运行：

```powershell
python usb6363_server.py
```

默认地址：

```text
http://127.0.0.1:8765
```

## 代码结构

```text
usb6363_core.py
    兼容入口，很薄。旧代码继续从这里 import DaqController。

usb6363/device.py
    设备信息、端子列表、通道名校验。

usb6363/nidaqmx_driver.py
    唯一直接 import / 调用 NI-DAQmx 的内部驱动层。

usb6363/ai.py
    AI 采样、动态采样率、后台缓存、统计、写 .npy 文件。
    不直接 import nidaqmx，只调用 nidaqmx_driver.py。

usb6363/ao.py
    AO 模拟输出。
    不直接 import nidaqmx，只调用 nidaqmx_driver.py。

usb6363/pfi.py
    PFI/数字线读写、PFI 边沿计数。
    不直接 import nidaqmx，只调用 nidaqmx_driver.py。

usb6363/controller.py
    把 device/ai/ao/pfi 组合成统一 DaqController。

usb6363_server.py
    HTTP API 服务。

usb6363_client.py
    给其他 Python 子程序调用的客户端。
```

## 3. 其他 Python 程序这样调用

```python
from usb6363_client import Usb6363Client

daq = Usb6363Client()

# 读取模拟输入 ai0。
ai0 = daq.read_ai(channel="ai0")
print(ai0["values"])

# 连续采样：只采 ai0，此时 ai0 会按单通道最大 2 MHz 运行。
daq.set_ai_channels(["ai0"])
print(daq.get_ai_status())
print(daq.get_ai_latest("ai0"))
print(daq.get_ai_stats("ai0"))

# 保存完整高速原始数据：接下来 0.1 秒的数据写入 .npy 文件。
# 返回值只包含文件路径和元数据，不会把大数组塞进 JSON。
capture = daq.record_ai_to_file(seconds=0.1)
print(capture["npy_file"])

# 连续采样：采 ai0 和 ai1，此时每个通道 500 kHz。
daq.set_ai_channels(["ai0", "ai1"])
print(daq.get_ai_status())

# 停止所有后台 AI 连续采样。
daq.clear_ai_channels()

# 输出模拟电压到 ao0。
daq.write_ao(channel="ao0", value=1.23)

# 读取 PFI0 的数字高低电平。
pfi0 = daq.read_pfi(line="PFI0")
print(pfi0["value"])

# 统计 PFI0 在 1 秒内的上升沿数量。
count = daq.count_pfi_edges(line="PFI0", counter="ctr0", seconds=1.0)
print(count["count"])
```

## API 路由

```text
GET  /health
GET  /api/devices
GET  /api/device
GET  /api/terminals
GET  /api/ai/read?channel=ai0&samples=1&rate=1000
GET  /api/ai/status
GET  /api/ai/latest?channel=ai0
GET  /api/ai/buffer?channel=ai0&max_samples=1000
GET  /api/ai/stats?channel=ai0&max_samples=10000
GET  /api/pfi/read?line=PFI0
GET  /api/pfi/count?line=PFI0&counter=ctr0&seconds=1.0&edge=RISING
POST /api/ai/subscribe
POST /api/ai/unsubscribe
POST /api/ai/set_channels
POST /api/ai/clear
POST /api/ai/record_to_file
POST /api/ao/write
POST /api/pfi/write
```

后台 AI 连续采样的采样率规则：

```text
0 个通道：不采样
1 个通道：每通道 2,000,000 samples/s
N 个通道：每通道 1,000,000 / N samples/s
```

设置 AI 连续采样通道的 JSON 例子：

```json
{
  "channels": ["ai0", "ai1"]
}
```

数据返回方式建议：

```text
实时状态、最近值、统计量、少量缓存：
    走 JSON，例如 get_ai_latest / get_ai_stats / get_ai_buffer。

完整高速原始波形：
    走文件，例如 record_ai_to_file，保存为 .npy。
```

记录当前 AI 采样流到文件的 JSON 例子：

```json
{
  "seconds": 1.0,
  "output_dir": "data",
  "prefix": "experiment_001"
}
```

`.npy` 文件的数据形状为：

```text
(通道数, 每通道采样点数)
```

例如只采 `ai0`，2 MHz，记录 1 秒：

```text
shape = (1, 2000000)
```

AO 输出 JSON 例子：

```json
{
  "channel": "ao0",
  "value": 1.23
}
```

PFI 输出 JSON 例子：

```json
{
  "line": "PFI0",
  "value": true
}
```

注意：AO 输出和 PFI 输出都会真实改变硬件端子电压/电平。
如果端子接了外部设备，请确认安全后再运行。

所有硬件操作都被 `usb6363_core.py` 里的锁保护。
HTTP 服务可以同时接收多个请求，但真正访问 USB-6363 时会排队执行。
