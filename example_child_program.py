"""子程序调用 USB-6363 API 的示例。

运行这个脚本前，请先在另一个终端启动：
    python usb6363_server.py

这个文件模拟“未来你的其他 Python 子程序”应该怎样访问 USB-6363。
重点：它只 import usb6363_client，不直接 import nidaqmx。
"""

from __future__ import annotations

from usb6363_client import Usb6363Client


def main() -> int:
    # 创建客户端。默认连接 http://127.0.0.1:8765。
    daq = Usb6363Client()

    # 查询当前 server 管理的是哪块设备。
    print(daq.get_device())

    # 读取 Dev2/ai0 的一个电压值。
    ai_result = daq.read_ai(channel="ai0")
    print(f"ai0 = {ai_result['values']:.6f} V")

    # 后台连续采样示例 1：
    # 只采 ai0 时，server 会自动把 ai0 的采样率设为 2 MHz。
    status = daq.set_ai_channels(["ai0"])
    print(f"AI status with ai0 only: {status}")

    # 从后台缓存里读取 ai0 的最近一个值。
    # 刚启动连续采样后可能需要很短时间才有数据；正式程序可稍等几十毫秒再读。
    try:
        latest_ai0 = daq.get_ai_latest("ai0")
        print(f"latest ai0 = {latest_ai0['value']:.6f} V")
    except RuntimeError as exc:
        print(f"latest ai0 is not ready yet: {exc}")

    # 实时反馈建议用统计量，而不是把大量原始数据塞进 JSON。
    # 这里返回最近缓存数据的 mean/min/max/rms 等小结果。
    try:
        stats_ai0 = daq.get_ai_stats("ai0")
        print(f"ai0 stats = {stats_ai0}")
    except RuntimeError as exc:
        print(f"ai0 stats are not ready yet: {exc}")

    # 完整高速原始数据建议写文件。
    # 下面会把“接下来 0.05 秒”的 ai0 原始数据保存成 .npy 文件。
    # 注意：这不会通过 JSON 返回大数组，只返回文件路径和元数据。
    capture = daq.record_ai_to_file(seconds=0.05, prefix="example_ai0")
    print(f"capture file = {capture['npy_file']}")

    # 后台连续采样示例 2：
    # 同时采 ai0 和 ai1 时，多通道总采样率按 1 MHz 均分，
    # 所以每个通道会变成 500 kHz。
    status = daq.set_ai_channels(["ai0", "ai1"])
    print(f"AI status with ai0 + ai1: {status}")

    # 停止所有后台 AI 连续采样。
    daq.clear_ai_channels()

    # 读取 PFI0 的数字高低电平。
    # True 表示高电平，False 表示低电平。
    pfi_result = daq.read_pfi(line="PFI0")
    print(f"PFI0 = {pfi_result['value']}")

    # 用 ctr0 统计 PFI0 在 0.1 秒内出现了多少个上升沿。
    # 如果 PFI0 没有接外部脉冲信号，通常会读到 0。
    count_result = daq.count_pfi_edges(line="PFI0", counter="ctr0", seconds=0.1)
    print(f"PFI0 rising edges = {count_result['count']}")

    # 把 Dev2/ao0 设置为 0V。
    # 注意：这会真实改变 ao0 的输出。如果 ao0 接了外部设备，请确认安全后再运行。
    ao_result = daq.write_ao(channel="ao0", value=0.0)
    print(f"Set {ao_result['channel']} to {ao_result['value']:.3f} V")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
