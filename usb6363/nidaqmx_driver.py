"""唯一直接调用 NI-DAQmx 的内部驱动层。

重要约定：
    usb6363 包里只有这个文件可以 import nidaqmx。
    其他模块只能调用本文件提供的函数，不能直接碰 nidaqmx。

这样做是为了让硬件访问边界非常清楚，方便排查“谁在占用设备”。
"""

from __future__ import annotations

import time
from typing import Any

import nidaqmx
from nidaqmx.constants import AcquisitionType, Edge, TerminalConfiguration
from nidaqmx.system import System


def list_devices() -> list[dict[str, str]]:
    """列出当前 NI-DAQmx 能看到的设备。"""

    return [
        {
            "name": device.name,
            "product_type": device.product_type,
            "serial_num": str(device.serial_num),
        }
        for device in System.local().devices
    ]


def get_device_info(device_name: str) -> dict[str, str]:
    """读取指定设备的基本信息。"""

    device = _get_device(device_name)
    return {
        "name": device.name,
        "product_type": device.product_type,
        "serial_num": str(device.serial_num),
    }


def list_signal_terminals(device_name: str) -> dict[str, list[str]]:
    """列出 PFI、数字线、计数器等常用端子。"""

    device = _get_device(device_name)
    terminals = [str(terminal) for terminal in device.terminals]
    return {
        "pfi": [terminal for terminal in terminals if "/PFI" in terminal],
        "digital_input_lines": [line.name for line in device.di_lines],
        "digital_output_lines": [line.name for line in device.do_lines],
        "counter_input_channels": [channel.name for channel in device.ci_physical_chans],
        "counter_output_channels": [channel.name for channel in device.co_physical_chans],
    }


def read_ai_voltage(
    device_name: str,
    physical_channel: str,
    samples: int,
    rate: float,
    terminal_config_name: str,
    min_val: float,
    max_val: float,
    timeout: float,
) -> float | list[float]:
    """读取 AI 电压。

    samples=1 时返回 float；samples>1 时返回 list[float]。
    """

    _get_device(device_name)
    config = _terminal_config(terminal_config_name)

    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan(
            physical_channel,
            terminal_config=config,
            min_val=min_val,
            max_val=max_val,
        )
        if samples == 1:
            return float(task.read(timeout=timeout))

        task.timing.cfg_samp_clk_timing(
            rate=rate,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=samples,
        )
        raw_values = task.read(
            number_of_samples_per_channel=samples,
            timeout=timeout,
        )
        return [float(value) for value in raw_values]


def capture_ai_frame(
    device_name: str,
    physical_channels: list[str],
    samples: int,
    rate: float,
    terminal_config_name: str,
    min_val: float,
    max_val: float,
    timeout: float,
    trigger_source: str | None = None,
    trigger_edge_name: str = "RISING",
) -> list[list[float]]:
    """同步读取多路 AI 的一帧数据。

    这里的“一帧”指：在同一个 NI-DAQmx Task 里，同时添加多个 AI 通道，
    然后用同一个采样时钟读取 samples 个点。这样 ai0/ai1 的时间轴是一致的，
    适合后面的双峰锁定程序读取 FP 透射信号和 PZT 监视信号。

    trigger_source 不为 None 时，AI Task 会等待这个数字端子的边沿后才开始采样。
    例如 trigger_source="/Dev2/PFI0"，trigger_edge_name="RISING"。
    """

    _get_device(device_name)
    if not physical_channels:
        raise ValueError("physical_channels must not be empty")

    config = _terminal_config(terminal_config_name)

    with nidaqmx.Task() as task:
        for physical_channel in physical_channels:
            task.ai_channels.add_ai_voltage_chan(
                physical_channel,
                terminal_config=config,
                min_val=min_val,
                max_val=max_val,
            )

        task.timing.cfg_samp_clk_timing(
            rate=rate,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=samples,
        )
        if trigger_source is not None:
            task.triggers.start_trigger.cfg_dig_edge_start_trig(
                trigger_source=trigger_source,
                trigger_edge=_edge(trigger_edge_name),
            )
            task.start()

        raw_values = task.read(
            number_of_samples_per_channel=samples,
            timeout=timeout,
        )
        return split_ai_read_values(raw_values, len(physical_channels))


def read_continuous_ai_chunk(
    task: Any,
    samples_per_read: int,
    channel_count: int,
    timeout: float,
) -> list[list[float]]:
    """从连续 AI Task 里读取一块数据，并统一成 list[list[float]]。"""

    raw_values = task.read(
        number_of_samples_per_channel=samples_per_read,
        timeout=timeout,
    )
    return split_ai_read_values(raw_values, channel_count)


def create_continuous_ai_task(
    channels: list[str],
    rate: float,
    samples_per_read: int,
    terminal_config_name: str = "RSE",
    min_val: float = -10.0,
    max_val: float = 10.0,
    start_trigger_source: str | None = None,
    start_trigger_edge_name: str = "RISING",
) -> Any:
    """创建并启动连续 AI Task。

    返回的 task 仍由调用方负责 close。这个函数保留在 driver 内，
    是为了让 nidaqmx 的创建细节集中在一个文件里。

    start_trigger_source 不为 None 时，连续采集会等待这个数字边沿后启动。
    注意：这是“启动触发”，不是每一帧都重新触发。
    """

    config = _terminal_config(terminal_config_name)
    task = nidaqmx.Task()
    try:
        for channel in channels:
            task.ai_channels.add_ai_voltage_chan(
                channel,
                terminal_config=config,
                min_val=min_val,
                max_val=max_val,
            )
        task.timing.cfg_samp_clk_timing(
            rate=rate,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=samples_per_read * 10,
        )
        if start_trigger_source is not None:
            task.triggers.start_trigger.cfg_dig_edge_start_trig(
                trigger_source=start_trigger_source,
                trigger_edge=_edge(start_trigger_edge_name),
            )
        task.start()
        return task
    except Exception:
        task.close()
        raise


def write_ao_voltage(
    device_name: str,
    physical_channel: str,
    value: float,
    min_val: float,
    max_val: float,
    timeout: float,
) -> None:
    """输出 AO 静态电压。"""

    _get_device(device_name)
    with nidaqmx.Task() as task:
        task.ao_channels.add_ao_voltage_chan(
            physical_channel,
            min_val=min_val,
            max_val=max_val,
        )
        task.write(float(value), auto_start=True, timeout=timeout)


def read_digital_line(device_name: str, physical_line: str, timeout: float) -> bool:
    """读取数字线或 PFI 电平。"""

    _get_device(device_name)
    with nidaqmx.Task() as task:
        task.di_channels.add_di_chan(physical_line)
        return bool(task.read(timeout=timeout))


def write_digital_line(
    device_name: str,
    physical_line: str,
    value: bool,
    timeout: float,
) -> None:
    """写数字线或 PFI 电平。"""

    _get_device(device_name)
    with nidaqmx.Task() as task:
        task.do_channels.add_do_chan(physical_line)
        task.write(bool(value), auto_start=True, timeout=timeout)


def count_pfi_edges(
    device_name: str,
    terminal: str,
    physical_counter: str,
    seconds: float,
    edge_name: str,
    timeout: float,
) -> dict[str, Any]:
    """统计 PFI 边沿数量。"""

    _get_device(device_name)
    edge_value = _edge(edge_name)
    with nidaqmx.Task() as task:
        channel = task.ci_channels.add_ci_count_edges_chan(
            physical_counter,
            edge=edge_value,
            initial_count=0,
        )
        channel.ci_count_edges_term = terminal
        task.start()
        time.sleep(seconds)
        count = int(task.read(timeout=timeout))

    return {
        "edge": edge_value.name,
        "count": count,
    }


def split_ai_read_values(raw_values: Any, channel_count: int) -> list[list[float]]:
    """把 nidaqmx.Task.read 的返回值统一整理成 list[list[float]]。"""

    if channel_count == 1:
        if isinstance(raw_values, list):
            return [[float(value) for value in raw_values]]
        return [[float(raw_values)]]

    return [
        [float(value) for value in channel_values]
        for channel_values in raw_values
    ]


def _get_device(device_name: str) -> Any:
    """确认目标设备存在，并返回 NI-DAQmx 设备对象。"""

    system = System.local()
    device_names = [device.name for device in system.devices]
    if device_name not in device_names:
        raise RuntimeError(
            f"{device_name!r} was not found in NI-DAQmx. "
            "Check NI MAX / NI-DAQmx device name and USB connection."
        )
    return system.devices[device_name]


def _terminal_config(name: str) -> TerminalConfiguration:
    """把 RSE/NRSE/DIFF 等字符串转换成 NI-DAQmx 枚举。"""

    key = name.strip().upper()
    try:
        return TerminalConfiguration[key]
    except KeyError as exc:
        valid = ", ".join(config.name for config in TerminalConfiguration)
        raise ValueError(f"terminal_config must be one of: {valid}") from exc


def _edge(name: str) -> Edge:
    """把 RISING/FALLING 字符串转换成 NI-DAQmx 枚举。"""

    key = name.strip().upper()
    try:
        return Edge[key]
    except KeyError as exc:
        valid = ", ".join(edge_value.name for edge_value in Edge)
        raise ValueError(f"edge must be one of: {valid}") from exc
