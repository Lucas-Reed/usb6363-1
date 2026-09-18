import unittest
import threading
from unittest.mock import MagicMock, patch

import numpy as np

from usb6363.pfi_frames import PfiFrameAssembler
from usb6363_core import DaqController, _counter_transition_samples


class PfiFrameTests(unittest.TestCase):
    def test_counter_events_across_blocks_and_uint32_wrap(self):
        events, previous = _counter_transition_samples(
            np.array([4294967295, 4294967295, 0, 0, 2], dtype=np.uint32), None, 100)
        self.assertEqual(events, [dict(sample_index=102, count=0), dict(sample_index=104, count=2)])
        events, previous = _counter_transition_samples(np.array([3, 3], dtype=np.uint32), previous, 105)
        self.assertEqual(events, [dict(sample_index=105, count=3)])
        self.assertEqual(_counter_transition_samples([], previous, 107), ([], 3))

    def test_two_megasamples_windows_follow_drifting_hardware_edges(self):
        assembler = PfiFrameAssembler(10000, 1)
        edges = [10003, 20006, 30005, 40008, 50009]
        events = [20005, 30005]
        frames = []
        for start in range(0, 60000, 10000):
            frames.extend(assembler.append([np.arange(start, start + 10000)], start,
                [dict(sample_index=s) for s in edges if start <= s < start + 10000],
                [dict(sample_index=s) for s in events if start <= s < start + 10000]))
        self.assertEqual([f["sample_start"] for f in frames], edges[:-1])
        for frame in frames:
            np.testing.assert_array_equal(frame["values"][0],
                np.arange(frame["sample_start"], frame["sample_start"] + 10000))
        self.assertEqual([f["pfi1_triggered"] for f in frames], [True, True, True, False])
        # The sample at 30005 belongs to two raw windows because this period
        # is one sample shorter than the configured 10000-point window.
        self.assertEqual(frames[1]["pfi0_period_samples"], 9999)

    def test_missing_trigger_retains_bounded_data_and_recovers(self):
        assembler = PfiFrameAssembler(10000, 1)
        frames = []
        for start in range(0, 200000, 10000):
            edges = [dict(sample_index=s) for s in (100, 150003, 160004, 170005)
                     if start <= s < start + 10000]
            frames.extend(assembler.append([np.arange(start, start + 10000)], start, edges, []))
            self.assertLessEqual(assembler.data.shape[1], 20000)
        self.assertEqual([f["sample_start"] for f in frames], [150003, 160004])
        self.assertTrue(all(f["values"].shape == (1, 10000) for f in frames))

    def test_worker_keeps_raw_counts_and_range_contiguous(self):
        controller = DaqController("Dev2")
        worker = controller._unified_ai_stream_worker
        with patch.object(controller, "_unified_ai_stream_worker"):
            controller.start_unified_ai_stream(channels=["ai0"], rate=2_000_000,
                samples_per_frame=10000, trigger_enabled=True, event_timeline_enabled=True)
            controller._unified_thread.join()
        stop = threading.Event()
        raw_cursor = 0
        read_buffer = np.empty((1, 10000), dtype=np.float64)
        counters = [object(), object()]
        def read_ai(**_kwargs):
            nonlocal raw_cursor
            raw_cursor += 10000
            read_buffer[0] = np.arange(raw_cursor - 10000, raw_cursor)
            return read_buffer
        def read_counter(task, samples, _timeout):
            indices = np.arange(raw_cursor - samples, raw_cursor)
            edges = [10003, 20006, 30009] if task is counters[0] else [25000]
            return np.searchsorted(edges, indices, side="right").tolist()
        publish = controller._publish_unified_window
        def publish_and_stop(*args):
            publish(*args)
            if controller._unified_frame_id == 2:
                stop.set()
        # Counter objects also need close() when the reader exits.
        counters = [MagicMock(), MagicMock()]
        with patch("usb6363_core.nidaqmx_driver.create_continuous_ai_task", return_value=MagicMock()) as create_ai, \
             patch("usb6363_core.nidaqmx_driver.create_buffered_pfi_counter_task", side_effect=counters), \
             patch("usb6363_core.nidaqmx_driver.create_numpy_ai_reader", return_value=lambda timeout: np.asarray(read_ai())), \
             patch("usb6363_core.nidaqmx_driver.create_numpy_counter_reader", side_effect=lambda task, samples: lambda timeout: np.asarray(read_counter(task, samples, timeout), dtype=np.uint32)), \
             patch.object(controller, "_publish_unified_window", side_effect=publish_and_stop):
            worker(controller._unified_settings, stop)
        self.assertEqual(create_ai.call_count, 1)
        self.assertIsNone(controller._unified_error)
        self.assertEqual(controller._unified_sample_counts, {"Dev2/ai0": 40000})
        np.testing.assert_array_equal(controller.read_unified_ai_range("ai0", 0, 40000)["values"], np.arange(40000))
        frames = list(controller._unified_frame_history)
        self.assertEqual([f["sample_start"] for f in frames], [10003, 20006])
        self.assertEqual([f["pfi1_triggered"] for f in frames], [False, True])
        read_buffer.fill(-999)
        for frame in frames:
            np.testing.assert_array_equal(frame["values"][0],
                np.arange(frame["sample_start"], frame["sample_start"] + 10000))
        controller.stop_unified_ai_stream()
