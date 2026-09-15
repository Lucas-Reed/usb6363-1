"""Fixed-size raw AI windows anchored to every hardware PFI0 edge."""

from collections import deque

import numpy as np


class PfiFrameAssembler:
    def __init__(self, samples_per_frame: int, channels: int) -> None:
        self.size = samples_per_frame
        self.data = np.empty((channels, 0), dtype=float)
        self.start = 0
        self.edges: deque[dict] = deque()
        self.pfi1: deque[dict] = deque()

    def append(self, values, start: int, pfi0: list[dict], pfi1: list[dict]) -> list[dict]:
        chunk = np.asarray(values, dtype=float)
        if start != self.start + self.data.shape[1]:
            raise RuntimeError("PFI frame input is not contiguous")
        self.data = np.concatenate((self.data, chunk), axis=1)
        self.edges.extend(pfi0)
        self.pfi1.extend(pfi1)
        end = start + chunk.shape[1]
        frames = []
        while len(self.edges) >= 2:
            edge, following = self.edges[0], self.edges[1]
            anchor = int(edge["sample_index"])
            next_anchor = int(following["sample_index"])
            window_end = anchor + self.size
            if max(window_end, next_anchor) > end:
                break
            # The window contains original samples; it is never interpolated.
            # Also flag the remainder of this trigger cycle if it is longer.
            events = [e for e in self.pfi1 if anchor <= e["sample_index"] < max(window_end, next_anchor)]
            offset = anchor - self.start
            frames.append({
                "sample_start": anchor,
                "sample_end": window_end,
                "pfi0_period_samples": next_anchor - anchor,
                "pfi0_events": [dict(edge)],
                "pfi1_events": events,
                "pfi1_triggered": bool(events),
                "values": self.data[:, offset:offset + self.size].copy(),
            })
            self.edges.popleft()
        # With missing triggers keep at most a window plus the current chunk.
        # A late next edge cannot recover an arbitrarily old trigger window.
        keep = max(self.start, end - self.size - chunk.shape[1])
        if self.edges:
            keep = max(keep, int(self.edges[0]["sample_index"]))
        while self.edges and self.edges[0]["sample_index"] < keep:
            self.edges.popleft()
        while self.pfi1 and self.pfi1[0]["sample_index"] < keep:
            self.pfi1.popleft()
        self.data = self.data[:, keep - self.start:].copy()
        self.start = keep
        return frames
