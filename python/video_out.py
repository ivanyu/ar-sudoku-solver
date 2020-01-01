# -*- coding: utf-8 -*-
from typing import Optional

import cv2


class VideoOut:
    def __init__(self, filename):
        self._filename = filename + ".mp4"
        # self._fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        # self._fourcc = cv2.VideoWriter_fourcc(*"I422")
        self._fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._out: Optional[cv2.VideoWriter] = None
        self.i = 0

    def write_frame(self, frame):
        if self._out is None:
            self._out = cv2.VideoWriter(
                self._filename, self._fourcc, 30, (frame.shape[1], frame.shape[0])
            )

        cv2.imwrite(f"x-{self.i:03d}.jpg", frame)
        self.i += 1
        self._out.write(frame)

    def release(self):
        self._out.release()


class NoOpVideoOut:
    def __init__(self, *args, **kwargs):
        pass

    def write_frame(self, frame):
        pass

    def release(self):
        pass
