"""
vision_tools.py
functions and classes for vision-related useful operations for TELLO Edu

1st written by: Wonhee Lee
1st written on: 2023 JAN 28
    updated on: 2023 JAN 29; improved the "closure" of thread but is still unsatisfactory
guided by: https://github.com/damiafuentes/DJITelloPy
           https://stackoverflow.com/questions/323972/is-there-any-way-to-kill-a-thread
           https://stackoverflow.com/questions/46921161/creating-a-thread-that-stops-when-its-own-stop-flag-is-set

"""

# IMPORT //////////////////////////////////////////////////////////////////////
from djitellopy import Tello
import cv2 as cv
import time
import threading


# DEF /////////////////////////////////////////////////////////////////////////
def take_photo(file_name, frame_read):
    # take a photo
    cv.imwrite(file_name, frame_read.frame)


# -----------------------------------------------------------------------------
def create_video(file_name, fourcc, fps, frame_read):
    # create an OpenCV video object
    [h, w, _] = frame_read.frame.shape
    video = cv.VideoWriter(file_name, cv.VideoWriter_fourcc(*fourcc), fps,
                           (w, h))
    # 2nd param. is 4-character code of codec used to compress the frames

    return video


# CLASS ///////////////////////////////////////////////////////////////////////
class recorder(threading.Thread):
    def __init__(self, video, frame_read, fps):
        super(recorder, self).__init__()
        self._stop_event = threading.Event()

        self.video = video
        self.frame_read = frame_read
        self.fps = fps

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        self.record_video(self.video, self.frame_read, self.fps)

    def record_video(self, video, frame_read, fps):
        # record a video
        while True:
            video.write(frame_read.frame)
            time.sleep(1 / fps)
            if self.stopped():
                break

        video.release()
