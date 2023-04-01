"""
VisionTools.py
functions and classes for vision-related useful operations for TELLO Edu

1st written by: Wonhee Lee
1st written on: 2023 JAN 28
    updated on: 2023 JAN 29; improved the "closure" of thread but is still unsatisfactory
    updated on: 2023 FEB 05; added Streamer and testing
guided by: https://github.com/damiafuentes/DJITelloPy
           https://stackoverflow.com/questions/323972/is-there-any-way-to-kill-a-thread
           https://stackoverflow.com/questions/46921161/creating-a-thread-that-stops-when-its-own-stop-flag-is-set

"""

# IMPORT //////////////////////////////////////////////////////////////////////
from djitellopy import Tello
import cv2 as cv
import time
import threading


# FUNCTION ////////////////////////////////////////////////////////////////////
def take_photo(file_name, frame_read):
    """
    take a photo
    :param file_name: file name to save
    :param frame_read: Tello().frame_read
    :return:
    """
    cv.imwrite(file_name, frame_read.frame)


# -----------------------------------------------------------------------------
def create_video(file_name, fourcc, fps, frame_read):
    """
    create an OpenCV video object
    :param file_name: file name to save
    :param fourcc: 4-character code of codec used to compress video
    :param fps: frames per second
    :param frame_read: Tello().frame_read
    :return: OpenCV video object
    """
    [h, w, _] = frame_read.frame.shape
    video = cv.VideoWriter(file_name, cv.VideoWriter_fourcc(*fourcc), fps,
                           (w, h))

    return video


# CLASS ///////////////////////////////////////////////////////////////////////
class Streamer(threading.Thread):
    """
    "Killable Thread" to show live stream
    """
    def __init__(self, frame_read):
        super(Streamer, self).__init__()
        self._stop_event = threading.Event()
        # self.timelast = time.time() - 5
        self.frame_read = frame_read

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        self.stream_video(self.frame_read)

    def stream_video(self, frame_read):
        # stream a video
        while True:
            cv.imshow('TELLO VIEW', frame_read.frame)
            cv.waitKey(1)

            if self.stopped():
                break

"""
    def refresh_5Hz(self):
        if time.time() - self.timelast >= 1/5:
            timelast = time.time()
            fun1
            fun2
"""

class CapStreamer(threading.Thread):
    """
    "Killable Thread" to show live stream
    """
    def __init__(self, cap):
        super(CapStreamer, self).__init__()
        self._stop_event = threading.Event()
        self.cap = cap

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        self.stream_video(self.cap)

    def stream_video(self, cap):
        # stream a video
        while True:
            ret, frame = cap.read()
            cv.imshow('TELLO VIEW', frame)
            cv.waitKey(1)

            if self.stopped():
                break


# -----------------------------------------------------------------------------
class Recorder(threading.Thread):
    """
    "Killable Thread" to record a video
    """
    def __init__(self, video, frame_read, fps):
        super(Recorder, self).__init__()
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
