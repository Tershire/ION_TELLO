"""
streamer_with_text.py
class for video stream with text

1st written by: Wonhee Lee
1st written on: 2023 FEB 05
guided by:

"""

# IMPORT //////////////////////////////////////////////////////////////////////
from djitellopy import Tello
import cv2 as cv
import time
import threading


# SETTING /////////////////////////////////////////////////////////////////////
# font
font       = cv.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (15, 20, 255)
thickness  = 2
line_type  = 2


# CLASS ///////////////////////////////////////////////////////////////////////
class Streamer(threading.Thread):
    def __init__(self, frame_read, tello, video, fps):
        super(Streamer, self).__init__()
        self._stop_event = threading.Event()

        self.frame_read = frame_read
        self.tello = tello
        self.video = video
        self.fps = fps

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        self.stream_video(self.frame_read, self.tello, self.video, self.fps)

    def stream_video(self, frame_read, tello, video, fps):
        # stream & record a video
        while True:
            frame = frame_read.frame

            # text
            v_x = tello.get_speed_x()
            v_y = tello.get_speed_y()
            v_z = tello.get_speed_z()
            string = f'velocity = ({v_x: 3.2f}, {v_y: 3.2f}, {v_z: 3.2f})'

            cv.putText(frame, string,
                       (5, 30),
                       font,
                       font_scale,
                       font_color,
                       thickness,
                       line_type)

            # stream
            cv.imshow('TELLO VIEW', frame)
            cv.waitKey(1)

            # record
            video.write(frame)
            time.sleep(1 / fps)

            if self.stopped():
                break

        video.release()
        