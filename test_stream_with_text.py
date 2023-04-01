"""
test_stream_with_text.py
show text in the video stream too
for the test of TELLO Edu

1st written by: Wonhee Lee
1st written on: 2023 FEb 05
    updated on:
guided by: https://docs.opencv.org/3.4/dd/d43/tutorial_py_video_display.html

"""

# IMPORT //////////////////////////////////////////////////////////////////////
from djitellopy import Tello
from streamer_with_text import Streamer
from VisionTools import create_video
from datetime import datetime


# CONNECT TO TELLO ////////////////////////////////////////////////////////////
tello = Tello()
tello.connect()    # enter SDK mode


# VISION SETUP ////////////////////////////////////////////////////////////////
# initiate visual stream
tello.streamon()
frame_read = tello.get_frame_read()

# stream setup
fps = 30
video = create_video('test_stream_with_text.avi', 'XVID', fps, frame_read)   # fourcc for .mp4?
streamer = Streamer(frame_read, tello, video, fps)


# MOVE ////////////////////////////////////////////////////////////////////////
# # take a photo
# take_photo("test.png", frame_read)

# start recorder
streamer.start()

# move
tello.takeoff()

tello.move('forward', 30)
tello.move('back', 30)
tello.move('right', 30)
tello.move('left', 30)
tello.rotate_counter_clockwise(90)

tello.land()

# finish streamer
streamer.stop()
streamer.join()

tello.streamoff()