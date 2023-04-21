"""
test_stream.py
what will be the best video stream method?
for the test of TELLO Edu

1st written by: Wonhee Lee
1st written on: 2023 FEb 05
    updated on:
guided by: https://docs.opencv.org/3.4/dd/d43/tutorial_py_video_display.html

"""

# IMPORT //////////////////////////////////////////////////////////////////////
from djitellopy import Tello
from VisionTools import take_photo, create_video, Recorder, Streamer


# CONNECT TO TELLO ////////////////////////////////////////////////////////////
tello = Tello()
tello.connect()    # enter SDK mode


# VISION SETUP ////////////////////////////////////////////////////////////////
# initiate visual stream
tello.streamon()
frame_read = tello.get_frame_read()

# stream setup
streamer = Streamer(frame_read)


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
tello.rotate_counter_clockwise(180)

tello.land()

# finish streamer
streamer.stop()
streamer.join()

tello.streamoff()
