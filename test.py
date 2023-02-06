"""
test.py
take photo, record video, takeoff, translate, land
for the test of TELLO Edu

1st written by: Wonhee Lee
1st written on: 2023 JAN 28
    updated on: 2023 JAN 29; command 'streamoff' was unsuccessful error occurs
    updated on: 2023 FEB 05; added tello.streamoff(), so previous issue fixed
guided by: https://github.com/damiafuentes/DJITelloPy

"""

# IMPORT //////////////////////////////////////////////////////////////////////
from djitellopy import Tello
from vision_tools import take_photo, create_video, Streamer, Recorder


# CONNECT TO TELLO ////////////////////////////////////////////////////////////
tello = Tello()
tello.connect()    # enter SDK mode


# VISION SETUP ////////////////////////////////////////////////////////////////
# initiate visual stream ------------------------------------------------------
tello.streamon()
frame_read = tello.get_frame_read()

# stream setup ----------------------------------------------------------------
streamer = Streamer(frame_read)

# video setup -----------------------------------------------------------------
fps = 30
video = create_video('test.avi', 'XVID', fps, frame_read)   # fourcc for .mp4?
recorder = Recorder(video, frame_read, fps)


# MOVE ////////////////////////////////////////////////////////////////////////
# take a photo ----------------------------------------------------------------
take_photo("test.png", frame_read)

# start streamer & recorder ---------------------------------------------------
streamer.start()
recorder.start()

# move ------------------------------------------------------------------------
tello.takeoff()

tello.move('forward', 30)
tello.move('back', 30)
tello.move('right', 30)
tello.move('left', 30)
tello.rotate_counter_clockwise(90)

tello.land()

# finish streamer & recorder --------------------------------------------------
recorder.stop()
recorder.join()
streamer.stop()
streamer.join()

tello.streamoff()
