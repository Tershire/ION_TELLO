"""
test.py
take photo, record video, takeoff, translate, land
for the test of TELLO Edu

1st written by: Wonhee Lee
1st written on: 2023 JAN 28
    updated on: 2023 JAN 29; command 'streamoff' was unsuccessful error occurs
guided by: https://github.com/damiafuentes/DJITelloPy

"""

# IMPORT //////////////////////////////////////////////////////////////////////
from djitellopy import Tello
from vision_tools import take_photo, create_video, recorder


# CONNECT TO TELLO ////////////////////////////////////////////////////////////
tello = Tello()
tello.connect()    # enter SDK mode


# VISION SETUP ////////////////////////////////////////////////////////////////
# initiate visual stream
tello.streamon()
frame_read = tello.get_frame_read()

# video setup
fps = 30
video = create_video('test.avi', 'XVID', fps, frame_read)   # fourcc for .mp4?
recorder = recorder(video, frame_read, fps)


# MOVE ////////////////////////////////////////////////////////////////////////
# take a photo
take_photo("test.png", frame_read)

# start recorder
recorder.start()

# move
tello.takeoff()

tello.move('forward', 30)
tello.move('back', 30)
tello.move('right', 30)
tello.move('left', 30)
tello.rotate_counter_clockwise(90)

tello.land()

# finish recorder
recorder.stop()
recorder.join()
