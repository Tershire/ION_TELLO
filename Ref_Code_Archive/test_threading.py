"""
test_threading.py
test code to test threading

1st written by: Wonhee Lee
1st written on: 2023 APR 04

https://docs.python.org/2/library/threading.html#event-objects
https://stackoverflow.com/questions/25698628/python-threading-script-is-not-starting-working-properly
"""

# IMPORT //////////////////////////////////////////////////////////////////////
import threading
import keyboard

# CLASS ///////////////////////////////////////////////////////////////////////
class SubThread(threading.Thread):
    """
    "Killable Thread" to show live stream
    """

    def __init__(self):
        super(SubThread, self).__init__()
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        self.func()

    def func(self):
        while True:
            print("sub")


# RUN /////////////////////////////////////////////////////////////////////////
# create Thread
subThread = SubThread()
subThread.start()

# main thread -----------------------------------------------------------------
while True:
    print("main")

# ending ----------------------------------------------------------------------
# finish Thread
subThread.stop()
subThread.join()
