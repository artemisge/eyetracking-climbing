from time import sleep, time
import zmq
import msgpack as serializer
import cv2
from matplotlib import pyplot as plt
import numpy as np

# PupilLabs/Tobii API in python + open camera with OpenCV
# https://www.e-consystems.com/blog/camera/technology/how-to-access-cameras-using-opencv-with-python/
# https://docs.pupil-labs.com/core/developer/network-api/

# To connect to Pupil Core API
# Have to start PupilCapture first
# Setup zmq context and remote helper
ctx = zmq.Context()
socket = zmq.Socket(ctx, zmq.REQ)
socket.connect("tcp://127.0.0.1:50020")

# Measure round trip delay
t = time()
socket.send_string("t")
print(socket.recv_string())
print("Round trip command delay:", time() - t)

# set current Pupil time to 0.0
socket.send_string("T 0.0")
print(socket.recv_string())

# start recording
sleep(1)

# sleep(5)
# stop recording -> on the bottom of the page

# Control phone Camera

video = cv2.VideoCapture(0)
# 'https://10.240.53.33:8080/video') # -> works with no latency
# ('rtsp://10.240.53.33:8080/h264_pcm.sdp') -> WORKS with less latency
# ('rtsp://10.240.53.33:8080/h264_ulaw.sdp') -> WORKS with latency

if not video.isOpened():
    print("Error: Could not open video stream.")
    exit()

cv2.namedWindow('frame',0)
cv2.resizeWindow('frame',1000,640)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('phone_v1_1.avi', fourcc, 29.83, (1000,  640))

eyetracker_start_time = time()
phone_start_time = 0
phone_end_time = 0
hasStarted = False

# eye tracker start
socket.send_string("R")
print(socket.recv_string())

while(True):
    ret, frame = video.read()
    
    if frame is not None:
        vidout=cv2.resize(frame,(1000,640))  
        out.write(vidout)
        cv2.imshow('frame',frame)

    if not hasStarted:
            phone_start_time = time()
            hasStarted = True
            
    if cv2.waitKey(1) & 0xFF == ord('q'):
        phone_end_time = time()
        break

# stop recording of eyetracker
socket.send_string("r")
eyetracker_end_time = time()
print(socket.recv_string())

video.release()
out.release()
cv2.destroyAllWindows()

print("eyetracker " + str(eyetracker_start_time) + str(eyetracker_end_time))
print("phone " + str(phone_start_time) + str(phone_end_time))