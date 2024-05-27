import os
import time
import numpy as np
import cv2

et_v = cv2.VideoCapture('data/test_videos/office/world.mp4')
p_v = cv2.VideoCapture('data/test_videos/office/phone_v1.avi')

while(et_v.isOpened() or p_v.isOpened()):
  ret1, frame1 = et_v.read()
  ret2, frame2 = p_v.read()

  if ret1 == True or ret2 == True:
    if ret1 == True:

        frame1=cv2.resize(frame1,(1000,640))
        cv2.imshow('Frame ET',frame1)
        # cv2.moveWindow("Frame ET", 0, 0) 
    if ret2 == True:

        frame2=cv2.resize(frame2,(1000,640)) 
        cv2.imshow('Frame Ph',frame2)
        cv2.moveWindow("Frame Ph", 0, 0) 

    # Press Q on keyboard to  exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
et_v.release()
p_v.release()