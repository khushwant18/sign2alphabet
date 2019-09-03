# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 16:28:25 2019

@author: Khushwant Rai
"""

import cv2
import numpy as np
import os

#counter for number of images clicked
count = 1
#flag to check if 786 images are clicked
stop = False
#output directory path for saving images
out_path = "D:\\uwo\\\computer_vision\\projects\\sign_lang_detector\\database\\"

def main(get_char):
    global stop, count, out_path
    
    cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False

    kernel = np.ones((5,5), np.uint8)
    
    while ret:
        ret, frame = cap.read()
        #new frame for detecting just hand
        frameNew = frame[57:257, 0:200]
        frameNew = cv2.cvtColor(frameNew, cv2.COLOR_BGR2GRAY)
        #seperating hand from background
        ret1, thresh = cv2.threshold(frameNew, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dilate = cv2.dilate(thresh, kernel, iterations=2)
        data = cv2.bitwise_and(frameNew, frameNew, mask= dilate)
        #wait for the input to initiate clicking images
        cv2.putText(frame,"Please press d whenever you are ready",(40,30), cv2.FONT_HERSHEY_PLAIN,1.0,(0,255,0))   
        key = cv2.waitKey(1)
        if key == ord('d') or stop == True:
            #crate folder for a sign to save 786 images
            if not os.path.exists(out_path+get_char+"\\"):
                os.mkdir(out_path+get_char+"\\")
                stop = True
            else:
                cv2.imwrite(out_path+get_char+"\\"+str(count)+".jpg", data)
                count+=1
                cv2.putText(frame,'clicking'+str(count),(40,50), cv2.FONT_HERSHEY_PLAIN,1.0,(0,255,0))
            if count == 786:
                stop = False
                count = 1
                cv2.putText(frame,'done', (40,70), cv2.FONT_HERSHEY_PLAIN,1.0,(0,255,0))
        cv2.imshow('live feed', frame)
        cv2.imshow('detect sign', data)
        #check for escape key to stop program
        if key == 27:
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #get character from user to set it as folder name
    get_char = input("Please enter the character: ")
    main(get_char)
    


