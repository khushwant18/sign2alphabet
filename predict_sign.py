# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 16:33:23 2019

@author: Khushwant Rai
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
import cv2

alpha = '#ABCDEFGHIKLMNORUWY'
#read cvs file containg traig data
dataset = pd.read_csv("D:\\uwo\\\computer_vision\\projects\\sign_lang_detector\\database\\dataset.csv")

def create_model(dataset):
    #fetching data from cvs file
    x = dataset.iloc[:,1:].values.reshape(len(dataset),28,28,1)
    y = dataset.iloc[:,0].values
    y = to_categorical(y)
    x=np.array(x)
    y=np.array(y)
    
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
    #creating and training CNN model
    model = Sequential()
    model.add(Convolution2D(64, 3, data_format='channels_last', activation='relu', input_shape=(28,28,1)))
    model.add(Convolution2D(32, 3, data_format='channels_last', activation='relu', input_shape=(28,28,1)))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(19, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test[1:20], y_test[1:20]), epochs=3)
    return model

def main():
    global alpha, dataset
    model = create_model(dataset)

    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False

    kernel = np.ones((5,5), np.uint8)
   
    while ret:
        ret, frame = cap.read()
        #new frame for detecting just hand
        newFrame = frame[57:257, 0:200]
        newFrame = cv2.cvtColor(newFrame, cv2.COLOR_BGR2GRAY)
        ret1, thresh = cv2.threshold(newFrame, 0, 255, cv2.THRESH_BINARY_INV +cv2.THRESH_OTSU)
        dilate = cv2.dilate(thresh, kernel, iterations =3)
        #seperating hand from background
        data = cv2.bitwise_and(newFrame, newFrame, mask=dilate)
        #resizing to 28X28
        mask_new = cv2.resize(data, None, fx=0.14, fy=0.14,interpolation=cv2.INTER_AREA )
        mask_new = mask_new.ravel().reshape(1,28,28,1)
        #predicting
        y = model.predict(mask_new)
        y_predict = np.round(y)
        output = np.argmax(y_predict)
        #displayig the result on the live feed frame
        cv2.putText(frame, 'The given sign is: '+alpha[output],(60,50), cv2.FONT_HERSHEY_PLAIN, 3.0,(0,0,255),2)
        cv2.imshow('live feed', frame)
        cv2.imshow('detect sign', data)
        if cv2.waitKey(1) == 27:
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()