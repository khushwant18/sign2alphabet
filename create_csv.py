# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 21:56:42 2019

@author: Khushwant Rai
"""

import cv2
import numpy as np
import os
import glob
import pandas as pd

#directory which contains image data
directory = "D:\\uwo\\\computer_vision\\projects\\sign_lang_detector\\database\\"
#path where csv file will be saved
path = "D:\\uwo\\\computer_vison\\projects\\sign_lang_detector\\database\\dataset.csv"
#folder names
alpha= 'abcdefghiklmnoruwy'

def main():
    for f in range(len(alpha)):
        paths = os.path.join(directory+alpha[f]+'\\',"*jpg")
        imgs_paths = glob.glob(paths)
        for p in imgs_paths:
            imgs = cv2.imread(p)
            imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)
            #resize images to 28x28
            imgs = cv2.resize(imgs, None, fx=0.14,fy=0.14,interpolation=cv2.INTER_AREA)
            imgs = imgs.ravel()
            #save id number in first column
            imgs = np.insert(imgs,0,f+1)
            #save data to csv file
            df = pd.DataFrame(imgs).T
            df.to_csv(path, mode='a',index=False, header=False)
        
if __name__ == '__main__':
    main()
    



