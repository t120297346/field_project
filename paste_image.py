# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:46:12 2019

@author: user
"""

import cv2
import numpy as np
import os

files = os.listdir('D:\\fruit_data\\predict\\')
structure = cv2.imread('D:\\fruit_data\\raw_data\\test.tif')
x = structure.shape[0] // 256
y = structure.shape[1] // 256

for f in files:
    path = 'D:\\fruit_data\\predict\\' + f
    image = cv2.imread(path)
    
    
    
    
    
    