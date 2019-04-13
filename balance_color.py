    # -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:59:22 2019

@author: user
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def adjustable_histogram_equalization(img):
    hist, bins= np.histogram(img, 256)
    uniform = np.zeros(hist.shape)
    uniform = np.sum(hist) / 255
    factor = 2
    modified_hist = (1/(1 + factor)) * hist + (factor / (1 + factor)) * uniform
    
    ade = np.interp(img.flatten(), bins[0:-1,], modified_hist)
    ade = np.uint8(((ade - np.min(ade)) / (np.max(ade) - np.min(ade))) * 255)
    ade_gray = np.uint8(ade.reshape(img.shape))
    return ade_gray

image = cv2.imread('D:\\field_project\\color_balance\\real_train.tif')



#rgb2hsv 將亮度平均
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
mean_saturated = np.uint8(np.mean(hsv_image[:, :, 1]))
mean_value = np.uint8(np.mean(hsv_image[:, :, 2]))
eq_value = cv2.equalizeHist(hsv_image[:, :, 2])
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(hsv_image[:, :, 2])
tuned_v = adjustable_histogram_equalization(hsv_image[:, :, 2])
n_hsv_image = np.zeros(hsv_image.shape)
n_hsv_image = hsv_image

n_hsv_image[:, :, 2] = tuned_v
n_image = cv2.cvtColor(n_hsv_image, cv2.COLOR_HSV2BGR)
'''

tuned_b = np.uint8(np.power(image[:, :, 0] / np.uint8(np.max(image[:, :, 0])), 0.9) * 255)
tuned_g = np.uint8(np.power(image[:, :, 1] / np.uint8(np.max(image[:, :, 1])), 0.9) * 255)
tuned_r = np.uint8(np.power(image[:, :, 2] / np.uint8(np.max(image[:, :, 2])), 0.9) * 255)
n_image = np.zeros(image.shape)
n_image[:, :, 0] = tuned_b
n_image[:, :, 1] = tuned_g
n_image[:, :, 2] = tuned_r
'''
'''
#色彩平衡
b, g, r = cv2.split(image)
B = np.mean(b)
G = np.mean(g)
R = np.mean(r)
K = (R + G + B) / 3
Kb = K / B
Kg = K / G
Kr = K / R
cv2.addWeighted(b, Kb, 0, 0, 0, b)
cv2.addWeighted(g, Kg, 0, 0, 0, g)
cv2.addWeighted(r, Kr, 0, 0, 0, r)
merged = cv2.merge([b,g,r])
'''


#cv2.imwrite('D:\\field_project\\color_balance\\hsv_vequalize.tif', n_hsv_image)
cv2.imwrite('D:\\field_project\\color_balance\\ade_lambda2.tif', n_image)
#cv2.imwrite('D:\\field_project\\color_balance.tif', merged)




