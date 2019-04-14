# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 21:14:13 2019

@author: user
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def normalize(x):
    norm = np.linalg.norm(x)
    if norm == 0:
        return x
    return x / norm

def map_function(x, i):
    f = np.floor(255 * np.sum(normalize(x[0:i])) + 0.5)
    return f

def adjustable_histogram_equalization(img):
    hist, bins= np.histogram(img, 256)
    uniform = np.zeros(hist.shape)
    uniform[:] = np.sum(hist) / 255
    hist = normalize(hist)
    uniform = normalize(uniform)
    modified_hist , optimized_factor = min_tone_distortion(hist, uniform)
    print(optimized_factor)
    ade = np.interp(img.flatten(), bins[0:-1,], modified_hist)
    ade = np.uint8(((ade - np.min(ade)) / (np.max(ade) - np.min(ade))) * 255)
    ade_gray = np.uint8(ade.reshape(img.shape))
    return ade_gray

def tone_distortion_measurement(hf):
    max_distortion = None
    for i in range(256):
        for j in range(i + 1, 256):
            if hf[i] > 0 and hf[j] > 0 and map_function(hf, i) == map_function(hf, j):
                distortion = j - i
                max_distortion = distortion if (max_distortion == None or distortion > max_distortion) else max_distortion
    return max_distortion

def min_tone_distortion(hist, uniform, factor_range = 0.1):
    min_distortion = None
    for factor in np.arange(0, factor_range, 0.1):
        modified_hist = (1/(1 + factor)) * hist + (factor / (1 + factor)) * uniform
        #modified_hist = normalize(modified_hist)
        distortion = tone_distortion_measurement(modified_hist)
        print(distortion)
        if min_distortion == None or distortion < min_distortion:
            min_distortion = distortion
            optimized_factor = factor
            optimized_modified_hist = modified_hist
    return optimized_modified_hist, optimized_factor

def adjusted_gamma_correction(img, gamma = 1.0):
    n_image = np.uint8(np.power((img /255), gamma) * 255)
    return(n_image)
    
# main 
image = cv2.imread('D:\\field_project1\\color_balance\\real_train.tif')

row = image.shape[0]
col = image.shape[1]
'''Global contrast adaptive enhancement'''
#linear stretching
for i in range(3):
    max = np.max(image[:, :, i])
    min = np.min(image[:, :, i])
    image[:, :, i] = np.uint8(255 * ((image[:, :, i] - min) / (max - min)))
# turn to grayscale
gray = np.uint8(image[:, :, 0] * 0.114 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.299)
#adopted enhance grayscale image_
ade_gray = adjustable_histogram_equalization(gray)

#enhance rgb image
enhanced_rgb = np.zeros(image.shape)
divider = ade_gray / (gray + 1)
divider = divider.reshape(row, col, 1)
greater_than_one = divider > 1
greater_than_one = greater_than_one.reshape(row, col, 1)


factor1 = ((255 - ade_gray) / (256 - gray))
factor1 = np.dstack((factor1, factor1, factor1))
gray = np.dstack((gray, gray, gray))
ade_gray = np.dstack((ade_gray, ade_gray, ade_gray))
num = ((255 - ade_gray) / (256 - gray)) * (image - gray) + ade_gray

enhanced_rgb = image * greater_than_one * divider + num * (greater_than_one ^ 1)

'''
for i in range(row):
    for j in range(col):
        enhanced_rgb[i, j, :] = image[i, j, :] * divider[i, j] if greater_than_one[i, j] else num[i, j, :]
'''
cv2.imshow('ade', np.uint8(enhanced_rgb))
cv2.imshow('ade', ade_gray)
cv2.imwrite('D:\\field_project1\\color_balance\\enhanced_rgb.tif', np.uint8(enhanced_rgb))
cv2.waitKey(0)
cv2.destroyAllWindows()




