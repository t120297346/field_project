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

def min_tone_distortion(hist, uniform, factor_range = 1):
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

def global_enhanced(image, gray):
    #adopted enhance grayscale image_
    ade_gray = adjustable_histogram_equalization(gray)

    #enhance rgb image
    divider = ade_gray / (gray + 1)
    divider = divider.reshape(gray.shape[0], gray.shape[1], 1)
    greater_than_one = divider > 1
    greater_than_one = np.dstack((greater_than_one, greater_than_one, greater_than_one))
    #greater_than_one = greater_than_one.reshape((gray.shape), 1)

    gray = np.dstack((gray, gray, gray))
    ade_gray = np.dstack((ade_gray, ade_gray, ade_gray))
    num = ((255 - ade_gray) / (256 - gray)) * (image - gray) + ade_gray

    global_enhanced_rgb = np.uint8(image * greater_than_one * divider + num * (greater_than_one ^ 1))
    
    return global_enhanced_rgb

def local_enhanced(image, gray):
    clahe_mask = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(256,256))
    clahe = clahe_mask.apply(gray)
    
    divider = clahe / gray
    divider = divider.reshape(gray.shape[0], gray.shape[1], 1)
    greater_than_one = divider > 1
    
    factor1 = ((255 - clahe) / (256 - gray))
    factor1 = np.dstack((factor1, factor1, factor1))
    gray = np.dstack((gray, gray, gray))
    clahe = np.dstack((clahe, clahe, clahe))
    num = ((255 - clahe) / (256 - gray)) * (image - gray) + clahe
    
    local_enhanced_rgb = np.uint8(image * greater_than_one * divider + num * (greater_than_one ^ 1))
    
    return local_enhanced_rgb
    
# main 
image = cv2.imread('D:\\field_project1\\color_balance\\real_train.tif')

row = image.shape[0]
col = image.shape[1]

#linear stretching
for i in range(3):
    max = np.max(image[:, :, i])
    min = np.min(image[:, :, i])
    image[:, :, i] = np.uint8(255 * ((image[:, :, i] - min) / (max - min)))
# turn to grayscale
gray = np.uint8(image[:, :, 0] * 0.114 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.299)
#global adaptive enhancement
global_enhanced_image = global_enhanced(image, gray)
#local contrast adaptive enhancement
local_enhanced_image = local_enhanced(image, gray)
#variational-based fusion
alpha = 0.5
beta = 0.5 
gamma = 1

I_z = np.uint8(np.ones([row, col, 3]))

#data fidelity
'''coefficient setting'''
sigma1 = 0.2
'''contrast measure factor'''
laplacian_G = cv2.Laplacian(global_enhanced_image, cv2.CV_16S)
laplacian_G = cv2.convertScaleAbs(laplacian_G) / 255
laplacian_E = cv2.Laplacian(local_enhanced_image, cv2.CV_16S)
laplacian_E = cv2.convertScaleAbs(laplacian_E) / 255
'''bright measure factor'''
b_g = global_enhanced_image / 255
b_e = local_enhanced_image / 255
brightness_g = np.exp(-np.power((b_g - 0.5), 2) / (2 * np.power(sigma1, 2)))
brightness_e = np.exp(-np.power((b_e - 0.5), 2) / (2 * np.power(sigma1, 2)))
'''weight map'''
boolean_g = b_g > b_e


#cv2.imshow('global', laplacian_G)
#cv2.imshow('local', laplacian_E)
#cv2.imwrite('D:\\field_project1\\color_balance\\global.tif', global_enhanced_image)
#cv2.imwrite('D:\\field_project1\\color_balance\\local_256.tif', local_enhanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()




