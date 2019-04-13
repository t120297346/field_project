"""
Created on Fri Apr  5 23:12:03 2019

@author: user
cut the image into 256-by-256-pieces
"""

import cv2
import numpy as np

image = cv2.imread('D:\\fruit_data\\raw_data\\fruit_test.tif')
labels = cv2.imread('D:\\fruit_data\\raw_data\\test_mask.tif')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_label = cv2.cvtColor(labels, cv2.COLOR_BGR2GRAY)
label_bw = cv2.threshold(gray_label, 230, 255, cv2.THRESH_BINARY)[1]


#get the real image
for row in range(image.shape[0]):
    if gray[row, image.shape[1] // 2] != 255:
        real_row_h = row
        break

for row in range(image.shape[0]):
    if gray[image.shape[0] - 1 - row, image.shape[1] // 2] != 255:
        real_row_b = image.shape[0] - 1 - row
        break

for col in range(image.shape[1]):
    if gray[np.int(image.shape[1] / 2), col] != 255:
        real_col_h = col
        break

for col in range(image.shape[1]):
    if gray[np.int(image.shape[1] / 2), image.shape[1] - 1 - col] != 255:
        real_col_b = image.shape[1] - 1 - col
        break
    
image = image[real_row_h:real_row_b, real_col_h:real_col_b, :]
label_bw = label_bw[real_row_h:real_row_b, real_col_h:real_col_b]
cv2.imwrite('D:\\fruit_data\\raw_data\\real_test.tif', image)
cv2.imwrite('D:\\fruit_data\\raw_data\\test.tif', label_bw)
x = np.int(image.shape[0] / 256)
y = np.int(image.shape[1] / 256)
num = 0


for i in range(x):
    for j in range(y):
        img = image[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256, :]
        label = label_bw[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256]
        
        cv2.imwrite('D:\\fruit_data\\test\\%d.tif' %num, img)
        cv2.imwrite('D:\\fruit_data\\test_label\\%d.tif' %num, label)
        num += 1



cv2.imshow('img', labels)
cv2.imshow('123', label_bw)
cv2.waitKey(0) 
cv2.destroyAllWindows()