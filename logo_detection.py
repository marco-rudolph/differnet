from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import cv2

from skimage.color import rgb2gray
from skimage import feature

for i in range(12):
    break

I1 = io.imread("bottle_logo_defective1.jpg")
cv2.imwrite("origin.jpg", I1)

image = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
cv2.imwrite("Gray.jpg", image)

image = cv2.GaussianBlur(image, (21, 21), 0)

# seg_img = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 1)
seg_img = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
cv2.imwrite('binary.jpg', seg_img)

h0, w0 = seg_img.shape

if seg_img[0][0] == 255:
    for i in range(w0):
        for j in range(2):
            if seg_img[j][i] == 255:
                seg_img[j][i] = 0
    for j in range(h0):
        for i in range(2):
            if seg_img[j][i] == 255:
                seg_img[j][i] = 0

cv2.imwrite("Threshold.jpg", seg_img)

num, labels, stats, centroids = cv2.connectedComponentsWithStats(seg_img)

h, w = seg_img.shape
print(h * w)

first_stat = 0
second_stat = 0

# Largest area should be the bottle
# Second largest area should be the logo
for istat in stats:
    if istat[4] > 2000 and istat[4] > second_stat:
        if istat[4] > first_stat:
            first_stat = istat[4]
        else:
            second_stat = istat[4]
            logo_stat = istat

print(logo_stat)
cv2.rectangle(I1, (logo_stat[0], logo_stat[1]), (logo_stat[0] + logo_stat[2], logo_stat[1] + logo_stat[3]),
              (255, 0, 255), 2)

cv2.imwrite("segmented.jpg", I1)
