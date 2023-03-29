import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("output/172.bmp", cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 115, 255, cv2.THRESH_BINARY)
sobel_x = cv2.Sobel(binary, cv2.CV_8U, 1, 0, ksize=3)
sobel_y = cv2.Sobel(binary, cv2.CV_8U, 0, 1, ksize=3)
edges = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)


plt.imshow(img, cmap="gray")
plt.imshow(edges, cmap="gray")
plt.show()
