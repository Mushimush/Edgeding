import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import convolve2d


def gaussian_kernel(size, sigma):
    x, y = np.meshgrid(np.arange(-size // 2 + 1, size // 2 + 1),
                       np.arange(-size // 2 + 1, size // 2 + 1))
    kernel = np.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2)))
    return kernel / np.sum(kernel)


def sobel_filters(img):
    sobel_v = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_h = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    # sobel_x = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)
    # sobel_y = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)
    # edged_img = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    vertical = ndimage.convolve(img, sobel_v)
    horizontal = ndimage.convolve(img, sobel_h)
    edged_img = np.sqrt(np.square(horizontal) + np.square(vertical))
    gradient_direction = np.arctan2(vertical, horizontal)

    return edged_img, gradient_direction


def non_max_suppression(img, theta):
    # Round the gradient direction to the nearest 45 degrees
    theta = np.rad2deg(theta)
    theta[theta < 0] += 180
    theta[theta < 22.5] = 0
    theta[(theta >= 22.5) & (theta < 67.5)] = 45
    theta[(theta >= 67.5) & (theta < 112.5)] = 90
    theta[(theta >= 112.5) & (theta < 157.5)] = 135
    theta[theta >= 157.5] = 0

    # Perform non-maximum suppression
    M, N = img.shape
    suppressed = np.zeros((M, N), dtype=np.float32)
    for i in range(1, M-1):
        for j in range(1, N-1):
            if theta[i, j] == 0:
                if (img[i, j] > img[i-1, j]) and (img[i, j] > img[i+1, j]):
                    suppressed[i, j] = img[i, j]
            elif theta[i, j] == 45:
                if (img[i, j] > img[i-1, j-1]) and (img[i, j] > img[i+1, j+1]):
                    suppressed[i, j] = img[i, j]
            elif theta[i, j] == 90:
                if (img[i, j] > img[i, j-1]) and (img[i, j] > img[i, j+1]):
                    suppressed[i, j] = img[i, j]
            elif theta[i, j] == 135:
                if (img[i, j] > img[i-1, j+1]) and (img[i, j] > img[i+1, j-1]):
                    suppressed[i, j] = img[i, j]

    return suppressed


def hysteresis_thresholding(img, low_thresh_ratio=0.09, high_thresh_ratio=0.15):
    low_thresh = np.max(img) * low_thresh_ratio
    print(low_thresh)
    high_thresh = np.max(img) * high_thresh_ratio
    print(high_thresh)

    # Apply high and low thresholds
    strong_edges = img > high_thresh
    weak_edges = (img >= low_thresh) & (img <= high_thresh)

    # Connect weak edges to strong edges
    M, N = img.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if strong_edges[i, j]:
                weak_edges[i-1:i+2, j-1:j+2] = True

    return weak_edges


# Load the input image
img = cv2.imread('retina2movingavg/332.bmp', cv2.IMREAD_GRAYSCALE)


# Apply Gaussian smoothing
gaussian_kernel = gaussian_kernel(9, 3)
img_smooth = convolve2d(img, gaussian_kernel, mode="same")

# Detect edges using Sobel filters
edge_magnitude, theta = sobel_filters(img_smooth)

# Perform non-maximum suppression
suppressed_edges = non_max_suppression(edge_magnitude, theta)

# Apply hysteresis thresholding
edge_map = hysteresis_thresholding(suppressed_edges)


# Visualize the binary edge map
# Display the original image
# plt.subplot(1, 4, 1)
# plt.imshow(img, cmap='gray')
# plt.title('Original')

# # Display the smoothed image
# plt.subplot(1, 4, 2)
# plt.imshow(img_smooth, cmap='gray')
# plt.title('Smoothed')

# # Display the edges after non-maximum suppression
# plt.subplot(1, 4, 3)
# plt.imshow(edge_magnitude, cmap='gray')
# plt.title('Edges after Sobel')

# Display the edges after non-maximum suppression
plt.subplot(1, 1, 1)
plt.imshow(edge_map, cmap='gray')
plt.title('final')
plt.show()
