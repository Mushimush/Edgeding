import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import convolve2d

# Gaussian kernell noise reduction



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

    vertical = ndimage.convolve(img, sobel_v)
    horizontal = ndimage.convolve(img, sobel_h)
    edged_img = np.sqrt(np.square(horizontal) + np.square(vertical))
    gradient_direction = np.arctan2(vertical, horizontal)

    return edged_img, gradient_direction


def hysteresis_thresholding(zero_crossings, low_threshold, high_threshold):
    # Create a binary image with the same dimensions as the zero-crossings image
    thresholded = np.zeros_like(zero_crossings)

    # Set pixels above the high threshold to white
    thresholded[zero_crossings > high_threshold] = 255

    # Find indices of pixels between the low and high thresholds
    r, c = np.where((zero_crossings >= low_threshold) &
                    (zero_crossings <= high_threshold))

    # Iterate over the indices and check neighbors
    for i in range(len(r)):
        row = r[i]
        col = c[i]
        if thresholded[row-1:row+2, col-1:col+2].max() == 255:
            thresholded[row, col] = 255

    return thresholded


# Load the input image
img = cv2.imread('0.jpg', cv2.IMREAD_GRAYSCALE)
gaussian_kernel = gaussian_kernel(3, 5)
img_smooth = convolve2d(img, gaussian_kernel, mode="same")
edge_magnitude, theta = sobel_filters(img_smooth)

# Compute the Laplacian of the image along the gradient direction
laplacian = ndimage.filters.laplace(img, mode='reflect')
laplacian_along_gradient = laplacian * np.cos(theta)

# Apply a Gaussian filter to the Laplacian image
log_sigma = 1.0
laplacian_along_gradient_smooth = ndimage.filters.gaussian_filter(
    laplacian_along_gradient, sigma=log_sigma)

zero_crossings = np.zeros(laplacian_along_gradient_smooth.shape)
zero_crossings[np.where(
    np.diff(np.signbit(laplacian_along_gradient_smooth)))] = 255

low_threshold = 20
high_threshold = 30

thresholded = hysteresis_thresholding(
    zero_crossings, low_threshold, high_threshold)

plt.imshow(thresholded, cmap='gray')
plt.title('Hysteresis Thresholding')
plt.show()

# Display the original image
plt.subplot(1, 5, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')

# Display the smoothed image
plt.subplot(1, 5, 2)
plt.imshow(img_smooth, cmap='gray')
plt.title('Smoothed')

# Display the edges after non-maximum suppression
plt.subplot(1, 5, 3)
plt.imshow(edge_magnitude, cmap='gray')
plt.title('Edges after Sobel')

# Display the edges after non-maximum suppression
plt.subplot(1, 5, 4)
plt.imshow(zero_crossings, cmap='gray')
plt.title('zerocrossing')

# Display the edges after non-maximum suppression
plt.subplot(1, 5, 5)
plt.imshow(thresholded, cmap='gray')
plt.title('zerocrossing')
plt.show()
