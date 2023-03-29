import cv2
import numpy as np
from scipy.signal import convolve2d
from scipy import ndimage
import matplotlib.pyplot as plt


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


def hysteresis_thresholding(img, low_thresh_ratio=0.25, high_thresh_ratio=0.35):
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


# Load the image
img = cv2.imread('retina2movingavg/172.bmp')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Create a Gaussian pyramid with 6 levels
gaussian_pyramid = [img]
for i in range(6):
    img = cv2.pyrDown(img)
    gaussian_pyramid.append(img)

# Create a Laplacian pyramid by subtracting each level of the Gaussian pyramid from the next level
laplacian_pyramid = []
for i in range(5):
    img_up = cv2.pyrUp(gaussian_pyramid[i+1])
    img_up = cv2.resize(
        img_up, (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
    laplacian = cv2.subtract(gaussian_pyramid[i], img_up)
    laplacian_pyramid.append(laplacian)

# Add the last level of the Gaussian pyramid to the Laplacian pyramid
laplacian_pyramid.append(gaussian_pyramid[5])

# ret, thresh_img = cv2.threshold(
#     laplacian_pyramid[0], 20, 255, cv2.THRESH_BINARY)

# cv2.imshow('hi', laplacian_pyramid[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Apply Gaussian smoothing
gaussian_kernel = gaussian_kernel(5, 3)
img_smooth = convolve2d(laplacian_pyramid[0], gaussian_kernel, mode="same")


# Detect edges using Sobel filters
edge_magnitude, theta = sobel_filters(img_smooth)

# Perform non-maximum suppression
suppressed_edges = non_max_suppression(edge_magnitude, theta)
# ret, thresh_img = cv2.threshold(
#     suppressed_edges, 3, 255, cv2.THRESH_BINARY)
# Apply hysteresis thresholding
print(np.min(suppressed_edges), np.max(suppressed_edges))
edge_map = hysteresis_thresholding(suppressed_edges)

plt.imshow(edge_map, cmap='gray')
plt.title('final')
plt.show()
