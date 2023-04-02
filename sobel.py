from PIL import Image
import numpy as np
from scipy.signal import convolve2d
from scipy import ndimage
import cv2
from scipy import signal
from skimage.morphology import thin
import matplotlib.pyplot as plt


# Load the input image
img = Image.open("retina2movingavg/332.bmp")

# Convert the image to grayscale
img_gray = img.convert("L")

# Convert the image to a numpy array
img_arr = np.array(img_gray)

# Define the Gaussian kernel
# gaussian_kernel = (
#     np.array(
#         [
#             [1, 4,  6,  4,  1],
#             [4, 16, 24, 16, 4],
#             [6, 24, 36, 24, 6],
#             [4, 16, 24, 16, 4],
#             [1, 4,  6,  4,  1]
#         ]
#     )
#     / 256
# )


def gaussian_kernel(size, sigma):
    x, y = np.meshgrid(np.arange(-size // 2 + 1, size // 2 + 1),
                       np.arange(-size // 2 + 1, size // 2 + 1))
    kernel = np.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2)))
    return kernel / np.sum(kernel)


gaussian_kernel = gaussian_kernel(7, 10)

# Apply the Gaussian kernel using convolve2d
img_smooth = convolve2d(img_arr, gaussian_kernel, mode="same")

# Define the Sobel operator kernels
sobel_v = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_h = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])


# Apply the Sobel operator using convolve
vertical = ndimage.convolve(img_smooth, sobel_v)
horizontal = ndimage.convolve(img_smooth, sobel_h)


# Compute the edge image using the squared magnitudes of horizontal and vertical
edged_img = np.sqrt(np.square(horizontal) + np.square(vertical))

# # Convert the filtered image back to a PIL image
img_filtered_pil = Image.fromarray(edged_img.astype(np.uint8))
_, binary = cv2.threshold(edged_img, 20, 255, cv2.THRESH_BINARY)

plt.subplot(1, 2, 1)
plt.imshow(binary, cmap='gray')
plt.title('Binary')

plt.subplot(1, 2, 2)
plt.imshow(edged_img, cmap='gray')
plt.titl
plt.show()
