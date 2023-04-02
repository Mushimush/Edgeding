from PIL import Image
import numpy as np
from scipy.signal import convolve2d
from scipy import ndimage
import cv2

# Load the input image
img = Image.open("retina2/337.bmp")

# Convert the image to grayscale
img_gray = img.convert("L")

# Convert the image to a numpy array
img_arr = np.array(img_gray)

# Define the Gaussian kernel
gaussian_kernel = (
    np.array(
        [
            [1, 4,  6,  4,  1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4,  6,  4,  1]
        ]
    )
    / 256
)

# Apply the Gaussian kernel using convolve2d
img_smooth = convolve2d(img_arr, gaussian_kernel, mode="same")

# Define the Roberts cross operator kernels
roberts_cross_v = np.array([[1, 0],
                            [0, -1]])

roberts_cross_h = np.array([[0, 1],
                            [-1, 0]])

# Apply the Roberts cross operator using convolve
vertical = ndimage.convolve(img_smooth, roberts_cross_v)
horizontal = ndimage.convolve(img_smooth, roberts_cross_h)

# Compute tpythe edge image using the squared magnitudes of horizontal and vertical
edged_img = np.sqrt(np.square(horizontal) + np.square(vertical))
edged_img = edged_img.astype('uint8')

# # Convert the filtered image back to a PIL image
img_filtered_pil = Image.fromarray(edged_img.astype(np.uint8))

# # Save the filtered image
img_filtered_pil.save("MEOW.jpg")
img_smooth = img_smooth.astype(np.uint8)
cv2.imshow("Filtered Image", edged_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
