import cv2
import numpy as np

# Load the input image
img = cv2.imread('retina2movingavg/172.bmp', cv2.IMREAD_GRAYSCALE)

# Apply a Gaussian filter to the image
img_smooth = cv2.GaussianBlur(img, (5, 5), 0)

# Compute the gradient magnitude and direction of the smoothed image using the Sobel operator
gradient_x = cv2.Sobel(img_smooth, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(img_smooth, cv2.CV_64F, 0, 1, ksize=3)
gradient_mag = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
gradient_dir = np.arctan2(gradient_y, gradient_x)

# Apply non-maximum suppression to the gradient magnitude
gradient_suppressed = cv2.morphologyEx(
    gradient_mag, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
gradient_thinned = gradient_mag * (gradient_mag == gradient_suppressed)

# Threshold the gradient magnitude to identify strong and weak edges
gradient_thinned = gradient_thinned.astype(np.uint8)
low_threshold, high_threshold = 20, 50
strong_edges = (gradient_thinned > high_threshold).astype(np.uint8)
weak_edges = ((gradient_thinned >= low_threshold) & (
    gradient_thinned <= high_threshold)).astype(np.uint8)

# Apply hysteresis thresholding to link weak edges to strong edges and form a complete edge map
edge_map = cv2.Canny(img_smooth, low_threshold, high_threshold)

# Display the results
cv2.imshow('Input Image', img)
cv2.imshow('Smoothed Image', img_smooth)
cv2.imshow('Gradient Magnitude', gradient_mag)
cv2.imshow('Gradient Direction', gradient_dir)
cv2.imshow('Non-Maximum Suppression', gradient_thinned)
cv2.imshow('Strong Edges', strong_edges * 255)
cv2.imshow('Weak Edges', weak_edges * 255)
cv2.imshow('Edge Map', edge_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
