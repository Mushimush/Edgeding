import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

images = sorted(glob.glob("./retina2/*"))

# Define the number of frames to average
n_frames = 1

# Define a list to store the previous n_frames binary images
binary_list = []

for image_path in images:

    # Load the image as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Threshold the grayscale image to create a binary image
    _, binary = cv2.threshold(img, 115, 255, cv2.THRESH_BINARY)

    # ========================================================================================================
    # # Define the Roberts kernel for edge detection
    # roberts_kernel_x = np.array([[1, 0], [0, -1]])
    # roberts_kernel_y = np.array([[0, 1], [-1, 0]])

    # edges_x = cv2.filter2D(binary, -1, roberts_kernel_x)
    # edges_y = cv2.filter2D(binary, -1, roberts_kernel_y)
    # edges = cv2.bitwise_or(edges_x, edges_y)

    # ========================================================================================================
    # Apply Sobel edge detection to the binary image
    sobel_x = cv2.Sobel(binary, cv2.CV_8U, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(binary, cv2.CV_8U, 0, 1, ksize=3)
    edges = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    # ========================================================================================================
    # # Apply Canny edge detection to the binary image
    # edges = cv2.Canny(binary, 100, 200)

    # ========================================================================================================

    # Invert the binary image to create a mask
    mask = cv2.bitwise_not(binary)

    # Apply the mask to the edges image to highlight the edges that contain white pixels
    highlighted_edges = cv2.bitwise_and(edges, edges, mask=mask)

    # Add the current binary image to the list
    binary_list.append(highlighted_edges)

    # If the list is longer than n_frames, remove the oldest binary image
    if len(binary_list) > n_frames:
        binary_list.pop(0)

    # Average the binary images over the last n_frames
    averaged_binary = sum(binary_list) // len(binary_list)

    # Sharpen the averaged binary image using a Laplacian filter
    laplacian = cv2.Laplacian(averaged_binary, cv2.CV_8U, ksize=3)
    sharpened = cv2.addWeighted(averaged_binary, 1.5, laplacian, -0.5, 0)

    # Replace img with merge on line 84 if wish to view the output w/ mask
    merge = cv2.bitwise_and(sharpened, mask)

    # Use the Probabilistic Hough Line Transform to detect curve lines
    min_line_length = 100
    max_line_gap = 20
    lines = cv2.HoughLinesP(merge, rho=1, theta=np.pi/180, threshold=50,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)

    # Draw the detected lines on the image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(merge, (x1, y1), (x2, y2), (255, 0, 0), 1)

    # Get the file name from the input image path using os.path.basename()
    img_filename = os.path.basename(image_path)

    # Save the image to the output directory
    cv2.imwrite("./sharpened" + "/" + img_filename, merge)

sharp = sorted(glob.glob("./sharpened/*"))

for image2_path, sharp_path in zip(images, sharp):

    # Load the base image
    img1 = cv2.imread(image2_path)

    # Load the second image to be overlaid
    img2 = cv2.imread(sharp_path)

    # Resize the second image to match the size of the first image
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Set the blending factor for the two images
    alpha = 0.4

    # Overlay the two images using cv2.addWeighted()
    overlay = cv2.addWeighted(img1, alpha, img2_resized, 1 - alpha, 0)

    # Get the file name from the input image path using os.path.basename()
    img_filename = os.path.basename(image2_path)

    # Save the image to the output directory
    cv2.imwrite("./output" + "/" + img_filename, overlay)

plt.imshow()
