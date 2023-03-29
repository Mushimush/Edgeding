from PIL import Image
import os
import numpy as np

# Define the window size for the moving average filter
window_size = 5

# Load all the input images
input_folder = "retina2"
output_folder = "retina2movingavg"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all the images in the input folder
for filename in os.listdir(input_folder):
    # Load the input image
    img = Image.open(os.path.join(input_folder, filename))

    # Convert the image to grayscale
    img_gray = img.convert("L")

    # Get the image size
    width, height = img_gray.size

    # Convert the image to a numpy array
    img_arr = np.array(img_gray)

    # Create an empty output array
    output_arr = np.zeros((height, width))

    # Loop through all the pixels in the image
    for i in range(height):
        for j in range(width):
            # Get the values for the current window
            window_values = []
            for k in range(-window_size//2, window_size//2+1):
                for l in range(-window_size//2, window_size//2+1):
                    # Handle edge cases by using the closest valid pixel
                    row = min(max(i+k, 0), height-1)
                    col = min(max(j+l, 0), width-1)
                    window_values.append(img_arr[row, col])

            # Compute the average value for the window
            avg_value = sum(window_values) / len(window_values)

            # Set the output pixel value to the computed average
            output_arr[i, j] = avg_value

    # Convert the output array to a PIL image
    output_img = Image.fromarray(output_arr.astype(np.uint8))

    # Save the output image to the output folder
    output_filename = os.path.join(output_folder, filename)
    output_img.save(output_filename)

    print(f"Processed image {filename}")
