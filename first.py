
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1b. Read and display image
img = cv2.imread('car.jpg')

# Check whether image is loaded
if img is None:
    print("Error: car.jpg not found. Check the file location.")
    exit()

print("Original image shape:", img.shape)

# Display original image without changing its actual data
height, width = img.shape[:2]

# Fit image inside a maximum 900x600 display area
scale = min(900 / width, 600 / height, 1.0)

display_width = int(width * scale)
display_height = int(height * scale)

original_display = cv2.resize(
    img, (display_width, display_height)
)

cv2.imshow('Original Image', original_display)


# 1c. Resize image
resized = cv2.resize(img, (300, 200))
cv2.imshow('Resized Image (300x200)', resized)

# Alternative: resize to half size
resized_factor = cv2.resize(img, None, fx=0.5, fy=0.5)
cv2.imshow('Half Size Image', resized_factor)


# 1d. Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale Image', gray)


# 1e. Convert to black and white (binary)
_, bw = cv2.threshold(
    gray, 127, 255, cv2.THRESH_BINARY
)
cv2.imshow('Black and White Image', bw)


# 1f. Draw image intensity profile
row_index = min(100, gray.shape[0] - 1)

profile = gray[row_index, :]

plt.figure()
plt.plot(profile)
plt.title('Image Profile (Row {})'.format(row_index))
plt.xlabel('Pixel Position')
plt.ylabel('Intensity Value')
plt.grid()
plt.show()


# 1g. Separate BGR planes
b, g, r = cv2.split(img)

cv2.imshow('Blue Plane', b)
cv2.imshow('Green Plane', g)
cv2.imshow('Red Plane', r)


# 1h. Create color images from separate planes
zeros = np.zeros_like(b)

red_image = cv2.merge([zeros, zeros, r])
green_image = cv2.merge([zeros, g, zeros])
blue_image = cv2.merge([b, zeros, zeros])

combined = cv2.merge([b, g, r])

cv2.imshow('Red Image', red_image)
cv2.imshow('Green Image', green_image)
cv2.imshow('Blue Image', blue_image)
cv2.imshow('Combined Image', combined)


# 1i. Write 2D data to an image file
data = np.array([
    [100, 150, 200],
    [50, 255, 128],
    [75, 180, 220]
], dtype=np.uint8)

success = cv2.imwrite('output_data.png', data)

if success:
    print("Image saved successfully as output_data.png")
else:
    print("Error: Image could not be saved.")


# Wait for a key press and close all OpenCV windows
cv2.waitKey(0)
cv2.destroyAllWindows()
