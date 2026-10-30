import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('flower.jpg', cv2.IMREAD_GRAYSCALE)  # Read as grayscale
h, w = img.shape  # Get dimensions (height, width)

# 2a. Negative image
negative = 255 - img  # Subtract each pixel from 255

# 2b. Flip image
flip_h = cv2.flip(img, 1)  # 1 = horizontal flip
flip_v = cv2.flip(img, 0)  # 0 = vertical flip
flip_both = cv2.flip(img, -1)  # -1 = both horizontal and vertical

# 2c. Thresholding
_, thresh_binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
_, thresh_inv = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
# Adaptive thresholding
adaptive = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 11, 2)

# 2d. Contrast stretching
min_val = np.min(img)  # Minimum intensity in image
max_val = np.max(img)  # Maximum intensity in image
stretched = ((img - min_val) / (max_val - min_val)) * 255  # Linear stretch to [0,255]
stretched = stretched.astype(np.uint8)  # Convert back to uint8

# Display results
titles = ['Original', 'Negative', 'Flip H', 'Flip V', 'Binary', 'Adaptive', 'Contrast Stretch']
images = [img, negative, flip_h, flip_v, thresh_binary, adaptive, stretched]
for i in range(len(images)):
    plt.subplot(3, 3, i+1), plt.imshow(images[i], cmap='gray')
    plt.title(titles[i]), plt.xticks([]), plt.yticks([])
plt.show()
