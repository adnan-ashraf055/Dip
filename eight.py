import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Untitled2.jpg', cv2.IMREAD_GRAYSCALE)

# Low Pass Filters (smoothing / blurring)

# 1. Averaging filter
kernel_avg = np.ones((5, 5), np.float32) / 25
lowpass_avg = cv2.filter2D(img, -1, kernel_avg)

# 2. Gaussian filter
lowpass_gauss = cv2.GaussianBlur(img, (5, 5), 1.5)

# 3. Median filter
lowpass_median = cv2.medianBlur(img, 5)

# 4. Box filter
lowpass_box = cv2.boxFilter(img, -1, (5, 5))  # Normalized box filter

# High Pass Filters (edge enhancement / sharpening)

# 1. Laplacian filter
laplacian = cv2.Laplacian(img, cv2.CV_64F)  # Second derivative edge detector
laplacian = np.uint8(np.absolute(laplacian))  # Convert to absolute values

# 2. Sobel filter (edges in x and y directions)
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in X
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in Y
sobel_comb = cv2.magnitude(sobel_x, sobel_y)  # Combined gradient magnitude

# 3. Custom sharpening kernel
kernel_sharpen = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], np.float32)
sharpened = cv2.filter2D(img, -1, kernel_sharpen)

# Display
titles = ['Original','LP: Average','LP: Gaussian','LP: Median',
          'HP: Laplacian','HP: Sobel X','HP: Sobel Y','HP: Sharpened']
images = [img, lowpass_avg, lowpass_gauss, lowpass_median,
          laplacian, sobel_x, sobel_y, sharpened]
plt.figure(figsize=(14, 8))
for i in range(8):
    plt.subplot(2, 4, i+1), plt.imshow(images[i], cmap='gray')
    plt.title(titles[i]), plt.xticks([]), plt.yticks([])
plt.tight_layout(); plt.show()

