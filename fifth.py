import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('flower.jpg', cv2.IMREAD_GRAYSCALE)

# 5a. Histogram using standard function
hist_cv = cv2.calcHist([img], [0], None, [256], [0, 256])

# Equalization using standard function
eq_cv = cv2.equalizeHist(img)

# 5b. Histogram without using standard functions
def manual_histogram(image):
    h, w = image.shape
    hist = np.zeros(256, dtype=np.int32)  # 256 bins initialized to 0
    for i in range(h):
        for j in range(w):
            pixel_val = image[i, j]
            hist[pixel_val] += 1  # Count occurrences of each pixel value
    return hist

hist_manual = manual_histogram(img)

# Manual equalization
def manual_equalization(image):
    h, w = image.shape
    total_pixels = h * w
    hist = manual_histogram(image)
    # Compute cumulative distribution function (CDF)
    cdf = np.zeros(256, dtype=np.float64)
    cdf[0] = hist[0] / total_pixels  # Normalized cumulative sum
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + hist[i] / total_pixels
    # Map each pixel: new_val = cdf[old_val] * 255
    equalized = np.zeros_like(image)
    for i in range(h):
        for j in range(w):
            equalized[i, j] = int(cdf[image[i, j]] * 255)
    return equalized.astype(np.uint8)

eq_manual = manual_equalization(img)

# Plot histograms
plt.figure(figsize=(12, 8))
plt.subplot(221), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(222), plt.plot(hist_cv, color='black'), plt.title('Histogram (OpenCV)')
plt.subplot(223), plt.bar(range(256), hist_manual, color='gray'), plt.title('Histogram (Manual)')
plt.subplot(224), plt.imshow(eq_cv, cmap='gray'), plt.title('Equalized')
plt.tight_layout(); plt.show()

# Compare manual vs OpenCV equalization
plt.figure(figsize=(10, 4))
plt.subplot(121), plt.imshow(eq_cv, cmap='gray'), plt.title('OpenCV Equalization')
plt.subplot(122), plt.imshow(eq_manual, cmap='gray'), plt.title('Manual Equalization')
plt.show()
