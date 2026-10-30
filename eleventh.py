import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Untitled.jpg', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)  # Need binary input

# Define structuring element (kernel)
kernel = np.ones((5, 5), np.uint8)  # 5x5 square kernel
kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))  # Cross-shaped
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Elliptical

# Erosion
erosion = cv2.erode(binary, kernel, iterations=1)
# Pixel set to 1 only if ALL neighbors under kernel are 1
# Shrinks foreground objects, removes small white noise

# Dilation
dilation = cv2.dilate(binary, kernel, iterations=1)
# Pixel set to 1 if ANY neighbor under kernel is 1
# Expands foreground objects, fills holes

# Additional morphological operations
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
# Erosion followed by dilation: removes small objects

closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
# Dilation followed by erosion: fills small holes

gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
# Dilation - Erosion: extracts object boundaries

tophat = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel)
# Original - Opening: extracts small bright details

blackhat = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel)
# Closing - Original: extracts small dark details

# Display
titles = ['Binary','Erosion','Dilation','Opening','Closing','Gradient','Top Hat','Black Hat']
images = [binary, erosion, dilation, opening, closing, gradient, tophat, blackhat]
plt.figure(figsize=(14, 8))
for i in range(8):
    plt.subplot(2, 4, i+1), plt.imshow(images[i], cmap='gray')
    plt.title(titles[i]), plt.xticks([]), plt.yticks([])
plt.tight_layout(); plt.show()

# Show effect of different kernels (erosion with square vs cross vs ellipse)
erosion_sq = cv2.erode(binary, kernel, iterations=1)
erosion_cross = cv2.erode(binary, kernel_cross, iterations=1)
erosion_ell = cv2.erode(binary, kernel_ellipse, iterations=1)

plt.figure(figsize=(10, 4))
plt.subplot(131), plt.imshow(erosion_sq, cmap='gray'), plt.title('Square Kernel')
plt.subplot(132), plt.imshow(erosion_cross, cmap='gray'), plt.title('Cross Kernel')
plt.subplot(133), plt.imshow(erosion_ell, cmap='gray'), plt.title('Ellipse Kernel')
plt.show()
