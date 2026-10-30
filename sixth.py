import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Untitled2.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]

# 6a. Translation (shift image)
tx, ty = 50, 30  # Shift right by 50, down by 30
M_trans = np.float32([[1, 0, tx], [0, 1, ty]])  # 2x3 affine transformation matrix
translated = cv2.warpAffine(img, M_trans, (w, h))

# 6b. Scaling (change size)
scaled = cv2.resize(img_rgb, None, fx=1.5, fy=1.5)  # 1.5x enlarged

# 6c. Rotation
center = (w // 2, h // 2)  # Center of image
angle = 45  # Degrees
M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)  # Rotation matrix
rotated = cv2.warpAffine(img, M_rot, (w, h))

# 6d. Shrinking (reduce size)
shrunk = cv2.resize(img_rgb, None, fx=0.3, fy=0.3)  # 30% of original

# 6e. Zooming (enlarge)
zoomed = cv2.resize(img_rgb, None, fx=2.5, fy=2.5)  # 2.5x enlarged
# Better zoom with interpolation
zoomed_inter = cv2.resize(img_rgb, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

# Display
plt.figure(figsize=(12, 8))
plt.subplot(231), plt.imshow(img_rgb), plt.title('Original')
plt.subplot(232), plt.imshow(cv2.cvtColor(translated,cv2.COLOR_BGR2RGB)), plt.title('Translated')
plt.subplot(233), plt.imshow(rotated), plt.title('Rotated 45')
plt.subplot(234), plt.imshow(scaled), plt.title('Scaled 1.5x')
plt.subplot(235), plt.imshow(shrunk), plt.title('Shrunk 0.3x')
plt.subplot(236), plt.imshow(zoomed), plt.title('Zoomed 2.5x')
plt.show()

