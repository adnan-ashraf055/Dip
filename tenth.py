import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('car.jpg', cv2.IMREAD_GRAYSCALE)

# 1. Prewitt operator
kernel_prewitt_x = np.array([[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]], np.float32)  # Horizontal edges
kernel_prewitt_y = np.array([[-1, -1, -1],
                             [ 0,  0,  0],
                             [ 1,  1,  1]], np.float32)  # Vertical edges
prewitt_x = cv2.filter2D(img, -1, kernel_prewitt_x)
prewitt_y = cv2.filter2D(img, -1, kernel_prewitt_y)
prewitt = cv2.magnitude(np.float32(prewitt_x), np.float32(prewitt_y))

# 2. Sobel operator
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobel_x, sobel_y)  # Combined magnitude

# 3. Laplacian operator
laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))

# 4. Canny edge detector
canny = cv2.Canny(img, 100, 200)  # Lower threshold=100, upper=200

# 5. Robert's Cross operator
kernel_roberts_x = np.array([[1, 0], [0, -1]], np.float32)
kernel_roberts_y = np.array([[0, 1], [-1, 0]], np.float32)
roberts_x = cv2.filter2D(img, -1, kernel_roberts_x)
roberts_y = cv2.filter2D(img, -1, kernel_roberts_y)
roberts = cv2.magnitude(np.float32(roberts_x), np.float32(roberts_y))

# 6. Scharr operator
scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)  # More accurate than Sobel
scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)
scharr = cv2.magnitude(scharr_x, scharr_y)

# Display
titles = ["Original","Prewitt","Sobel","Laplacian","Canny","Robert's","Scharr"]
images = [img, prewitt, sobel, laplacian, canny, roberts, scharr]
plt.figure(figsize=(14, 8))
for i in range(7):
    plt.subplot(2, 4, i+1), plt.imshow(images[i], cmap='gray')
    plt.title(titles[i]), plt.xticks([]), plt.yticks([])
plt.tight_layout(); plt.show()
