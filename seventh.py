import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Untitled2.jpg', cv2.IMREAD_GRAYSCALE)

# 7a. Image restoration using averaging filter
kernel_avg = np.ones((5, 5), dtype=np.float32) / 25  # 5x5 averaging kernel
restored_avg = cv2.filter2D(img, -1, kernel_avg)  # Apply kernel via convolution

# 7b. Add salt & pepper noise + remove it
def add_salt_pepper(image, amount=0.05):
    noisy = image.copy()
    h, w = noisy.shape
    num_pixels = int(h * w * amount)
    # Salt (white) pixels
    coords = [np.random.randint(0, h-1, num_pixels//2),
              np.random.randint(0, w-1, num_pixels//2)]
    noisy[coords[0], coords[1]] = 255
    # Pepper (black) pixels
    coords = [np.random.randint(0, h-1, num_pixels//2),
              np.random.randint(0, w-1, num_pixels//2)]
    noisy[coords[0], coords[1]] = 0
    return noisy

sp_noisy = add_salt_pepper(img, 0.05)

# Remove salt & pepper with median filter
sp_restored = cv2.medianBlur(sp_noisy, 5)  # 5x5 median filter

# 7c. Add Gaussian noise + minimize
def add_gaussian(image, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, image.shape).astype(np.int16)
    noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy

gauss_noisy = add_gaussian(img, 0, 25)

# Minimize Gaussian noise with Gaussian blur
gauss_restored = cv2.GaussianBlur(gauss_noisy, (5, 5), 1.5)  # 5x5 kernel, sigma=1.5

# 7d. Median filter (also works for Gaussian, but best for salt & pepper)
median_filtered = cv2.medianBlur(gauss_noisy, 5)

# Display
titles = ['Original','Avg Filter (Restored)','Salt & Pepper','Median (SP Restored)',
          'Gaussian Noise','Gaussian Blur (G Restored)','Median on Gaussian']
images = [img, restored_avg, sp_noisy, sp_restored, gauss_noisy, gauss_restored, median_filtered]
plt.figure(figsize=(12, 8))
for i in range(7):
    plt.subplot(2, 4, i+1), plt.imshow(images[i], cmap='gray')
    plt.title(titles[i]), plt.xticks([]), plt.yticks([])
plt.tight_layout(); plt.show()

