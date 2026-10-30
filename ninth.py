import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Untitled2.jpg', cv2.IMREAD_GRAYSCALE)

# 9a. Apply FFT on image
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)  # Shift zero frequency to center

# Magnitude spectrum for visualization
magnitude = cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])
magnitude = 20 * np.log(magnitude + 1)  # Log scale for better visibility

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2  # Center coordinates

# 9b. Low pass filter in frequency domain
radius_lp = 30  # Keep frequencies within this radius
mask_lp = np.zeros((rows, cols, 2), np.uint8)
cv2.circle(mask_lp, (ccol, crow), radius_lp, (1, 1), -1)  # Create circular mask
dft_lp = dft_shift * mask_lp  # Apply mask (keep center frequencies)

# 9b. High pass filter in frequency domain
radius_hp = 30
mask_hp = np.ones((rows, cols, 2), np.uint8)
cv2.circle(mask_hp, (ccol, crow), radius_hp, (0, 0), -1)  # Block center frequencies
dft_hp = dft_shift * mask_hp  # Apply mask (keep outer frequencies)

# 9c. Apply IFFT to reconstruct
def inverse_dft(filtered_dft):
    dft_ishift = np.fft.ifftshift(filtered_dft)  # Shift back
    img_back = cv2.idft(dft_ishift)  # Inverse DFT
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])  # Extract magnitude
    return img_back

img_lp = inverse_dft(dft_lp)  # Reconstructed low-pass image
img_hp = inverse_dft(dft_hp)  # Reconstructed high-pass image

# Display
plt.figure(figsize=(12, 8))
plt.subplot(231), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(232), plt.imshow(magnitude, cmap='gray'), plt.title('FFT Magnitude')
plt.subplot(233), plt.imshow(mask_lp[:,:,0], cmap='gray'), plt.title('LP Mask')
plt.subplot(234), plt.imshow(img_lp, cmap='gray'), plt.title('Low Pass (Blurred)')
plt.subplot(235), plt.imshow(mask_hp[:,:,0], cmap='gray'), plt.title('HP Mask')
plt.subplot(236), plt.imshow(img_hp, cmap='gray'), plt.title('High Pass (Edges)')
plt.tight_layout(); plt.show()
