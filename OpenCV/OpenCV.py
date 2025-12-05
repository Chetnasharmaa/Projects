import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to display images in a grid
def display_images(images, titles, rows, cols, figsize=(15, 10)):#images- list of image to display, titles: list of titles for each image, rows, cols-in grid,figsixe:width and height
    """Display multiple images in a grid layout"""
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if rows * cols > 1 else [axes]#easier to iterate for us, suppose 100 elements(axes), it becomes 50
    
    for idx, (img, title) in enumerate(zip(images, titles)):
        if len(img.shape) == 2:  # Grayscale
            axes[idx].imshow(img, cmap='gray')#imshow shows image
        else:  # Color
            axes[idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[idx].set_title(title)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

# Load images
print("Loading images...")
color_img = cv2.imread('color.png')
lion_img = cv2.imread('lion.jpg')

# Check if images are loaded
if color_img is None or lion_img is None:
    print("Error: Could not load images. Please check the file paths.")
    exit()

print(f"Color image shape: {color_img.shape}")#shapes of both images
print(f"Lion image shape: {lion_img.shape}")

# ===== 1. COLOR CONVERSIONS =====
print("\n1. Performing color conversions...")

# Convert to Grayscale
gray_color = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)#function that conversts BGR to Gray
gray_lion = cv2.cvtColor(lion_img, cv2.COLOR_BGR2GRAY)

# Convert to  # a format (hue, saturation & value)
hsv_color = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
hsv_lion = cv2.cvtColor(lion_img, cv2.COLOR_BGR2HSV)

# Display color conversions
display_images(
    [color_img, gray_color, hsv_color],
    ['Original Color', 'Grayscale', 'HSV'],
    1, 3 # 1 row, 3 images in 3 columns
)

display_images(
    [lion_img, gray_lion, hsv_lion],
    ['Original Lion', 'Grayscale Lion', 'HSV Lion'],
    1, 3
)

# ===== 2. EDGE DETECTION =====
print("\n2. Performing edge detection...")

# Canny Edge Detection
edges_color = cv2.Canny(gray_color, 100, 200)
edges_lion = cv2.Canny(gray_lion, 100, 200)

# Sobel Edge Detection
sobelx = cv2.Sobel(gray_color, cv2.CV_64F, 1, 0, ksize=5)#1,0 is computing the gradient in x direction
sobely = cv2.Sobel(gray_color, cv2.CV_64F, 0, 1, ksize=5)
sobel_combined = np.sqrt(sobelx**2 + sobely**2)
sobel_combined = np.uint8(sobel_combined)

# Display edge detection
display_images(
    [gray_color, edges_color, sobel_combined],
    ['Grayscale', 'Canny Edges', 'Sobel Edges'],
    1, 3
)

display_images(
    [gray_lion, edges_lion],
    ['Grayscale Lion', 'Canny Edges Lion'],
    1, 2
)

# ===== 3. EROSION AND DILATION =====
print("\n3. Performing erosion and dilation...")

# Define kernel for morphological operations, larger kernel = stronger 
kernel = np.ones((5, 5), np.uint8)#5,5 kernel length nd width

# Erosion
eroded_color = cv2.erode(color_img, kernel, iterations=1)
eroded_lion = cv2.erode(lion_img, kernel, iterations=1)

# Dilation
dilated_color = cv2.dilate(color_img, kernel, iterations=1)
dilated_lion = cv2.dilate(lion_img, kernel, iterations=1)

# Opening (Erosion followed by Dilation)
opening_color = cv2.morphologyEx(color_img, cv2.MORPH_OPEN, kernel)

# Closing (Dilation followed by Erosion)
closing_color = cv2.morphologyEx(color_img, cv2.MORPH_CLOSE, kernel)

# Display erosion and dilation
display_images(
    [color_img, eroded_color, dilated_color, opening_color, closing_color],
    ['Original', 'Eroded', 'Dilated', 'Opening', 'Closing'],
    2, 3, figsize=(15, 10)
)

display_images(
    [lion_img, eroded_lion, dilated_lion],
    ['Original Lion', 'Eroded Lion', 'Dilated Lion'],
    1, 3
)

# ===== 4. IMAGE DENOISING =====
print("\n4. Performing image denoising...")

# Add some noise to demonstrate denoising
def add_noise(img):
    """Add Gaussian noise to image"""
    row, col, ch = img.shape
    mean = 0
    sigma = 25
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = np.clip(img + gauss, 0, 255).astype(np.uint8)
    return noisy

# Create noisy images
noisy_color = add_noise(color_img)
noisy_lion = add_noise(lion_img)

# Apply different denoising techniques
# 1. Gaussian Blur
gaussian_color = cv2.GaussianBlur(noisy_color, (5, 5), 0)

# 2. Median Blur
median_color = cv2.medianBlur(noisy_color, 5)

# 3. Bilateral Filter (preserves edges)
bilateral_color = cv2.bilateralFilter(noisy_color, 9, 75, 75)

# 4. Non-local Means Denoising
nlm_color = cv2.fastNlMeansDenoisingColored(noisy_color, None, 10, 10, 7, 21)
nlm_lion = cv2.fastNlMeansDenoisingColored(noisy_lion, None, 10, 10, 7, 21)

# Display denoising results
display_images(
    [color_img, noisy_color, gaussian_color, median_color, bilateral_color, nlm_color],
    ['Original', 'Noisy', 'Gaussian Blur', 'Median Blur', 'Bilateral Filter', 'NLM Denoising'],
    2, 3, figsize=(15, 10)
)

display_images(
    [lion_img, noisy_lion, nlm_lion],
    ['Original Lion', 'Noisy Lion', 'Denoised Lion'],
    1, 3
)

# ===== 5. SAVE PROCESSED IMAGES =====
print("\n5. Saving processed images...")

# Save some key results
cv2.imwrite('gray_color.jpg', gray_color)
cv2.imwrite('hsv_color.jpg', hsv_color)
cv2.imwrite('edges_color.jpg', edges_color)
cv2.imwrite('eroded_color.jpg', eroded_color)
cv2.imwrite('dilated_color.jpg', dilated_color)
cv2.imwrite('denoised_color.jpg', nlm_color)

cv2.imwrite('gray_lion.jpg', gray_lion)
cv2.imwrite('edges_lion.jpg', edges_lion)
cv2.imwrite('denoised_lion.jpg', nlm_lion)

print("\nAll operations completed successfully!")
print("Processed images have been saved.")

# ===== BONUS: Image Statistics =====
print("\n6. Image Statistics:")
print(f"Original Color Image - Mean: {np.mean(color_img):.2f}, Std: {np.std(color_img):.2f}")
print(f"Grayscale Image - Mean: {np.mean(gray_color):.2f}, Std: {np.std(gray_color):.2f}")
print(f"Edge Image - Non-zero pixels: {np.count_nonzero(edges_color)}")