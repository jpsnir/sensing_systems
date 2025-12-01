import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

# 1. Load the image
image_path = 'zidane.jpg'  # <--- REPLACE THIS with your image filename
image = cv2.imread(image_path)
image = cv2.resize(image, (640, 480))

if image is None:
    print(f"Error: Could not load image from {image_path}")
    exit()

# 2. Pre-processing
# HOG works on Grayscale images (color is usually irrelevant for shape)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Optional: Resize if the image is huge to speed up calculation
gray_image = cv2.resize(gray_image, (640, 480)) 

# 3. Compute HOG features and the visualization
# pixels_per_cell: Size of the "window" to capture gradients (8x8 is standard)
# cells_per_block: Normalizes lighting changes
features, hog_image = hog(
    gray_image,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    visualize=True,
    channel_axis=None 
)

# 4. Enhance the "ghost" image for better viewing
# This scales the intensity so we can see the lines clearly
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# 5. Display Original vs. HOG
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax1.set_title('Original Image')

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('What the Computer "Sees" (HOG)')

plt.show()