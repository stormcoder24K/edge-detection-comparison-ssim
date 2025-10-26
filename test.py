import cv2
from matplotlib import pyplot as plt
from PIL import Image

# Read the image in grayscale
image = cv2.imread(r"C:\Users\aarus\Downloads\test1.png", cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur to smooth the image
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Perform Canny edge detection
edges = cv2.Canny(blurred_image, threshold1=100, threshold2=200)

# Save the edge-detected image as PNG (lossless)
cv2.imwrite('edges_output.png', edges)  # PNG format by default is lossless

# Alternatively, use PIL to save the image with no compression (quality = 100)
pil_image = Image.fromarray(edges)
pil_image.save('edges_output.png', format='PNG', compress_level=0)  # No compression for PNG

# Display the edge-detected image
plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detection')
plt.axis('off')
plt.show()

