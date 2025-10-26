import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Load Image
# -------------------------
def load_image(image_path):
    """Load an image in grayscale mode."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("Image not found at the specified path.")
    return image


# -------------------------
# Custom SSIM Calculation
# -------------------------
def ssim_custom(img1, img2):
    """Compute Structural Similarity Index (SSIM) between two images."""
    C1 = 6.5025
    C2 = 58.5225

    mu1, mu2 = np.mean(img1), np.mean(img2)
    sigma1_sq, sigma2_sq = np.var(img1), np.var(img2)
    sigma12 = np.cov(img1.flatten(), img2.flatten())[0][1]

    num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denom = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)

    return num / denom


# -------------------------
# Edge Detection Methods
# -------------------------
def canny_edge(image, low_threshold, high_threshold):
    """Canny Edge Detector."""
    return cv2.Canny(image, low_threshold, high_threshold)


def sobel_edge(image):
    """Sobel Edge Detector."""
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.magnitude(sobel_x, sobel_y).astype(np.uint8)


def prewitt_edge(image):
    """Prewitt Edge Detector."""
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    
    prewitt_x = cv2.filter2D(image, -1, kernel_x)
    prewitt_y = cv2.filter2D(image, -1, kernel_y)

    return cv2.magnitude(prewitt_x.astype(np.float64), prewitt_y.astype(np.float64)).astype(np.uint8)


def roberts_edge(image):
    """Roberts Cross Edge Detector."""
    kernel_x = np.array([[1, 0], [0, -1]])
    kernel_y = np.array([[0, 1], [-1, 0]])
    
    roberts_x = cv2.filter2D(image, -1, kernel_x)
    roberts_y = cv2.filter2D(image, -1, kernel_y)

    return cv2.magnitude(roberts_x.astype(np.float64), roberts_y.astype(np.float64)).astype(np.uint8)


def laplacian_edge(image):
    """Laplacian Edge Detector."""
    return cv2.Laplacian(image, cv2.CV_64F).astype(np.uint8)


def scharr_edge(image):
    """Scharr Edge Detector (more accurate than Sobel)."""
    scharr_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
    return cv2.magnitude(scharr_x, scharr_y).astype(np.uint8)


def kirsch_edge(image):
    """Kirsch Compass Edge Detector."""
    kernels = [
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, 5]]),
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
    ]
    return np.max([cv2.filter2D(image, -1, k) for k in kernels], axis=0)


def difference_of_gaussians(image):
    """Difference of Gaussians (DoG) Edge Detector."""
    blur1 = cv2.GaussianBlur(image, (3, 3), 0)
    blur2 = cv2.GaussianBlur(image, (5, 5), 0)
    return cv2.subtract(blur1, blur2)


def basic_gradient_edge(image):
    """Custom-built Gradient Edge Detector with Non-Maximum Suppression."""
    blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)  
    
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  

    grad_x = cv2.filter2D(blurred_image, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(blurred_image, cv2.CV_64F, kernel_y)

    magnitude = cv2.magnitude(grad_x, grad_y)

    # Apply non-maximum suppression for edge thinning
    non_max_suppressed = non_maximum_suppression(magnitude, grad_x, grad_y)

    # Thresholding edges
    thresholded_edges = cv2.threshold(non_max_suppressed, 50, 255, cv2.THRESH_BINARY)[1]  

    return np.uint8(np.clip(thresholded_edges, 0, 255))


def non_maximum_suppression(magnitude, grad_x, grad_y):
    """Non-Maximum Suppression for edge thinning."""
    rows, cols = magnitude.shape
    result = np.zeros_like(magnitude)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            angle = np.arctan2(grad_y[i, j], grad_x[i, j]) * 180 / np.pi
            angle = (angle + 180) % 180  

            # Determine neighbors based on gradient direction
            if (0 <= angle < 22.5) or (157.5 <= angle < 180): 
                neighbor1, neighbor2 = magnitude[i, j + 1], magnitude[i, j - 1]
            elif (22.5 <= angle < 67.5): 
                neighbor1, neighbor2 = magnitude[i + 1, j - 1], magnitude[i - 1, j + 1]
            elif (67.5 <= angle < 112.5): 
                neighbor1, neighbor2 = magnitude[i + 1, j], magnitude[i - 1, j]
            else:  
                neighbor1, neighbor2 = magnitude[i + 1, j + 1], magnitude[i - 1, j - 1]

            # Suppress non-maximum pixels
            result[i, j] = magnitude[i, j] if magnitude[i, j] >= max(neighbor1, neighbor2) else 0

    return result


# -------------------------
# Compare Methods
# -------------------------
def compare_methods(image, ground_truth):
    """Run multiple edge detection methods and compare with ground truth using SSIM."""
    methods = {
        "Canny": lambda img: canny_edge(img, 50, 150),
        "Sobel": sobel_edge,
        "Prewitt": prewitt_edge,
        "Roberts": roberts_edge,
        "Laplacian": laplacian_edge,
        "Scharr": scharr_edge,
        "Kirsch": kirsch_edge,
        "DoG": difference_of_gaussians,
        "Custom Built Model": basic_gradient_edge  
    }
    
    scores, results = {}, {}

    for name, method in methods.items():
        edges = method(image)
        similarity = ssim_custom(ground_truth, edges)
        scores[name] = similarity
        results[name] = edges

    return scores, results


def resize_image(image, size=(300, 300)):
    """Resize image to specified size."""
    return cv2.resize(image, size)


# -------------------------
# Main Execution
# -------------------------
image_path = r"C:\Users\aarus\Downloads\test1.png"
ground_truth_path = r"C:\Users\aarus\Downloads\ground.png" 

image = load_image(image_path)
ground_truth = load_image(ground_truth_path)

scores, results = compare_methods(image, ground_truth)

# Display results
plt.figure(figsize=(15, 10))
for i, (name, edges) in enumerate(results.items()):
    resized_edges = resize_image(edges, size=(300, 300))
    
    plt.subplot(3, 3, i + 1)
    plt.title(f"{name} (SSIM: {scores[name]:.4f})")
    plt.imshow(resized_edges, cmap='gray')
    plt.axis('off')

plt.tight_layout(pad=3.0)  
plt.show()

# SSIM ranges from -1 to 1:
# - 1 → perfect similarity
# - -1 → complete dissimilarity
print("Comparison scores (SSIM):")
for method, score in scores.items():
    print(f"{method}: {score:.4f}")
