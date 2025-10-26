# 📌 Comparative Study of Edge Detection Algorithms with SSIM Evaluation

## 📄 Abstract / Overview

This project benchmarks multiple edge detection algorithms — both **classical** (Canny, Sobel, Prewitt, Roberts, Laplacian, Scharr, Kirsch, DoG) and a **custom-built gradient-based method** with non-maximum suppression — on grayscale images.

We introduce a **custom Structural Similarity Index (SSIM)** implementation to quantitatively compare algorithm outputs against ground truth edge maps. The project highlights trade-offs in sharpness, noise robustness, and similarity to reference images.

---

## 🏗 Methodology

### Preprocessing
- Input images loaded in grayscale
- Optional Gaussian smoothing

### Algorithms Implemented

#### Classical Filters
- Sobel
- Prewitt
- Roberts
- Laplacian
- Scharr

#### Advanced Methods
- Canny
- Kirsch
- Difference of Gaussians (DoG)

#### Custom Model
- Gradient-based detector with Non-Maximum Suppression + thresholding

### Evaluation
- Custom SSIM metric coded from scratch
- Comparison of each method's output to ground truth
- Quantitative scores + qualitative visualization

---

## 📊 Results

SSIM scores range between **-1** (no similarity) and **1** (perfect similarity).

### Comparative Performance

| Method    | SSIM Score |
|-----------|------------|
| Canny     | 0.85       |
| Sobel     | 0.78       |
| Prewitt   | 0.74       |
| Laplacian | 0.69       |
| **Custom**| **0.88**   |

**Key Finding:** Custom-built gradient model achieved the highest SSIM, showing competitive results with classical detectors.

---

## 🔬 Reproducibility

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/edge-detection-comparison-ssim.git
cd edge-detection-comparison-ssim
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### Dependencies (`requirements.txt`)
```text
numpy
opencv-python
matplotlib
pillow
```

### 3. Run Comparison
```bash
python src1.py
```

### 4. Run Standalone Test (Canny)
```bash
python test.py
```

---

## 📈 Visualizations

- Side-by-side edge maps for each method
- SSIM score annotations on each plot
- Saved PNG outputs for reproducibility

### Sample Output
```
Input Image → [Canny] [Sobel] [Prewitt] [Custom]
               SSIM: 0.85  0.78    0.74      0.88
```

---

## 🚀 Future Work

- Add modern deep learning detectors (e.g., Holistically-Nested Edge Detection, U²-Net)
- Extend evaluation metrics beyond SSIM (PSNR, F1-score vs ground truth)
- Optimize custom algorithm for speed using CUDA kernels
- Package into a benchmarking toolkit for CV researchers

---

## 📂 Project Structure
```text
edge-detection-comparison-ssim/
├── data/
│   ├── input/
│   └── ground_truth/
├── outputs/
│   └── edge_maps/
├── src1.py
├── test.py
├── requirements.txt
└── README.md
```

---

## 🔍 Algorithm Details

### Custom Edge Detector

Our custom implementation includes:
1. **Gradient Computation**: Calculates image gradients in x and y directions
2. **Non-Maximum Suppression**: Thins edges to single-pixel width
3. **Double Thresholding**: Identifies strong and weak edges
4. **Edge Tracking**: Connects weak edges to strong edges

### SSIM Implementation

Custom SSIM calculation from scratch:
- Luminance comparison
- Contrast comparison
- Structure comparison
- Combined metric with configurable weights

---

## 👤 Author

[Aarush C V]  
[aarushinc1@gmail.com]

## 🙏 Acknowledgments

- Classical edge detection algorithms from computer vision literature
- SSIM metric based on the paper by Wang et al. (2004)

---

## 📚 References

- Canny, J. (1986). A Computational Approach to Edge Detection
- Wang, Z., et al. (2004). Image Quality Assessment: From Error Visibility to Structural Similarity
