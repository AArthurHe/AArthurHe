# Image Stitching using SURF with Thin Plate Spline (TPS) Fusion

This project provides an implementation of image stitching using the SURF (Speeded Up Robust Features) algorithm for feature detection and matching, combined with Thin Plate Spline (TPS) interpolation for smooth image blending.

## Features

- **SURF Feature Detection**: Uses SURF algorithm to detect robust keypoints and compute descriptors
- **FLANN-based Matching**: Efficient feature matching using FLANN (Fast Library for Approximate Nearest Neighbors)
- **Robust Homography Estimation**: Uses RANSAC to compute reliable homography transformations
- **Thin Plate Spline Blending**: Smooth image fusion using TPS interpolation for natural-looking results

## Installation

### Requirements

- Python 3.7+
- OpenCV with contrib modules (for SURF)
- NumPy
- SciPy

### Install dependencies

```bash
pip install -r requirements.txt
```

**Note**: SURF is a patented algorithm and requires `opencv-contrib-python` package. Make sure to install it:

```bash
pip install opencv-contrib-python
```

## Usage

### Command Line

```bash
python src/surf_tps_stitcher.py image1.jpg image2.jpg -o output.jpg
```

Options:
- `-o, --output`: Output file path (default: `stitched_result.jpg`)
- `--no-tps`: Disable TPS blending (use simple averaging instead)

### Python API

```python
from src.surf_tps_stitcher import SURFImageStitcher, stitch_images
import cv2

# Simple usage with file paths
result = stitch_images('image1.jpg', 'image2.jpg', 'output.jpg')

# Advanced usage with more control
img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')

stitcher = SURFImageStitcher(
    hessian_threshold=400,  # SURF Hessian threshold
    ratio_thresh=0.7,       # Lowe's ratio test threshold
    min_matches=10          # Minimum required matches
)

result = stitcher.stitch(img1, img2, use_tps_blend=True)

if result is not None:
    cv2.imwrite('output.jpg', result)
```

### Thin Plate Spline Interpolation

The TPS class can also be used independently for other interpolation tasks:

```python
from src.surf_tps_stitcher import ThinPlateSpline
import numpy as np

# Define control points and values
control_points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
values = np.array([0.0, 1.0, 1.0, 0.5])

# Create TPS interpolator
tps = ThinPlateSpline(control_points, values, regularization=1e-6)

# Evaluate at new points
new_points = np.array([[0.5, 0.5], [0.25, 0.75]])
interpolated = tps.evaluate(new_points)
```

## Algorithm Overview

### 1. Feature Detection (SURF)
- SURF detects blob-like features using Hessian matrix approximation
- Features are scale and rotation invariant
- Each keypoint is described by a 64-dimensional descriptor

### 2. Feature Matching
- FLANN matcher finds nearest neighbors efficiently
- Lowe's ratio test filters ambiguous matches
- Only confident matches are kept

### 3. Homography Estimation
- RANSAC algorithm robustly estimates the transformation
- Outliers (wrong matches) are automatically rejected
- 3x3 homography matrix maps coordinates between images

### 4. TPS Blending
- Control points are placed along the overlap boundary
- TPS interpolates blend weights smoothly across the overlap
- The radial basis function r²log(r) ensures smooth transitions
- Regularization prevents overfitting

## Technical Details

### Thin Plate Spline

The TPS minimizes the bending energy functional:

E(f) = Σ|f(xᵢ) - yᵢ|² + λ∫∫(fₓₓ² + 2fₓᵧ² + fᵧᵧ²)dxdy

The solution has the form:
f(x) = a₀ + a₁x + a₂y + Σwᵢ U(|x - xᵢ|)

where U(r) = r²log(r) is the TPS radial basis function.

## License

This project is provided for educational purposes. Note that SURF is patented, so commercial use may require a license.

## References

1. Bay, H., Tuytelaars, T., & Van Gool, L. (2006). SURF: Speeded up robust features.
2. Bookstein, F. L. (1989). Principal warps: Thin-plate splines and the decomposition of deformations.
