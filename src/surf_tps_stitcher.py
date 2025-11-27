"""
Image Stitching using SURF (Speeded Up Robust Features) with Thin Plate Spline (TPS) Fusion

This module provides functionality for stitching two images together using:
1. SURF feature detection and description
2. Feature matching using FLANN-based matcher
3. Homography estimation using RANSAC
4. Thin Plate Spline (TPS) based image fusion for smooth blending

Author: AArthurHe
"""

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import cdist


class ThinPlateSpline:
    """
    Thin Plate Spline (TPS) interpolation for smooth image blending.
    
    TPS is a spline-based technique for smooth interpolation that minimizes
    the bending energy of the surface, providing natural-looking blends.
    """
    
    def __init__(self, control_points, target_values, regularization=0.0):
        """
        Initialize the Thin Plate Spline interpolator.
        
        Args:
            control_points: ndarray of shape (n, 2), control point coordinates
            target_values: ndarray of shape (n,) or (n, k), values at control points
            regularization: float, regularization parameter (lambda)
        """
        self.control_points = np.array(control_points)
        self.target_values = np.array(target_values)
        self.regularization = regularization
        self._compute_weights()
    
    def _tps_kernel(self, r):
        """
        Compute the TPS radial basis function: U(r) = r^2 * log(r)
        
        Args:
            r: distances
            
        Returns:
            TPS kernel values
        """
        # Handle r = 0 case (log(0) is undefined)
        result = np.zeros_like(r)
        mask = r > 0
        result[mask] = r[mask] ** 2 * np.log(r[mask])
        return result
    
    def _compute_weights(self):
        """Compute the TPS weights using the control points and target values."""
        n = len(self.control_points)
        
        # Compute pairwise distances
        distances = cdist(self.control_points, self.control_points, 'euclidean')
        
        # Build the K matrix using TPS kernel
        K = self._tps_kernel(distances)
        
        # Add regularization
        K += self.regularization * np.eye(n)
        
        # Build the P matrix (for affine part)
        P = np.hstack([np.ones((n, 1)), self.control_points])
        
        # Build the full system matrix
        L = np.zeros((n + 3, n + 3))
        L[:n, :n] = K
        L[:n, n:] = P
        L[n:, :n] = P.T
        
        # Build the right-hand side
        if self.target_values.ndim == 1:
            b = np.zeros(n + 3)
            b[:n] = self.target_values
        else:
            b = np.zeros((n + 3, self.target_values.shape[1]))
            b[:n] = self.target_values
        
        # Solve the linear system
        try:
            params = np.linalg.solve(L, b)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            params = np.linalg.lstsq(L, b, rcond=None)[0]
        
        self.weights = params[:n]
        self.affine_params = params[n:]
    
    def evaluate(self, points):
        """
        Evaluate the TPS at given points.
        
        Args:
            points: ndarray of shape (m, 2), points to evaluate
            
        Returns:
            Interpolated values at the given points
        """
        points = np.array(points)
        
        # Compute distances to control points
        distances = cdist(points, self.control_points, 'euclidean')
        
        # Apply TPS kernel
        K = self._tps_kernel(distances)
        
        # Compute the TPS part
        tps_part = K @ self.weights
        
        # Compute the affine part
        P = np.hstack([np.ones((len(points), 1)), points])
        affine_part = P @ self.affine_params
        
        return tps_part + affine_part


class SURFImageStitcher:
    """
    Image stitcher using SURF/SIFT features and TPS-based blending.
    
    This class provides methods to stitch two images together using:
    - SURF (or SIFT as fallback) for feature detection and description
    - FLANN-based matching for finding correspondences
    - RANSAC for robust homography estimation
    - Thin Plate Spline for smooth image blending
    """
    
    def __init__(self, hessian_threshold=400, ratio_thresh=0.7, min_matches=10, 
                 use_sift_fallback=True):
        """
        Initialize the SURF Image Stitcher.
        
        Args:
            hessian_threshold: Hessian threshold for SURF detector
            ratio_thresh: Lowe's ratio test threshold
            min_matches: Minimum number of matches required
            use_sift_fallback: If True, use SIFT when SURF is not available
        """
        self.hessian_threshold = hessian_threshold
        self.ratio_thresh = ratio_thresh
        self.min_matches = min_matches
        self.detector_type = None
        
        # Initialize SURF detector (or SIFT as fallback)
        # Note: SURF is patented and may require special OpenCV build
        try:
            self.detector = cv2.xfeatures2d.SURF_create(hessianThreshold=hessian_threshold)
            self.detector_type = "SURF"
            print("Using SURF detector")
        except (AttributeError, cv2.error):
            if use_sift_fallback:
                # SIFT is now patent-free (since 2020) and available in standard OpenCV
                self.detector = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, 
                                                contrastThreshold=0.04, edgeThreshold=10)
                self.detector_type = "SIFT"
                print("SURF not available, using SIFT detector as fallback")
            else:
                raise ImportError(
                    "SURF is not available. Please rebuild OpenCV with OPENCV_ENABLE_NONFREE=ON, "
                    "or set use_sift_fallback=True to use SIFT instead."
                )
        
        # Initialize FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
    
    def detect_and_describe(self, image):
        """
        Detect SURF keypoints and compute descriptors.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            keypoints: List of cv2.KeyPoint objects
            descriptors: ndarray of SURF descriptors
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        return keypoints, descriptors
    
    def match_features(self, descriptors1, descriptors2):
        """
        Match features between two sets of descriptors using FLANN.
        
        Args:
            descriptors1: Descriptors from first image
            descriptors2: Descriptors from second image
            
        Returns:
            List of good matches after ratio test
        """
        if descriptors1 is None or descriptors2 is None:
            return []
        
        # Ensure descriptors are float32
        descriptors1 = descriptors1.astype(np.float32)
        descriptors2 = descriptors2.astype(np.float32)
        
        # Find k nearest matches
        try:
            matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
        except cv2.error:
            return []
        
        # Apply Lowe's ratio test
        good_matches = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < self.ratio_thresh * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def find_homography(self, keypoints1, keypoints2, matches):
        """
        Find homography matrix using RANSAC.
        
        Args:
            keypoints1: Keypoints from first image
            keypoints2: Keypoints from second image
            matches: List of good matches
            
        Returns:
            Homography matrix (3x3) or None if not enough matches
            Mask of inliers
        """
        if len(matches) < self.min_matches:
            return None, None
        
        # Extract matched point coordinates
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find homography using RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        return H, mask
    
    def create_tps_blend_mask(self, shape, overlap_region, num_control_points=20):
        """
        Create a blending mask using Thin Plate Spline interpolation.
        
        Args:
            shape: Shape of the output mask (height, width)
            overlap_region: Tuple of (x_start, x_end) defining overlap area
            num_control_points: Number of control points for TPS
            
        Returns:
            Blending mask with smooth TPS-based transition
        """
        height, width = shape[:2]
        x_start, x_end = overlap_region
        overlap_width = x_end - x_start
        
        if overlap_width <= 0:
            # No overlap, return binary mask
            mask = np.zeros((height, width), dtype=np.float32)
            mask[:, :x_start] = 1.0
            return mask
        
        # Create control points along the overlap region
        control_y = np.linspace(0, height - 1, num_control_points)
        
        # Control points at the start of overlap (weight = 1)
        start_points = np.column_stack([np.full(num_control_points, x_start), control_y])
        
        # Control points at the end of overlap (weight = 0)
        end_points = np.column_stack([np.full(num_control_points, x_end), control_y])
        
        # Control points in the middle (weight = 0.5)
        mid_x = (x_start + x_end) / 2
        mid_points = np.column_stack([np.full(num_control_points, mid_x), control_y])
        
        # Combine control points
        all_points = np.vstack([start_points, mid_points, end_points])
        all_values = np.concatenate([
            np.ones(num_control_points),
            np.full(num_control_points, 0.5),
            np.zeros(num_control_points)
        ])
        
        # Create TPS interpolator
        tps = ThinPlateSpline(all_points, all_values, regularization=1e-6)
        
        # Create grid of points in overlap region
        x_range = np.arange(width)
        y_range = np.arange(height)
        xx, yy = np.meshgrid(x_range, y_range)
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        
        # Evaluate TPS at all grid points
        mask_values = tps.evaluate(grid_points)
        mask = mask_values.reshape(height, width)
        
        # Clamp values to [0, 1]
        mask = np.clip(mask, 0, 1)
        
        # Set regions outside overlap
        mask[:, :x_start] = 1.0
        mask[:, x_end:] = 0.0
        
        return mask.astype(np.float32)
    
    def warp_and_blend(self, img1, img2, H, use_tps_blend=True):
        """
        Warp first image using homography and blend with second image using TPS.
        
        Args:
            img1: First input image
            img2: Second input image
            H: Homography matrix
            use_tps_blend: Whether to use TPS blending (default True)
            
        Returns:
            Stitched and blended image
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Get the corners of both images
        corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
        corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        
        # Transform corners of first image
        corners1_transformed = cv2.perspectiveTransform(corners1, H)
        
        # Find the bounding box of the combined image
        all_corners = np.concatenate([corners1_transformed, corners2], axis=0)
        x_min = int(np.floor(all_corners[:, 0, 0].min()))
        x_max = int(np.ceil(all_corners[:, 0, 0].max()))
        y_min = int(np.floor(all_corners[:, 0, 1].min()))
        y_max = int(np.ceil(all_corners[:, 0, 1].max()))
        
        # Translation matrix to shift the image to positive coordinates
        translation = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Combined transformation (using np.dot for broader compatibility)
        H_combined = np.dot(translation, H)
        
        # Output size
        output_width = x_max - x_min
        output_height = y_max - y_min
        
        # Warp the first image
        warped1 = cv2.warpPerspective(img1, H_combined, (output_width, output_height))
        
        # Create output canvas and place second image
        output = np.zeros((output_height, output_width, 3), dtype=img2.dtype)
        
        # Position of img2 in the output
        x_offset = -x_min
        y_offset = -y_min
        
        # Place img2 in the output
        output[y_offset:y_offset + h2, x_offset:x_offset + w2] = img2
        
        # Find overlap region
        # Create masks for both images
        mask1 = (warped1.sum(axis=2) > 0).astype(np.float32)
        mask2 = np.zeros((output_height, output_width), dtype=np.float32)
        mask2[y_offset:y_offset + h2, x_offset:x_offset + w2] = 1.0
        
        # Find overlap
        overlap = (mask1 > 0) & (mask2 > 0)
        
        if use_tps_blend and np.any(overlap):
            # Find overlap bounds
            overlap_cols = np.where(np.any(overlap, axis=0))[0]
            if len(overlap_cols) > 0:
                x_start = overlap_cols[0]
                x_end = overlap_cols[-1]
                
                # Create TPS blend mask
                blend_mask = self.create_tps_blend_mask(
                    (output_height, output_width),
                    (x_start, x_end)
                )
                
                # Apply blending
                blend_mask_3c = blend_mask[:, :, np.newaxis]
                blended = (warped1 * blend_mask_3c + output * (1 - blend_mask_3c)).astype(img2.dtype)
                
                # Use warped1 where only img1 exists
                only_img1 = (mask1 > 0) & (mask2 == 0)
                output[only_img1] = warped1[only_img1]
                
                # Use blended in overlap region
                output[overlap] = blended[overlap]
            else:
                # Fallback to simple overlay
                output[mask1 > 0] = warped1[mask1 > 0]
        else:
            # Simple blending (average in overlap)
            both = (mask1 > 0) & (mask2 > 0)
            only_img1 = (mask1 > 0) & (mask2 == 0)
            
            output[only_img1] = warped1[only_img1]
            output[both] = ((warped1[both].astype(np.float32) + output[both].astype(np.float32)) / 2).astype(img2.dtype)
        
        return output
    
    def stitch(self, img1, img2, use_tps_blend=True):
        """
        Stitch two images together using SURF features and TPS blending.
        
        Args:
            img1: First input image (will be warped)
            img2: Second input image (reference)
            use_tps_blend: Whether to use TPS blending (default True)
            
        Returns:
            Stitched image or None if stitching failed
        """
        # Detect features
        kp1, desc1 = self.detect_and_describe(img1)
        kp2, desc2 = self.detect_and_describe(img2)
        
        if desc1 is None or desc2 is None:
            print("Error: Could not compute descriptors for one or both images")
            return None
        
        print(f"Found {len(kp1)} keypoints in image 1")
        print(f"Found {len(kp2)} keypoints in image 2")
        
        # Match features
        matches = self.match_features(desc1, desc2)
        print(f"Found {len(matches)} good matches")
        
        if len(matches) < self.min_matches:
            print(f"Error: Not enough matches ({len(matches)} < {self.min_matches})")
            return None
        
        # Find homography
        H, mask = self.find_homography(kp1, kp2, matches)
        
        if H is None:
            print("Error: Could not compute homography")
            return None
        
        inliers = mask.ravel().sum()
        print(f"Homography computed with {inliers} inliers")
        
        # Warp and blend
        result = self.warp_and_blend(img1, img2, H, use_tps_blend)
        
        return result
    
    def visualize_matches(self, img1, img2, kp1, kp2, matches, max_matches=100):
        """
        Visualize feature matches between two images.
        
        Args:
            img1: First input image
            img2: Second input image
            kp1: Keypoints from first image
            kp2: Keypoints from second image
            matches: List of matches
            max_matches: Maximum number of matches to draw
            
        Returns:
            Image showing matches
        """
        # Limit number of matches for visualization
        matches_to_draw = matches[:max_matches]
        
        # Draw matches
        result = cv2.drawMatches(
            img1, kp1, img2, kp2, matches_to_draw, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        return result


def stitch_images(img1_path, img2_path, output_path=None, use_tps_blend=True):
    """
    Convenience function to stitch two images from file paths.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        output_path: Optional path to save the result
        use_tps_blend: Whether to use TPS blending
        
    Returns:
        Stitched image
    """
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None:
        raise ValueError(f"Could not load image: {img1_path}")
    if img2 is None:
        raise ValueError(f"Could not load image: {img2_path}")
    
    # Create stitcher and stitch
    stitcher = SURFImageStitcher()
    result = stitcher.stitch(img1, img2, use_tps_blend=use_tps_blend)
    
    # Save if output path provided
    if result is not None and output_path is not None:
        cv2.imwrite(output_path, result)
        print(f"Result saved to: {output_path}")
    
    return result


def main():
    """Main entry point for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Stitch two images using SURF and TPS blending")
    parser.add_argument("image1", help="Path to first image")
    parser.add_argument("image2", help="Path to second image")
    parser.add_argument("-o", "--output", default="stitched_result.jpg", help="Output path")
    parser.add_argument("--no-tps", action="store_true", help="Disable TPS blending")
    
    args = parser.parse_args()
    
    result = stitch_images(args.image1, args.image2, args.output, not args.no_tps)
    
    if result is not None:
        print("Stitching completed successfully!")
    else:
        print("Stitching failed!")


if __name__ == "__main__":
    main()
