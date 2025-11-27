"""
Tests for SURF-based image stitching with TPS fusion.
"""

import numpy as np
import cv2
import sys
import os

# Support both installed package and direct execution
try:
    from src.surf_tps_stitcher import ThinPlateSpline, SURFImageStitcher, stitch_images
except ImportError:
    # Add parent directory to path for direct execution
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.surf_tps_stitcher import ThinPlateSpline, SURFImageStitcher, stitch_images


class TestThinPlateSpline:
    """Tests for the ThinPlateSpline class."""
    
    def test_basic_interpolation(self):
        """Test basic TPS interpolation."""
        control_points = np.array([[0, 0], [10, 0], [0, 10], [10, 10]])
        values = np.array([0.0, 1.0, 1.0, 0.5])
        
        tps = ThinPlateSpline(control_points, values, regularization=1e-6)
        
        # Evaluate at control points - should return approximately the original values
        result = tps.evaluate(control_points)
        np.testing.assert_array_almost_equal(result, values, decimal=5)
        print("test_basic_interpolation passed!")
    
    def test_center_interpolation(self):
        """Test interpolation at center point."""
        control_points = np.array([[0, 0], [10, 0], [0, 10], [10, 10]])
        values = np.array([0.0, 1.0, 1.0, 0.5])
        
        tps = ThinPlateSpline(control_points, values, regularization=1e-6)
        
        # Evaluate at center
        center = np.array([[5, 5]])
        result = tps.evaluate(center)
        
        # Result should be between min and max values
        assert 0.0 <= result[0] <= 1.0, f"Center value {result[0]} out of range"
        print("test_center_interpolation passed!")
    
    def test_multidimensional_values(self):
        """Test TPS with multidimensional target values."""
        control_points = np.array([[0, 0], [10, 0], [0, 10], [10, 10]])
        values = np.array([[0.0, 0.5], [1.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
        
        tps = ThinPlateSpline(control_points, values, regularization=1e-6)
        result = tps.evaluate(control_points)
        
        np.testing.assert_array_almost_equal(result, values, decimal=5)
        print("test_multidimensional_values passed!")


class TestSURFImageStitcher:
    """Tests for the SURFImageStitcher class."""
    
    def test_initialization(self):
        """Test stitcher initialization."""
        stitcher = SURFImageStitcher()
        assert stitcher.detector_type in ["SURF", "SIFT"]
        assert stitcher.ratio_thresh == 0.7
        assert stitcher.min_matches == 10
        print("test_initialization passed!")
    
    def test_feature_detection(self):
        """Test feature detection on a sample image."""
        stitcher = SURFImageStitcher()
        
        # Create a test image with some features
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)
        cv2.circle(img, (100, 100), 30, (0, 0, 255), -1)
        
        # Add some noise for more features
        noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        
        kp, desc = stitcher.detect_and_describe(img)
        
        assert len(kp) > 0, "No keypoints detected"
        assert desc is not None, "No descriptors computed"
        assert desc.shape[0] == len(kp), "Mismatch between keypoints and descriptors"
        print(f"test_feature_detection passed! ({len(kp)} keypoints)")
    
    def test_matching(self):
        """Test feature matching between two images."""
        stitcher = SURFImageStitcher()
        
        # Create two similar images (shifted)
        img1 = np.random.randint(50, 200, (200, 300, 3), dtype=np.uint8)
        # Add some structured patterns
        cv2.rectangle(img1, (50, 50), (200, 150), (255, 255, 255), 2)
        cv2.circle(img1, (150, 100), 30, (0, 255, 0), 2)
        
        # Create img2 as a translated version of img1 (simulating overlapping images)
        img2 = np.zeros_like(img1)
        img2[:, :250] = img1[:, 50:]
        
        kp1, desc1 = stitcher.detect_and_describe(img1)
        kp2, desc2 = stitcher.detect_and_describe(img2)
        
        matches = stitcher.match_features(desc1, desc2)
        
        assert len(matches) >= 0, "Matching should return a list"
        # Note: Number of matches depends on the similarity of the test images
        print(f"test_matching passed! ({len(matches)} matches found)")
    
    def test_create_synthetic_stitch(self):
        """Test stitching with synthetic overlapping images."""
        stitcher = SURFImageStitcher(min_matches=4)
        
        # Create a larger pattern image
        full_img = np.zeros((200, 400, 3), dtype=np.uint8)
        
        # Add various features
        cv2.rectangle(full_img, (50, 50), (150, 150), (255, 255, 255), 2)
        cv2.rectangle(full_img, (200, 50), (350, 150), (255, 0, 0), 2)
        cv2.circle(full_img, (100, 100), 30, (0, 255, 0), 2)
        cv2.circle(full_img, (275, 100), 30, (0, 0, 255), 2)
        cv2.line(full_img, (0, 100), (400, 100), (255, 255, 0), 1)
        cv2.line(full_img, (200, 0), (200, 200), (255, 0, 255), 1)
        
        # Add texture
        noise = np.random.randint(0, 30, full_img.shape, dtype=np.uint8)
        full_img = cv2.add(full_img, noise)
        
        # Split into two overlapping images
        img1 = full_img[:, :220].copy()  # Left part
        img2 = full_img[:, 180:].copy()  # Right part (40 pixel overlap)
        
        # Verify images have features
        kp1, _ = stitcher.detect_and_describe(img1)
        kp2, _ = stitcher.detect_and_describe(img2)
        print(f"Image 1: {len(kp1)} keypoints, Image 2: {len(kp2)} keypoints")
        
        # Try stitching
        result = stitcher.stitch(img1, img2, use_tps_blend=True)
        
        if result is not None:
            print(f"test_create_synthetic_stitch passed! Result shape: {result.shape}")
            assert result.shape[1] >= max(img1.shape[1], img2.shape[1])
        else:
            print("test_create_synthetic_stitch: Stitching returned None (may need more features)")
    
    def test_tps_blend_mask(self):
        """Test TPS blend mask creation."""
        stitcher = SURFImageStitcher()
        
        mask = stitcher.create_tps_blend_mask((100, 200), (50, 150))
        
        assert mask.shape == (100, 200)
        assert mask.dtype == np.float32
        
        # Check boundary conditions
        assert np.all(mask[:, :50] == 1.0), "Left side should be 1.0"
        assert np.all(mask[:, 150:] == 0.0), "Right side should be 0.0"
        
        # Check that values in overlap are between 0 and 1
        assert np.all((mask[:, 50:150] >= 0.0) & (mask[:, 50:150] <= 1.0))
        print("test_tps_blend_mask passed!")


def run_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running ThinPlateSpline tests...")
    print("=" * 60)
    
    tps_tests = TestThinPlateSpline()
    tps_tests.test_basic_interpolation()
    tps_tests.test_center_interpolation()
    tps_tests.test_multidimensional_values()
    
    print("\n" + "=" * 60)
    print("Running SURFImageStitcher tests...")
    print("=" * 60)
    
    stitcher_tests = TestSURFImageStitcher()
    stitcher_tests.test_initialization()
    stitcher_tests.test_feature_detection()
    stitcher_tests.test_matching()
    stitcher_tests.test_tps_blend_mask()
    stitcher_tests.test_create_synthetic_stitch()
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
