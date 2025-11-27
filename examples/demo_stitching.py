"""
Example script demonstrating image stitching using SURF/SIFT with TPS fusion.

This script creates sample test images and demonstrates the stitching process.
"""

import cv2
import numpy as np
import os
import sys

# Support both installed package and direct execution
try:
    from src.surf_tps_stitcher import SURFImageStitcher
except ImportError:
    # Add parent directory to path for direct execution
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.surf_tps_stitcher import SURFImageStitcher


def create_sample_images():
    """
    Create sample images for testing the stitching algorithm.
    
    Creates two overlapping images with various features that can be matched.
    """
    # Create a larger pattern image
    full_img = np.zeros((300, 600, 3), dtype=np.uint8)
    
    # Add a gradient background
    for i in range(600):
        full_img[:, i] = [int(255 * i / 600), 50, int(255 * (600 - i) / 600)]
    
    # Add geometric features
    cv2.rectangle(full_img, (50, 50), (200, 200), (255, 255, 255), 3)
    cv2.rectangle(full_img, (350, 100), (550, 250), (255, 255, 0), 3)
    cv2.circle(full_img, (125, 125), 50, (0, 255, 0), 3)
    cv2.circle(full_img, (450, 175), 50, (0, 255, 255), 3)
    cv2.ellipse(full_img, (300, 150), (80, 40), 0, 0, 360, (255, 0, 255), 3)
    
    # Add some lines crossing the overlap region
    cv2.line(full_img, (200, 0), (400, 300), (255, 128, 64), 2)
    cv2.line(full_img, (250, 300), (350, 0), (64, 128, 255), 2)
    
    # Add text
    cv2.putText(full_img, 'LEFT', (100, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(full_img, 'RIGHT', (400, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add texture noise for better feature detection
    noise = np.random.randint(0, 30, full_img.shape, dtype=np.uint8)
    full_img = cv2.add(full_img, noise)
    
    # Split into two overlapping images (100 pixel overlap in the center)
    overlap_size = 100
    mid_point = 300
    
    img1 = full_img[:, :mid_point + overlap_size // 2].copy()
    img2 = full_img[:, mid_point - overlap_size // 2:].copy()
    
    return img1, img2, full_img


def main():
    """Main function to demonstrate image stitching."""
    print("=" * 60)
    print("Image Stitching Example using SURF/SIFT with TPS Fusion")
    print("=" * 60)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sample images
    print("\n1. Creating sample images...")
    img1, img2, original = create_sample_images()
    print(f"   Image 1 size: {img1.shape}")
    print(f"   Image 2 size: {img2.shape}")
    print(f"   Original size: {original.shape}")
    
    # Save input images
    cv2.imwrite(os.path.join(output_dir, 'input_image1.jpg'), img1)
    cv2.imwrite(os.path.join(output_dir, 'input_image2.jpg'), img2)
    cv2.imwrite(os.path.join(output_dir, 'original_full.jpg'), original)
    print("   Saved input images to output directory")
    
    # Create stitcher
    print("\n2. Initializing stitcher...")
    stitcher = SURFImageStitcher(
        hessian_threshold=400,
        ratio_thresh=0.7,
        min_matches=4
    )
    print(f"   Using {stitcher.detector_type} detector")
    
    # Detect features
    print("\n3. Detecting features...")
    kp1, desc1 = stitcher.detect_and_describe(img1)
    kp2, desc2 = stitcher.detect_and_describe(img2)
    print(f"   Image 1: {len(kp1)} keypoints")
    print(f"   Image 2: {len(kp2)} keypoints")
    
    # Visualize keypoints
    img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), 
                                 flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0), 
                                 flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(os.path.join(output_dir, 'keypoints_image1.jpg'), img1_kp)
    cv2.imwrite(os.path.join(output_dir, 'keypoints_image2.jpg'), img2_kp)
    print("   Saved keypoint visualizations")
    
    # Match features
    print("\n4. Matching features...")
    matches = stitcher.match_features(desc1, desc2)
    print(f"   Found {len(matches)} good matches")
    
    # Visualize matches
    match_img = stitcher.visualize_matches(img1, img2, kp1, kp2, matches)
    cv2.imwrite(os.path.join(output_dir, 'feature_matches.jpg'), match_img)
    print("   Saved match visualization")
    
    # Stitch with TPS blending
    print("\n5. Stitching with TPS blending...")
    result_tps = stitcher.stitch(img1, img2, use_tps_blend=True)
    
    if result_tps is not None:
        cv2.imwrite(os.path.join(output_dir, 'stitched_tps.jpg'), result_tps)
        print(f"   Result size: {result_tps.shape}")
        print("   Saved TPS blended result")
    else:
        print("   Stitching failed!")
    
    # Stitch without TPS blending (for comparison)
    print("\n6. Stitching without TPS blending (simple average)...")
    result_simple = stitcher.stitch(img1, img2, use_tps_blend=False)
    
    if result_simple is not None:
        cv2.imwrite(os.path.join(output_dir, 'stitched_simple.jpg'), result_simple)
        print(f"   Result size: {result_simple.shape}")
        print("   Saved simple blended result")
    
    print("\n" + "=" * 60)
    print("Done! Check the 'examples/output' directory for results.")
    print("=" * 60)
    
    # List output files
    print("\nGenerated files:")
    for f in sorted(os.listdir(output_dir)):
        filepath = os.path.join(output_dir, f)
        size = os.path.getsize(filepath)
        print(f"  - {f} ({size} bytes)")


if __name__ == "__main__":
    main()
