"""
SURF-based Image Stitching with TPS (Thin Plate Spline) Fusion

This package provides image stitching capabilities using:
- SURF (Speeded Up Robust Features) for feature detection
- Thin Plate Spline for smooth image blending
"""

from .surf_tps_stitcher import SURFImageStitcher, ThinPlateSpline, stitch_images, main

__all__ = ['SURFImageStitcher', 'ThinPlateSpline', 'stitch_images', 'main']
__version__ = '1.0.0'
