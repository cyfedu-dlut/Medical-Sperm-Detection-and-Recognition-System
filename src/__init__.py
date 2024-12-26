"""
蝌蚪/精子图像智能检测与分类系统

该模块提供了图像处理、目标检测和分类的核心功能。
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# 导入主要功能，方便外部直接导入
from .detection import (
    is_abnormally_large_box,
    extract_small_bboxes
)
from .utils import pixel_ratio_classification
from .visualization import visualize_detection_steps

# 定义模块对外公开的接口
__all__ = [
    'is_abnormally_large_box',
    'extract_small_bboxes',
    'pixel_ratio_classification',
    'visualize_detection_steps'
]