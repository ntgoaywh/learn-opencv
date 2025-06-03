import cv2
import numpy as np
from scipy.ndimage import minimum_filter

try:
    from cv2.ximgproc import guidedFilter
    GUIDED_FILTER_AVAILABLE = True
except ImportError:
    GUIDED_FILTER_AVAILABLE = False
    print('Warning:cv2.ximgsproc.guidedFilter not found.Transmission refinement will be suboptimal')
    print("C")
