"""Video Jev for ComfyUI. https://huggingface.co/sp1d5r/video-jev"""

import os
import sys

# The vendored `vjev` package imports itself by absolute name.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
