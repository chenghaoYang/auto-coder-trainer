"""SSD (Simple Self-Distillation) trainer — launcher bundle for Sample → Fine-tune → Evaluate pipeline.

Based on Apple's ml-ssd (https://github.com/apple/ml-ssd) and the paper
"Embarrassingly Simple Self-Distillation Improves Code Generation" (arXiv: 2604.01193).
"""

from trainers.ssd.launcher import SSDLauncher

__all__ = ["SSDLauncher"]
