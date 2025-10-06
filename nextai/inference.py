import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deploy.inference import NextAI
import torch

__all__ = ["NextAI"]
