"""
Physics-Based Watercolor Engine
Fluid Dynamics, Capillarity, Deegan Evaporation, and Kubelka-Munk Optics
"""

from .paper import PaperSubstrate
from .fluid import ShallowWaterSolver
from .pigment import PigmentManager, PigmentLayer
from .optics import KubelkaMunkRenderer
from .simulation import WatercolorEngine

__all__ = [
    "PaperSubstrate",
    "ShallowWaterSolver",
    "PigmentManager",
    "PigmentLayer",
    "KubelkaMunkRenderer",
    "WatercolorEngine",
]
