"""
Fiziksel Sulu Boya Simülasyon Motoru (Physics-Based Watercolor Engine)
Akışkanlar Mekaniği, Kılcallık, Deegan Buharlaşması ve Kubelka-Munk Optiği
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
