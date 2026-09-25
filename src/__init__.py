"""
DOOM-FlyWire: 139,248-Neuron Biological Connectome Autonomous Agent
===================================================================
A real-time whole-brain biological digital twin mapping the complete adult
Drosophila melanogaster connectome (FlyWire Nature 2024: 139,248 neurons,
15,090,883 synapses) to ViZDoom with a real-time 3D neural activity visualizer.
"""

from .graph_loader import FlyConnectomeLoader, load_flywire_graph
from .connectome_engine import ConnectomeEngine
from .retinotopy_encoder import RetinotopyEncoder
from .motor_decoder import MotorDecoder
from .doom_agent import DoomConnectomeAgent

__all__ = [
    "FlyConnectomeLoader",
    "load_flywire_graph",
    "ConnectomeEngine",
    "RetinotopyEncoder",
    "MotorDecoder",
    "DoomConnectomeAgent",
]
