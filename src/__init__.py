"""
DOOM-FlyWire: 139,248-Neuron Biological Connectome Autonomous Agent
===================================================================
A real-time whole-brain biological digital twin mapping the complete adult
Drosophila melanogaster connectome (FlyWire Nature 2024: 139,248 neurons,
15,090,883 synapses) to ViZDoom with an interactive in silico neuro-sandbox.
"""

from .graph_loader import FlyConnectomeLoader, load_flywire_graph

__all__ = [
    "FlyConnectomeLoader",
    "load_flywire_graph",
]
