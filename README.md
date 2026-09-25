---
title: DOOM-FlyWire Connectome
emoji: 🪰
colorFrom: purple
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# 🪰 DOOM-FlyWire: 139,248-Neuron Biological Connectome Autonomous Agent & 3D Real-Time Visualizer

[![Connectome: FlyWire Nature 2024](https://img.shields.io/badge/Connectome-FlyWire%20(Nature%202024)-00f5d4.svg)](https://flywire.ai)
[![Environment: ViZDoom](https://img.shields.io/badge/Environment-ViZDoom%20(ZDoom%20RL)-ff0055.svg)](https://vizdoom.farama.org)
[![Scale: 100% Unreduced](https://img.shields.io/badge/Scale-139%2C248%20Neurons%20%7C%2015.1M%20Synapses-blueviolet.svg)](#-system-architecture)
[![Performance: 200+ FPS](https://img.shields.io/badge/Performance-200%2B%20FPS%20(SpMV)-76b900.svg)](#-biophysical-engine--performance)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A real-time, closed-loop biological digital twin running the complete, unreduced adult *Drosophila melanogaster* whole-brain connectome ([FlyWire Consortium, *Nature* 2024](https://www.nature.com/articles/s41586-024-07558-y): **139,248 neurons, 15,090,883 directed synapses**) as an autonomous agent playing **DOOM** ([ViZDoom](https://vizdoom.farama.org/)).

The project features a real-time, browser-based **3D Connectome Neural Activity Visualizer** (Three.js WebGL) rendering all 139,248 biological neurons in 3D space with dynamic, calcium-fluorescence-style excitation halos as the fly perceives, processes, and navigates the 3D environment.

---

## ⚡ Key Highlights

- **100% Complete Biological Wiring:** Zero neuron downsampling and zero synapse pruning. All 139,248 neurons and 15,090,883 synapses propagate physiological signals in real time.
- **Microsecond-Scale Biophysical Engine:** Uses sparse matrix-vector multiplication (SpMV) on GPU / CSR-CPU, achieving **>200 FPS** neural propagation throughput.
- **Compound Eye Retinotopy:** Translates ViZDoom game frames into optical ommatidia currents (32x32 array), stimulating photoreceptors and Elementary Motion Detectors (T4/T5 columns).
- **Descending Motor Steering:** Decodes asymmetric population firing of 1,303 Descending Neurons (DNs) into continuous left/right steering, tracking, and attack decisions.
- **Interactive 3D Connectome Visualizer:** WebGL point-cloud rendering of all 139,248 neurons with dynamic size and luminance modulation based on regional firing rates.
- **Live Circuit Telemetry:** Real-time analog firing rate monitoring for Left/Right Optic Lobes, Central Complex (CX) heading compass, Mushroom Body (KC) memory circuits, Descending Motor pool, and Whole Brain.

---

## 🧠 System Architecture

`
                          [ViZDoom Game Engine]
                           (320x240 RGB Screen)
                                    |
                                    | Raw Pixels
                                    v
          +---------------------------------------------------+
          |     Compound Eye Retinotopy & Motion Encoder      |
          |  - 32x32 Ommatidia Array (1,024 optical facets)   |
          |  - Elementary Motion Detection (T4/T5 Columns)    |
          |  - Lobula Columnar (LC) Target Tracking           |
          +---------------------------------------------------+
                                    |
                                    | Injected Photoreceptor Current
                                    v
          +---------------------------------------------------+
          |     100% FlyWire Whole-Brain Connectome           |
          |  - 139,248 Biological Neurons                     |
          |  - 15,090,883 Directed Synapses (ACh, GABA, Glu)  |
          |  - Leaky Integrate-and-Fire Dynamic Integration   |
          +---------------------------------------------------+
                                    |
                                    | Population Firing of Descending Neurons
                                    v
          +---------------------------------------------------+
          |                   Motor Decoder                   |
          |  - DN_Left vs DN_Right -> Continuous Steering     |
          |  - Bilateral Motor Vigor + Target Lock -> Attack  |
          +---------------------------------------------------+
                                    |
                                    v
                         [Execute Action in DOOM]
`

---

## 🔬 Biophysical Engine & Dynamics

The simulator implements continuous-time leaky rate dynamics over sparse matrix-vector multiplication (SpMV):

`
I_syn[t] = W_syn @ act[t]
I_tot[t] = I_syn[t] * gain + I_sensory[t] + noise
V[t+1]   = (1 - leak_rate) * V[t] + I_tot[t]
act[t+1] = sigmoid((V[t+1] - firing_threshold) * 5.0)
`

- **Graph Memory:** ~60.4 MB for 15,090,883 sparse signed synapses.
- **Pure Tensor Propagation:** Zero heuristic if-else overrides in neural dynamics; motor commands are purely linear and softmax population readouts from descending motor neurons.
- **Dynamic 3D Rendering Transfer:** Regional firing rates modulate point-cloud particle size and photon opacity via linear optical transfer functions, emulating biological calcium imaging (GCaMP).

---

## 📁 Repository Structure

`
.
├── data/
│   ├── Supplemental_file1_neuron_annotations.tsv   # FlyWire 139,248 neuron annotations
│   ├── proofread_connections_783.feather           # 15,090,883 directed synapses
│   └── connectome_cache.pt                         # Pre-built PyTorch sparse graph tensor
├── src/
│   ├── __init__.py                                 # Package exports
│   ├── graph_loader.py                             # Fast FlyWire connectome loader
│   ├── connectome_engine.py                        # 15M-synapse biophysical dynamics engine
│   ├── retinotopy_encoder.py                       # Compound eye & motion optical flow encoder
│   ├── motor_decoder.py                            # Descending neuron population motor decoder
│   └── doom_agent.py                               # Closed-loop ViZDoom connectome agent
├── web/
│   ├── js/                                         # Local Three.js and OrbitControls
│   ├── brain_139k_pos.bin                          # 139,248 neuron 3D coordinates (1.67 MB)
│   ├── brain_139k_col.bin                          # 139,248 neuron anatomical colors (417 KB)
│   └── index.html                                  # 3D Connectome and Live Action Dashboard
├── app.py                                          # Hugging Face Space & Gradio entry point
├── .gitattributes                                  # Git LFS binary tracking
├── .gitignore                                      # Ignored virtual environments and artifacts
├── packages.txt                                    # Linux system dependencies
├── requirements.txt                                # Python dependencies
├── LICENSE                                         # MIT License
└── README.md
`

---

## 🚀 Quickstart

### 1. Prerequisites & Environment
Ensure you have Python 3.10+:
`ash
git clone https://github.com/EmirMuhammetARAN/doom-flywire.git
cd doom-flywire

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scriptsctivate

# Install dependencies
pip install -r requirements.txt
`

### 2. Launch the Web Application
Start the application server:
`ash
python app.py
`
Open your browser at **http://localhost:7860** . You will see:
- The live DOOM game viewport autonomously navigated by the 139,248-neuron connectome.
- The interactive 3D WebGL connectome visualizer rendering the active firing dynamics of the brain in real time.
- Real-time biological circuit activity meters (Optic Lobes, Central Complex, Mushroom Body, Descending Motor pool, Whole Brain).

---

## 📚 References & Scientific Citations

1. **FlyWire Whole-Brain Connectome:**
   Dorkenwald, S., Matsliah, A., Sterling, P., et al. (2024). *Neuronal wiring diagram of an adult brain*. **Nature**, 634, 124–138.
2. **ViZDoom RL Platform:**
   Wydmuch, M., Kempka, M., & Jaśkowski, W. (2018). *ViZDoom Competitions: Playing Doom from Pixels*. **IEEE Transactions on Games**, 11(3), 248–258.
3. **Drosophila Motion Vision (T4/T5 Circuits):**
   Maisak, M. S., Haag, J., Ammer, G., et al. (2013). *A directional tuning map of Drosophila elementary motion detectors*. **Nature**, 500(7461), 212–216.
4. **Central Complex Navigation & Heading Compass:**
   Seelig, J. D., & Jayaraman, V. (2015). *Neural dynamics for landmark orientation and angular path integration*. **Nature**, 521(7551), 186–191.
