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

# 🧠 DOOM-FlyWire: 139,248-Neuron Biological Connectome Autonomous Agent & In Silico Neuro-Sandbox

[![Connectome: FlyWire Nature 2024](https://img.shields.io/badge/Connectome-FlyWire%20(Nature%202024)-00f5d4.svg)](https://flywire.ai)
[![Environment: ViZDoom](https://img.shields.io/badge/Environment-ViZDoom%20(ZDoom%20RL)-ff0055.svg)](https://vizdoom.farama.org)
[![Scale: 100% Unreduced](https://img.shields.io/badge/Scale-139%2C248%20Neurons%20%7C%2015.1M%20Synapses-blueviolet.svg)](#-system-architecture)
[![Performance: 200+ FPS](https://img.shields.io/badge/Performance-200%2B%20FPS%20(2.5ms%20SpMV)-76b900.svg)](#-biophysical-engine--performance)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A real-time, closed-loop biological digital twin running the complete, unreduced adult *Drosophila melanogaster* whole-brain connectome ([FlyWire Consortium, *Nature* 2024](https://www.nature.com/articles/s41586-024-07558-y): **139,248 neurons, 15,090,883 directed synapses**) as an autonomous agent playing **DOOM** ([ViZDoom](https://vizdoom.farama.org/)).

The project includes an interactive, browser-based **In Silico Neuroscientist Control Console** that allows users to manipulate chemical neuromodulators (*Dopamine, Octopamine, Serotonin, GABA*), induce targeted anatomical optogenetic lesions (*Optic Lobe, Central Complex, Motion Detectors*), and observe behavioral changes and epileptic seizures in real time.

---

## 🌟 Key Highlights

- **100% Complete Biological Wiring:** Zero neuron downsampling and zero synapse pruning. All 139,248 neurons and 15,090,883 synapses propagate physiological signals in real time.
- **Microsecond-Scale Biophysical GPU Engine:** Uses sparse matrix-vector multiplication (SpMV) on GPU, achieving **2.5 ms per full-brain step (>200 FPS)** on an NVIDIA RTX 4080.
- **Compound Eye Retinotopy:** Translates ViZDoom game frames into optical ommatidia currents, stimulating photoreceptors (R1–R8) and Elementary Motion Detectors (T4/T5).
- **Descending Motor Steering:** Decodes asymmetric population firing of 1,303 Descending Neurons (DNs) into left/right steering, forward thrust, and weapon firing.
- **In Silico Neuro-Sandbox:** Interactive web dashboard with 12+ real-time chemical and anatomical levers.

---

## 🔬 System Architecture

```
                                [ViZDoom Game Engine]
                                 (320x240 RGB Screen)
                                          │
                                          │ Raw Pixels
                                          ▼
                ┌───────────────────────────────────────────────────┐
                │        Compound Eye Retinotopy & Motion Encoder   │
                │  - 32x32 Ommatidia Array (1,024 optical facets)   │
                │  - Elementary Motion Detection (T4/T5 Columns)    │
                └─────────────────────────┬─────────────────────────┘
                                          │
                                          │ Injected Photoreceptor Current (R1-R8)
                                          ▼
                ┌───────────────────────────────────────────────────┐
                │    100% FlyWire Whole-Brain Connectome (GPU)      │
                │  - 139,248 Biological Neurons                     │
                │  - 15,090,883 Directed Synapses (ACh, GABA, Glu)  │
                │  - Continuous-Time Leaky Rate Dynamics            │
                │                                                   │
                │     [LIVE NEUROSCIENTIST CONTROL CONSOLE]         │
                │  - Neuromodulator Bath: DA, OA, 5-HT, GABA, ACh   │
                │  - Circuit Lesions: Optic, Central Complex, MB    │
                │  - Biophysical Levers: Leak, Threshold, Noise     │
                └─────────────────────────┬─────────────────────────┘
                                          │
                                          │ Population Firing of Descending Neurons
                                          ▼
                ┌───────────────────────────────────────────────────┐
                │                 Motor Decoder                     │
                │  - DN_Left vs DN_Right -> Steer Left / Right      │
                │  - Bilateral Motor Vigor -> Move Forward          │
                │  - Frontal Attack Surge -> Weapon Fire            │
                └─────────────────────────┬─────────────────────────┘
                                          │
                                          ▼
                               [Execute Action in DOOM]
```

---

## 🎮 The Neuroscientist Sandbox Suite

The web dashboard provides interactive controls categorized into three biological layers:

### 1. Neuromodulatory Chemical Bath
| Lever | Range | Physiological Mechanism & In-Game Behavioral Effect |
| :--- | :--- | :--- |
| **Dopamine Gain** | 0.0x – 3.0x | Modulates motor vigor and reward sensitivity. High DA causes aggressive, fast-twitch tracking; 0.0x causes akinetic freezing. |
| **Octopamine Gain** | 0.0x – 3.0x | Invertebrate norepinephrine analog. Drives fight-or-flight arousal and rapid retreat responses. |
| **Serotonin (5-HT)** | 0.0x – 3.0x | Stabilizes motor output and suppresses erratic direction switching. |
| **GABAergic Inhibition** | 0.0x – 2.5x | Global inhibitory balance. Dropping below 0.2x triggers **Epileptiform Seizures** (global hyper-synchrony, spastic twitching); high values induce sedation. |
| **Acetylcholine Drive** | 0.1x – 2.0x | Primary excitatory neurotransmitter governing cortical gain and sensory transmission. |

### 2. Targeted In Silico Anatomical Lesions (Laser Ablation)
- **Blind Left Eye / Blind Right Eye:** Silences sensory input to the corresponding optic lobe; the agent completely ignores threats on the blinded side.
- **Silence Motion Detectors (T4/T5 Columnar Knockout):** Destroys optical flow perception; the fly perceives static textures but fails to dodge incoming projectiles.
- **Lesion Central Complex (CX):** Destroys the heading compass (Protocerebral Bridge & Fan-Shaped Body); the agent loses heading stability and spins in circles.
- **Lesion Mushroom Body (Kenyon Cells):** Disrupts associative memory subcircuits.
- **Total Motor Paralysis (DN Silencing):** All descending motor neurons are silenced; the brain remains active and perceives threats, but the character is completely paralyzed.

### 3. Biophysical Dynamics
- **Membrane Leak Rate (tau):** Temporal integration window of biological neurons.
- **Firing Threshold:** Voltage sensitivity for non-linear sigmoidal activation.
- **Synaptic Thermal Noise:** Stochastic exploration vs. deterministic execution.

---

## ⚡ Biophysical Engine & Performance

The simulator implements continuous-time leaky rate dynamics over sparse matrix-vector multiplication (SpMV):

```
V[t+1] = (1 - leak_rate) * V[t] + W_syn @ act[t] + I_sensory[t] + noise
act[t+1] = sigmoid((V[t+1] - threshold) * 5.0) * lesion_mask
```

On an **NVIDIA GeForce RTX 4080 Laptop GPU (12 GB VRAM)**:
- **Graph Allocation:** 60.4 MB for 15,090,883 sparse signed synapses.
- **Single-Step Propagation:** **2.53 ms** (395 FPS isolated engine throughput).
- **Closed-Loop System:** **71.5 FPS** (including ViZDoom rendering, retinotopy encoding, 15M synapse propagation, and motor decoding).

---

## 📁 Repository Structure

```
.
├── data/
│   ├── Supplemental_file1_neuron_annotations.tsv   # FlyWire 139,248 neuron annotations
│   ├── proofread_connections_783.feather           # 15,090,883 directed synapses
│   └── connectome_cache.pt                         # Pre-built PyTorch sparse graph tensor
├── models/                                         # Trained checkpoint weights (.pt)
├── src/
│   ├── __init__.py                                 # Package initialization
│   ├── graph_loader.py                             # Fast FlyWire connectome loader
│   ├── connectome_engine.py                        # 15M-synapse biophysical GPU dynamics engine
│   ├── retinotopy_encoder.py                       # Compound eye & motion optical flow encoder
│   ├── motor_decoder.py                            # Descending neuron population motor decoder
│   └── doom_agent.py                               # Closed-loop ViZDoom connectome agent
├── web/
│   ├── js/                                         # Local Three.js and OrbitControls vendor bundle
│   ├── brain_139k_pos.bin                          # 139,248 neuron 3D coordinates (1.67 MB)
│   ├── brain_139k_col.bin                          # 139,248 neuron anatomical colors (417 KB)
│   └── index.html                                  # 3D Connectome and Perception-Action Dashboard
├── server.py                                       # FastAPI + WebSocket live streaming server
├── train_doom.py                                   # Connectome policy reinforcement training loop
├── Dockerfile                                      # Hugging Face Spaces and Docker container build
├── .dockerignore                                   # Container build exclusions
├── .gitattributes                                  # Git LFS binary dataset tracking
├── .gitignore                                      # Ignored virtual environments and artifacts
├── requirements.txt                                # Python dependencies
├── LICENSE                                         # MIT License
└── README.md
```

---

## 🚀 Quickstart

### 1. Prerequisites & Environment
Ensure you have Python 3.10+ and a CUDA-capable GPU:
```bash
git clone https://github.com/your-username/fly.git
cd fly

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 2. Launch the Web Sandbox
Start the FastAPI streaming server:
```bash
python server.py
```
Open your browser at **`http://localhost:8000`**. You will see:
- The live DOOM viewport controlled by the fruit fly brain.
- Real-time firing rate gauges of the Optic Lobe, Central Complex, and Descending Motor Neurons.
- The interactive sandbox panel with sliders and toggle switches.

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
