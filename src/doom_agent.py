"""
DOOM-FlyWire Agent
==================
Closed-loop autonomous agent connecting the ViZDoom environment to the
139,248-neuron FlyWire connectome.
"""

import os
import sys
import io
import base64
import time
import numpy as np
from PIL import Image
import vizdoom as vzd
import torch

from .connectome_engine import ConnectomeEngine
from .retinotopy_encoder import RetinotopyEncoder
from .motor_decoder import MotorDecoder

class DoomConnectomeAgent:
    def __init__(
        self,
        scenario_name: str = "defend_the_center.cfg",
        window_visible: bool = False,
        device: str = "cuda"
    ):
        print(f"[DoomAgent] Initializing DOOM-FlyWire Agent on {device}...")
        self.device = device
        
        # 1. Connectome Neural Engine (100% FlyWire)
        self.engine = ConnectomeEngine(device=device)

        # 2. Retinotopy Encoder
        self.encoder = RetinotopyEncoder(
            num_neurons=self.engine.num_neurons,
            optic_left_indices=self.engine.optic_left_indices,
            optic_right_indices=self.engine.optic_right_indices,
            t4t5_indices=self.engine.t4t5_indices,
            eye_res=32,
            device=device
        )

        # 3. ViZDoom Game Instance
        self.game = vzd.DoomGame()
        scenario_path = os.path.join(vzd.scenarios_path, scenario_name)
        self.game.load_config(scenario_path)
        self.game.set_window_visible(window_visible)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_labels_buffer_enabled(True)
        
        # Ensure key game variables are accessible
        try:
            self.game.add_available_game_variable(vzd.GameVariable.KILLCOUNT)
            self.game.add_available_game_variable(vzd.GameVariable.HEALTH)
            self.game.add_available_game_variable(vzd.GameVariable.AMMO2)
        except Exception:
            pass

        self.game.init()
        self.game.new_episode()
        # Enable Infinite Ammo
        try:
            self.game.send_game_command("sv_infiniteammo 1")
        except Exception:
            pass

        # Extract available buttons from ViZDoom
        available_buttons = [b.name for b in self.game.get_available_buttons()]
        print(f"[DoomAgent] ViZDoom Scenario '{scenario_name}' loaded. Buttons: {available_buttons}")

        # 4. Motor Decoder (Pure Linear Synaptic Readout)
        self.decoder = MotorDecoder(
            dn_left_indices=self.engine.dn_left_indices,
            dn_right_indices=self.engine.dn_right_indices,
            available_actions=available_buttons,
            device=device
        )

        self.step_count = 0
        self.episode_kills = 0
        self.total_kills = 0
        self.episode_reward = 0.0
        self.total_reward = 0.0
        self.is_running = True
        self.is_game_over = False
        self.last_frame_b64 = ""

    def update_sandbox(self, params_dict: dict):
        """Allows live user adjustments of chemical bath and lesion levers."""
        self.engine.set_parameters(params_dict)

    def reset_episode(self):
        """Manually resets the game episode upon user command."""
        self.game.new_episode()
        try:
            self.game.send_game_command("sv_infiniteammo 1")
        except Exception:
            pass
        self.engine.reset_state()
        self.episode_reward = 0.0
        self.episode_kills = 0
        self.is_running = True
        self.is_game_over = False

    def toggle_pause(self) -> bool:
        """Toggles manual pause state."""
        self.is_running = not self.is_running
        return self.is_running

    def step(self) -> dict:
        """
        Executes one full perception-cognition-action loop:
        DOOM Frame -> Optic Lobe Current -> 15M Synapses -> Descending Neurons -> DOOM Action.
        Perpetual autonomous gameplay: automatically starts a new round when the fly dies.
        """
        if self.game.is_episode_finished() or (self.game.get_state() is not None and self.game.get_game_variable(vzd.GameVariable.HEALTH) <= 0):
            self.game.new_episode()
            try:
                self.game.send_game_command("sv_infiniteammo 1")
            except Exception:
                pass
            self.episode_reward = 0.0
            self.episode_kills = 0
            self.is_game_over = False

        if not self.is_running:
            return {
                "step": self.step_count,
                "reward": 0.0,
                "episode_reward": self.episode_reward,
                "total_reward": self.total_reward,
                "health": 100,
                "ammo": 999,
                "kills": self.episode_kills,
                "total_kills": self.total_kills,
                "is_game_over": False,
                "is_running": False,
                "telemetry": self.engine.get_telemetry(),
                "action_taken": [],
                "action_probs": {"TURN_LEFT": 0.0, "TURN_RIGHT": 0.0, "ATTACK": 0.0},
                "frame_b64": self.last_frame_b64
            }

        state = self.game.get_state()
        if state is None:
            return {}

        # 1. Visual Perception (Compound Eye with Optic Lesions & Target Tracking)
        screen_rgb = state.screen_buffer  # (H, W, 3) RGB
        sensory_current, visual_meta = self.encoder.encode(screen_rgb, state.labels, params=self.engine.params)

        # 2. Whole-Brain Biological Propagation (15M synapses)
        act, telemetry = self.engine.step(sensory_current)

        # 3. Motor Action Decoding (Honors Chemical Bath, Seizures, Sedation, & Lesions)
        action_binary, action_probs = self.decoder.decode(act, visual_meta, params=self.engine.params)

        # 4. Execute in DOOM (2 frame skip for natural, smooth, non-hyper speed)
        reward = self.game.make_action(action_binary, 2)
        self.episode_reward += reward
        self.total_reward += reward
        self.step_count += 1

        # Check kills / game variables
        kills = self.game.get_game_variable(vzd.GameVariable.KILLCOUNT) if vzd.GameVariable.KILLCOUNT in self.game.get_available_game_variables() else 0
        self.episode_kills = kills
        health = self.game.get_game_variable(vzd.GameVariable.HEALTH) if vzd.GameVariable.HEALTH in self.game.get_available_game_variables() else 100
        ammo = 999  # Infinite Ammo active

        if health <= 0 or self.game.is_episode_finished():
            self.total_kills += self.episode_kills
            self.game.new_episode()
            try:
                self.game.send_game_command("sv_infiniteammo 1")
            except Exception:
                pass
            self.episode_kills = 0
            self.episode_reward = 0.0
            self.is_game_over = False
            self.is_running = True

        # Encode crisp 640x480 frame as lightweight JPEG base64
        pil_img = Image.fromarray(screen_rgb)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=70)
        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        self.last_frame_b64 = img_b64

        return {
            "step": self.step_count,
            "reward": reward,
            "episode_reward": self.episode_reward,
            "total_reward": self.episode_reward,
            "health": max(0, health),
            "ammo": ammo,
            "kills": kills,
            "is_game_over": self.is_game_over,
            "is_running": self.is_running,
            "telemetry": telemetry,
            "visual_meta": visual_meta,
            "action_taken": [name for name, val in zip(self.decoder.available_actions, action_binary) if val == 1],
            "action_probs": action_probs,
            "frame_b64": img_b64
        }


    def close(self):
        self.game.close()

if __name__ == "__main__":
    agent = DoomConnectomeAgent(window_visible=False)
    print("Testing 50 steps of closed-loop DOOM-FlyWire gameplay...")
    t0 = time.perf_counter()
    for s in range(50):
        res = agent.step()
        if s % 10 == 0:
            print(f"Step {s:3d} | Action: {res['action_taken']} | Telemetry: Optic={res['telemetry']['optic_left']:.2f}/{res['telemetry']['optic_right']:.2f} | Brain={res['telemetry']['whole_brain_firing']:.3f}")
    agent.close()
    elapsed = time.perf_counter() - t0
    print(f"50 closed-loop game steps completed in {elapsed:.2f}s ({50/elapsed:.1f} FPS)!")
