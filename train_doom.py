"""
DOOM-FlyWire Connectome Training Pipeline (1,000 Epochs + Early Stopping)
=========================================================================
Trains the 139,248-neuron Drosophila connectome policy head on ViZDoom (defend_the_center)
for up to 1,000 epochs with Early Stopping, reward shaping, and checkpointing.
"""

import os
import sys
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import vizdoom as vzd

from src.connectome_engine import ConnectomeEngine
from src.retinotopy_encoder import RetinotopyEncoder

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 6 Discrete Action Combinations for defend_the_center: [TURN_LEFT, TURN_RIGHT, ATTACK]
COMBINED_ACTIONS = [
    [1, 0, 0],  # 0: TURN_LEFT
    [0, 1, 0],  # 1: TURN_RIGHT
    [0, 0, 1],  # 2: ATTACK (stand and fire)
    [1, 0, 1],  # 3: TURN_LEFT + ATTACK (turn left while firing)
    [0, 1, 1],  # 4: TURN_RIGHT + ATTACK (turn right while firing)
    [0, 0, 0],  # 5: WAIT / FOCUS
]

class ConnectomePolicyHead(nn.Module):
    def __init__(self, num_dn: int = 1295, num_actions: int = 6):
        super().__init__()
        # Synaptic readout directly from Descending Neurons (DNs) into motor actions
        self.net = nn.Sequential(
            nn.Linear(num_dn, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, dn_activations: torch.Tensor) -> torch.Tensor:
        return self.net(dn_activations)

def train(
    max_epochs: int = 1000,
    patience: int = 50,
    min_delta: float = 0.5,
    min_epochs: int = 80,
    lr: float = 1e-3,
    entropy_coef: float = 0.01,
    checkpoint_dir: str = "models",
    device: str = "cuda"
):
    print("=" * 75)
    print("[BRAIN] STARTING DOOM-FLYWIRE 1,000-EPOCH CONNECTOME TRAINING")
    print(f"   Max Epochs     : {max_epochs}")
    print(f"   Early Stopping : Patience={patience}, MinDelta={min_delta}, MinEpochs={min_epochs}")
    print(f"   Action Space   : 6 Actions (Turn, Shoot, Combined Turn+Shoot, Wait)")
    print(f"   Device         : {device}")
    print("=" * 75)

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_doom_policy.pt")

    # 1. Initialize Biological Brain on GPU (139,248 neurons, 15,090,883 synapses)
    engine = ConnectomeEngine(device=device)
    encoder = RetinotopyEncoder(
        num_neurons=engine.num_neurons,
        optic_left_indices=engine.optic_left_indices,
        optic_right_indices=engine.optic_right_indices,
        t4t5_indices=engine.t4t5_indices,
        eye_res=32,
        device=device
    )

    n_dn = len(engine.dn_left_indices) + len(engine.dn_right_indices)
    num_actions = len(COMBINED_ACTIONS)
    policy = ConnectomePolicyHead(num_dn=n_dn, num_actions=num_actions).to(device)
    optimizer = optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)

    # 2. Setup ViZDoom Headless Game (Optimized 320x240 for high GPU throughput)
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "defend_the_center.cfg"))
    game.set_window_visible(False)
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    # Ensure game variables are registered
    try:
        game.add_available_game_variable(vzd.GameVariable.KILLCOUNT)
        game.add_available_game_variable(vzd.GameVariable.HEALTH)
    except Exception:
        pass
    game.init()

    # 3. Training State Tracking
    recent_rewards = deque(maxlen=20)
    recent_kills = deque(maxlen=20)
    best_avg_reward = -float("inf")
    patience_counter = 0

    t_train_start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        game.new_episode()
        try:
            game.send_game_command("sv_infiniteammo 1")
        except Exception:
            pass
        engine.reset_state()

        episode_reward = 0.0
        episode_kills = 0
        prev_health = 100
        prev_kills = 0

        log_probs = []
        entropies = []
        rewards = []

        step = 0
        while not game.is_episode_finished() and step < 200:
            state = game.get_state()
            if state is None:
                break

            # 1. Retinotopic Sensory Injection
            screen_rgb = state.screen_buffer
            I_sensory, visual_meta = encoder.encode(screen_rgb)

            # 2. Whole-Brain Biological Propagation (15M synapses)
            act, _ = engine.step(I_sensory)

            # 3. Policy Head Forward Pass on Descending Neurons
            dn_act = torch.cat([act[engine.dn_left_indices], act[engine.dn_right_indices]])
            logits = policy(dn_act)

            dist = Categorical(logits=logits)
            action_idx = dist.sample()
            log_probs.append(dist.log_prob(action_idx))
            entropies.append(dist.entropy())

            chosen_action = COMBINED_ACTIONS[action_idx.item()]

            # 4. ViZDoom Step (frame skip = 2)
            raw_r = game.make_action(chosen_action, 2)
            step += 1

            # Extract state metrics
            cur_kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT) if vzd.GameVariable.KILLCOUNT in game.get_available_game_variables() else 0
            cur_health = game.get_game_variable(vzd.GameVariable.HEALTH) if vzd.GameVariable.HEALTH in game.get_available_game_variables() else 100

            # 5. Domain-Specific Reward Shaping
            step_reward = 0.05  # Living / survival incentive

            # Kill reward (+15.0 per kill)
            if cur_kills > prev_kills:
                step_reward += 15.0 * (cur_kills - prev_kills)
                episode_kills = cur_kills
                prev_kills = cur_kills

            # Damage penalty
            if cur_health < prev_health:
                damage_taken = prev_health - cur_health
                step_reward -= 0.15 * damage_taken
                prev_health = cur_health

            # Attack combat shaping (reward shooting when motion detected)
            motion_lvl = visual_meta.get("motion_total", 0.0)
            is_shooting = chosen_action[2] == 1
            if is_shooting and motion_lvl > 0.003:
                step_reward += 0.2
            elif is_shooting and motion_lvl < 0.001:
                step_reward -= 0.01  # Minor penalty for spraying into empty darkness

            if game.is_player_dead():
                step_reward -= 5.0

            rewards.append(step_reward)
            episode_reward += step_reward

        # 6. Policy Gradient Update (Standardized REINFORCE with Entropy Regularization)
        if len(rewards) > 0:
            gamma = 0.99
            discounted_returns = []
            G = 0.0
            for r_t in reversed(rewards):
                G = r_t + gamma * G
                discounted_returns.insert(0, G)

            returns_tensor = torch.tensor(discounted_returns, dtype=torch.float32, device=device)
            if len(returns_tensor) > 1 and returns_tensor.std() > 1e-7:
                returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-7)

            policy_loss = []
            for lp, G_t, ent in zip(log_probs, returns_tensor, entropies):
                policy_loss.append(-lp * G_t - entropy_coef * ent)

            total_loss = torch.stack(policy_loss).sum()

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

        recent_rewards.append(episode_reward)
        recent_kills.append(episode_kills)
        avg_reward = float(np.mean(recent_rewards))
        avg_kills = float(np.mean(recent_kills))

        # 7. Checkpoint & Early Stopping Evaluation
        improved = False
        if avg_reward > best_avg_reward + min_delta:
            best_avg_reward = avg_reward
            patience_counter = 0
            improved = True
            torch.save({
                "epoch": epoch,
                "policy_state_dict": policy.state_dict(),
                "best_avg_reward": best_avg_reward,
                "avg_kills": avg_kills,
                "num_dn": n_dn,
                "num_actions": num_actions
            }, best_model_path)
        else:
            patience_counter += 1

        # Periodic logging
        if epoch % 10 == 0 or improved or epoch == 1:
            save_tag = "[BEST MODEL SAVED]" if improved else f"Patience: {patience_counter}/{patience}"
            elapsed_min = (time.perf_counter() - t_train_start) / 60.0
            print(f"Epoch {epoch:4d}/{max_epochs} | Ep Rew: {episode_reward:6.1f} | Avg Rew(20): {avg_reward:6.1f} | Kills: {int(episode_kills):2d} (Avg: {avg_kills:.1f}) | {save_tag} [{elapsed_min:.1f}m]")

        # Early Stopping Trigger (guaranteed at least min_epochs)
        if patience_counter >= patience and epoch >= min_epochs:
            print("=" * 75)
            print(f"[EARLY STOPPING TRIGGERED] at Epoch {epoch}!")
            print(f"   Model converged! No reward improvement for {patience} epochs.")
            print(f"   Best Avg Reward: {best_avg_reward:.2f} | Best Avg Kills: {avg_kills:.1f}")
            print(f"   Trained model saved to: {best_model_path}")
            print("=" * 75)
            break

    game.close()
    total_time = (time.perf_counter() - t_train_start) / 60.0
    print(f"[TRAINING COMPLETE] Finished in {total_time:.1f} minutes.")
    print(f"   Saved Checkpoint: {best_model_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000, help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--min-delta", type=float, default=0.5, help="Minimum reward delta for improvement")
    parser.add_argument("--min-epochs", type=int, default=80, help="Minimum epochs before early stopping")
    args = parser.parse_args()

    train(
        max_epochs=args.epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        min_epochs=args.min_epochs
    )
