"""
DOOM-FlyWire Retinotopy & Compound Eye Encoder
==============================================
Maps ViZDoom screen buffer into Drosophila compound eye retinotopy:
- Downsamples RGB game frames into ommatidia resolution (e.g. 32x32 facets).
- Computes Elementary Motion Detectors (EMD / optical flow) for T4/T5 columnar neurons.
- Injects current into left/right optic lobes (R1-R8 photoreceptors, L1-L5, Mi1).
"""

from typing import Tuple, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F

class RetinotopyEncoder:
    def __init__(
        self,
        num_neurons: int = 139248,
        optic_left_indices: np.ndarray = None,
        optic_right_indices: np.ndarray = None,
        t4t5_indices: np.ndarray = None,
        eye_res: int = 32,
        device: str = "cuda"
    ):
        self.num_neurons = num_neurons
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.eye_res = eye_res
        
        # Subsample visual receptor pools for direct stimulation
        # (A subset of optic neurons act as first-order sensory recipients: ~2,000 per eye)
        n_sensors = min(2048, len(optic_left_indices) // 10)
        self.idx_left = torch.from_numpy(optic_left_indices[:n_sensors]).to(self.device)
        self.idx_right = torch.from_numpy(optic_right_indices[:n_sensors]).to(self.device)
        
        if t4t5_indices is not None and len(t4t5_indices) > 0:
            n_motion = min(1024, len(t4t5_indices))
            self.idx_motion = torch.from_numpy(t4t5_indices[:n_motion]).to(self.device)
        else:
            self.idx_motion = None

        self.prev_frame_gray = None

    def encode(
        self,
        frame_rgb: np.ndarray,
        labels: list = None,
        params: dict = None,
        contrast_gain: float = 2.0
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Pure neural retinotopy encoder:
        - Downsamples RGB into 32x32 ommatidia array.
        - Computes local spatial contrast and temporal frame-difference (Elementary Motion).
        - Injects sensory currents into Left Optic, Right Optic, and T4/T5 Columnar Neurons.
        - Maps detected visual targets (enemies) to biological Lobula Columnar (LC) hemifields.
        - Honors live optic lesions from neuroscientist sandbox console.
        """
        params = params or {}
        lesion_optic_l = params.get("lesion_optic_left", False)
        lesion_optic_r = params.get("lesion_optic_right", False)
        lesion_motion = params.get("lesion_motion_t4t5", False)

        # Convert to HWC format if CHW
        if frame_rgb.shape[0] == 3 and frame_rgb.ndim == 3:
            frame_rgb = np.transpose(frame_rgb, (1, 2, 0))

        # Downsample grayscale to compound eye ommatidia grid (eye_res x eye_res)
        gray = (0.299 * frame_rgb[:, :, 0] + 0.587 * frame_rgb[:, :, 1] + 0.114 * frame_rgb[:, :, 2]) / 255.0
        gray_tensor = torch.from_numpy(gray.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
        ommatidia = F.interpolate(gray_tensor, size=(self.eye_res, self.eye_res), mode="area").squeeze()

        # Compute Elementary Motion (temporal frame difference / optical flow)
        if self.prev_frame_gray is not None and not lesion_motion:
            motion_field = torch.abs(ommatidia - self.prev_frame_gray)
            motion_intensity = motion_field.flatten()
            motion_total = float(motion_field.mean().item())
        else:
            motion_intensity = torch.zeros_like(ommatidia.flatten())
            motion_total = 0.0

        self.prev_frame_gray = ommatidia.clone()

        # Compute spatial contrast deviation (local center-surround contrast)
        contrast_field = torch.abs(ommatidia - ommatidia.mean())

        # Split ommatidia into Left Hemifield and Right Hemifield
        mid_res = self.eye_res // 2
        left_eye_contrast = contrast_field[:, :mid_res].flatten() if not lesion_optic_l else torch.zeros_like(contrast_field[:, :mid_res].flatten())
        right_eye_contrast = contrast_field[:, mid_res:].flatten() if not lesion_optic_r else torch.zeros_like(contrast_field[:, mid_res:].flatten())

        # Receptive field mapping to Left and Right Optic Lobes
        n_left = len(self.idx_left)
        l_photoreceptors = left_eye_contrast.repeat((n_left // len(left_eye_contrast)) + 1)[:n_left]
        
        n_right = len(self.idx_right)
        r_photoreceptors = right_eye_contrast.repeat((n_right // len(right_eye_contrast)) + 1)[:n_right]

        # Initialize whole-brain sensory injection tensor
        I_sensory = torch.zeros(self.num_neurons, device=self.device, dtype=torch.float32)

        # Inject sensory currents via pure tensor operations
        if not lesion_optic_l:
            I_sensory[self.idx_left] = l_photoreceptors * (contrast_gain * 3.0)
        if not lesion_optic_r:
            I_sensory[self.idx_right] = r_photoreceptors * (contrast_gain * 3.0)

        # Map optical motion to T4/T5 columnar motion detection neurons
        if self.idx_motion is not None and not lesion_motion:
            n_mot = len(self.idx_motion)
            m_receptors = motion_intensity.repeat((n_mot // len(motion_intensity)) + 1)[:n_mot]
            I_sensory[self.idx_motion] = m_receptors * (contrast_gain * 2.5)

        # Target Tracking: Detect enemies on screen and stimulate corresponding visual hemifield
        has_target = False
        target_offset = 0.0
        target_locked = False
        enemy_count = 0

        ENEMY_TYPES = {'MarineChainsawVzd', 'Demon', 'Cacodemon', 'HellKnight', 'DoomImp', 'BaronOfHell', 'LostSoul', 'ZombieMan', 'ShotgunGuy'}
        if labels:
            enemies = [l for l in labels if l.object_name in ENEMY_TYPES]
            screen_w = frame_rgb.shape[1] if frame_rgb.ndim == 3 and frame_rgb.shape[0] != 3 else (frame_rgb.shape[2] if frame_rgb.ndim == 3 else 640)
            center_x = screen_w / 2.0

            # Filter out enemies in lesioned hemifields!
            if lesion_optic_l and lesion_optic_r:
                # 100% Total Blindness: cannot see ANY enemies anywhere on screen!
                visible_enemies = []
            elif lesion_optic_l:
                # Left eye blind: cannot see anything in the left hemifield
                visible_enemies = [e for e in enemies if (e.x + e.width / 2.0) > (center_x + 10.0)]
            elif lesion_optic_r:
                # Right eye blind: cannot see anything in the right hemifield
                visible_enemies = [e for e in enemies if (e.x + e.width / 2.0) < (center_x - 10.0)]
            else:
                visible_enemies = enemies

            enemy_count = len(visible_enemies)
            if visible_enemies:
                has_target = True
                closest = min(visible_enemies, key=lambda e: abs((e.x + e.width / 2.0) - center_x))
                target_cx = closest.x + closest.width / 2.0
                target_offset = (target_cx - center_x) / center_x
                target_locked = abs(target_cx - center_x) < (screen_w * 0.08)

                if target_locked:
                    # Bilateral surge on frontal target lock (both eyes stimulated)
                    if not lesion_optic_l:
                        I_sensory[self.idx_left] += contrast_gain * 3.5
                    if not lesion_optic_r:
                        I_sensory[self.idx_right] += contrast_gain * 3.5
                elif target_offset < -0.05 and not lesion_optic_l:
                    # Enemy is in Left Hemifield: stimulate Left Optic Lobe
                    I_sensory[self.idx_left] += (1.0 + abs(target_offset) * 2.0) * contrast_gain * 3.0
                elif target_offset > 0.05 and not lesion_optic_r:
                    # Enemy is in Right Hemifield: stimulate Right Optic Lobe
                    I_sensory[self.idx_right] += (1.0 + abs(target_offset) * 2.0) * contrast_gain * 3.0

        visual_meta = {
            "has_target": has_target,
            "target_offset": round(target_offset, 3),
            "target_locked": target_locked,
            "enemy_count": enemy_count,
            "optic_left_input": float(I_sensory[self.idx_left].mean().item()) if not lesion_optic_l else 0.0,
            "optic_right_input": float(I_sensory[self.idx_right].mean().item()) if not lesion_optic_r else 0.0,
            "motion_total": motion_total
        }

        return I_sensory, visual_meta



