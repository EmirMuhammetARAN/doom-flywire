import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, Optional

class FlyConnectomeLoader:
    """
    High-performance graph loader for the adult Drosophila melanogaster
    whole-brain connectome (FlyWire 783 release, Nature 2024).
    Maps 139,248 neurons and 15,090,883 synapses into GPU-ready sparse tensors.
    """
    def __init__(self, data_dir: str = "data", cache_file: str = "data/connectome_cache.pt"):
        self.data_dir = data_dir
        self.cache_file = cache_file
        self.annotations_path = os.path.join(data_dir, "Supplemental_file1_neuron_annotations.tsv")
        self.connections_path = os.path.join(data_dir, "proofread_connections_783.feather")
        
        self.num_neurons = 0
        self.root_to_idx: Dict[int, int] = {}
        self.idx_to_root = np.array([], dtype=np.int64)
        self.coords: Optional[np.ndarray] = None
        self.cell_metadata: Optional[pd.DataFrame] = None
        self.adj_matrix: Optional[torch.Tensor] = None

    def load(self, force_recompute: bool = False, device: str = "cuda") -> Tuple[torch.Tensor, pd.DataFrame, np.ndarray]:
        """
        Loads the connectome graph. Loads from cached torch bundle in milliseconds,
        or parses raw TSV and Feather tables on first execution.
        """
        if not force_recompute and os.path.exists(self.cache_file):
            t0 = time.perf_counter()
            cached = torch.load(self.cache_file, map_location="cpu", weights_only=False)
            self.adj_matrix = cached["adj_matrix"]
            self.coords = cached["coords"]
            self.cell_metadata = cached["metadata"]
            self.idx_to_root = cached["idx_to_root"]
            self.root_to_idx = {int(rid): i for i, rid in enumerate(self.idx_to_root)}
            self.num_neurons = len(self.idx_to_root)
            elapsed = time.perf_counter() - t0
            print(f"[ConnectomeLoader] Cache loaded in {elapsed:.2f}s: {self.num_neurons:,} neurons, {self.adj_matrix._nnz():,} synapses.")
            return self.adj_matrix.to(device), self.cell_metadata, self.coords

        print("[ConnectomeLoader] Parsing raw tables (first-time initialization)...")
        if not os.path.exists(self.annotations_path):
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_path}")
        if not os.path.exists(self.connections_path):
            raise FileNotFoundError(f"Connections feather file not found: {self.connections_path}")

        # 1. Parse neuron annotations table
        t0 = time.perf_counter()
        ann_df = pd.read_csv(self.annotations_path, sep="\t", low_memory=False)
        self.num_neurons = len(ann_df)
        self.idx_to_root = ann_df["root_id"].values.astype(np.int64)
        self.root_to_idx = {int(rid): i for i, rid in enumerate(self.idx_to_root)}

        coords_x = np.where(pd.notnull(ann_df["soma_x"]) & (ann_df["soma_x"] > 0), ann_df["soma_x"], ann_df["pos_x"])
        coords_y = np.where(pd.notnull(ann_df["soma_y"]) & (ann_df["soma_y"] > 0), ann_df["soma_y"], ann_df["pos_y"])
        coords_z = np.where(pd.notnull(ann_df["soma_z"]) & (ann_df["soma_z"] > 0), ann_df["soma_z"], ann_df["pos_z"])
        self.coords = np.column_stack([coords_x, coords_y, coords_z]).astype(np.float32)

        self.cell_metadata = ann_df[[
            "root_id", "super_class", "cell_class", "cell_type", "top_nt", "top_nt_conf", "side"
        ]].copy()
        print(f"  -> {self.num_neurons:,} neurons and 3D spatial coordinates loaded ({time.perf_counter()-t0:.2f}s)")

        # 2. Parse synaptic connections
        t0 = time.perf_counter()
        conn_df = pd.read_feather(self.connections_path, columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"])
        grouped = conn_df.groupby(["pre_pt_root_id", "post_pt_root_id"], as_index=False)["syn_count"].sum()

        pre_mapped = grouped["pre_pt_root_id"].map(self.root_to_idx)
        post_mapped = grouped["post_pt_root_id"].map(self.root_to_idx)
        valid_mask = pre_mapped.notnull() & post_mapped.notnull()

        pre_indices = pre_mapped[valid_mask].values.astype(np.int64)
        post_indices = post_mapped[valid_mask].values.astype(np.int64)
        syn_weights = grouped["syn_count"][valid_mask].values.astype(np.float32)

        # 3. Apply physiological neurotransmitter polarity (excitatory vs inhibitory)
        nt_sign_map = {
            "acetylcholine": 1.0,
            "gaba": -1.0,
            "glutamate": -0.8,
            "dopamine": 0.5,
            "serotonin": 0.5,
            "octopamine": 0.8
        }
        neuron_signs = np.array([
            nt_sign_map.get(str(nt).lower(), 1.0) for nt in self.cell_metadata["top_nt"]
        ], dtype=np.float32)
        signed_weights = syn_weights * neuron_signs[pre_indices]

        # 4. Construct sparse COO tensor
        indices = torch.from_numpy(np.vstack([pre_indices, post_indices]))
        values = torch.from_numpy(signed_weights)
        self.adj_matrix = torch.sparse_coo_tensor(
            indices, values, (self.num_neurons, self.num_neurons), dtype=torch.float32
        ).coalesce()

        # Save cache
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        torch.save({
            "adj_matrix": self.adj_matrix,
            "coords": self.coords,
            "metadata": self.cell_metadata,
            "idx_to_root": self.idx_to_root
        }, self.cache_file)
        print(f"  -> Connectome cache written to {self.cache_file} ({time.perf_counter()-t0:.2f}s)")
        return self.adj_matrix.to(device), self.cell_metadata, self.coords

def load_flywire_graph(data_dir: str = "data", cache_file: str = "data/connectome_cache.pt", device: str = "cuda") -> Tuple[torch.Tensor, pd.DataFrame, np.ndarray]:
    """Convenience function to load or parse the FlyWire connectome graph."""
    loader = FlyConnectomeLoader(data_dir=data_dir, cache_file=cache_file)
    return loader.load(device=device)

if __name__ == "__main__":
    adj, meta, coords = load_flywire_graph()
