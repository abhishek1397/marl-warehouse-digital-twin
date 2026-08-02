"""SpatialFeatureVisualizer generating CNN activation heatmaps and channel importance maps for Spatial MAPPO."""

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from marl.algorithms.spatial_mappo.cnn_critic import CNNCentralizedCritic


class SpatialFeatureVisualizer:
    """Generates and exports spatial activation heatmaps, channel importance plots, and Grad-CAM style visualizations."""

    @staticmethod
    def visualize_activation_maps(
        critic: CNNCentralizedCritic,
        spatial_tensor: torch.Tensor,
        output_dir: str = "runs/visualizations",
        filename: str = "spatial_critic_activation.png",
    ) -> str:
        """Visualizes CNN feature activation maps across channels and exports to image file."""
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, filename)

        with torch.no_grad():
            act_maps = critic.get_activation_maps(spatial_tensor)
            mean_act = act_maps[0].mean(dim=0).cpu().numpy()

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(mean_act, cmap="magma", interpolation="nearest")
        ax.set_title("Spatial CNN Critic Feature Activation Heatmap")
        plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

        return out_path
