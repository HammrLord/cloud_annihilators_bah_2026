"""Visualization utilities for cloud removal results."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def plot_comparison(cloudy, pred, target, title='Cloud Removal Result',
                    save_path=None):
    """Plot side-by-side comparison: cloudy → predicted → ground truth.
    
    Args:
        cloudy: [3, H, W] or [H, W, 3] — cloudy image
        pred: [3, H, W] or [H, W, 3] — predicted cloud-free
        target: [3, H, W] or [H, W, 3] — ground truth
        title: plot title
        save_path: path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, img, label in zip(axes, [cloudy, pred, target],
                               ['Cloudy', 'Predicted', 'Ground Truth']):
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))
        if img.shape[-1] == 1:
            img = img.squeeze(-1)
        if img.ndim == 2:
            ax.imshow(img, cmap='gray')
        else:
            # RGB (bands 2,1,0 for G,R,NIR -> R,G,B display)
            display = img[:, :, ::-1]
            display = np.clip(display, 0, 1)
            ax.imshow(display)
        ax.set_title(label)
        ax.axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def plot_error_map(pred, target, title='Error Map', save_path=None):
    """Plot per-band error map."""
    error = np.abs(pred - target)
    if error.ndim == 3:
        error = np.mean(error, axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(error, cmap='hot', vmin=0, vmax=0.3)
    ax.set_title(title)
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def plot_ndvi_comparison(pred, target, title='NDVI Comparison', save_path=None):
    """Plot NDVI maps side by side."""
    eps = 1e-6
    ndvi_pred = (pred[2] - pred[1]) / (pred[2] + pred[1] + eps)
    ndvi_target = (target[2] - target[1]) / (target[2] + target[1] + eps)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, ndvi, label in zip(axes,
                                [ndvi_pred, ndvi_target, ndvi_pred - ndvi_target],
                                ['Predicted NDVI', 'Target NDVI', 'NDVI Difference']):
        im = ax.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
        ax.set_title(label)
        ax.axis('off')
        plt.colorbar(im, ax=ax)

    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def plot_spectral_curves(pred, target, title='Spectral Profiles', save_path=None):
    """Plot spectral profiles for selected pixels."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    bands = ['Green', 'Red', 'NIR']
    for i, (ax, band) in enumerate(zip(axes, bands)):
        pred_profile = pred[i].flatten()
        target_profile = target[i].flatten()

        # Sample 100 random pixels
        idx = np.random.choice(len(pred_profile), min(100, len(pred_profile)),
                               replace=False)
        ax.plot(sorted(pred_profile[idx]), alpha=0.7, label='Predicted')
        ax.plot(sorted(target_profile[idx]), alpha=0.7, label='Target')
        ax.set_title(f'{band} Band')
        ax.set_xlabel('Pixel Index (sorted)')
        ax.set_ylabel('Reflectance')
        ax.legend()

    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig
