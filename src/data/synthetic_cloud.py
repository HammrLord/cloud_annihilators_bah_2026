"""
Synthetic cloud generation for creating training pairs.

Since real paired cloudy/clear LISS-IV images are rare,
synthetic clouds can be generated from clear images using
physics-based atmospheric scattering model.
"""

import numpy as np
import torch


class SyntheticCloudGenerator:
    """Generate synthetic cloudy images from clear images.
    
    Uses the Atmospheric Scattering Model (ASM):
        I(x) = T(x) * R(x) + (1 - T(x)) * A
    where:
        I = observed (cloudy) image
        R = scene radiance (clear image)
        T = transmission map (cloud opacity)
        A = atmospheric light
    """

    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)

    def generate_cloud_mask(self, h, w, num_blobs=10, max_radius_frac=0.3):
        """Generate a smooth cloud mask using Gaussian blobs."""
        mask = np.zeros((h, w), dtype=np.float32)

        for _ in range(num_blobs):
            cx = self.rng.randint(0, w)
            cy = self.rng.randint(0, h)
            radius = self.rng.uniform(10, int(max(h, w) * max_radius_frac))
            intensity = self.rng.uniform(0.3, 1.0)

            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            blob = intensity * np.exp(-dist ** 2 / (2 * radius ** 2))
            mask = np.maximum(mask, blob)

        return np.clip(mask, 0, 1)

    def apply_asm(self, clear_image, cloud_mask, atmospheric_light=None):
        """Apply Atmospheric Scattering Model.
        
        Args:
            clear_image: [C, H, W] — clear scene radiance
            cloud_mask: [H, W] — cloud opacity (0=clear, 1=thick cloud)
            atmospheric_light: [C] — atmospheric light per band (optional)
            
        Returns:
            cloudy_image: [C, H, W] — synthetic cloudy image
            transmission: [1, H, W] — transmission map
        """
        C, H, W = clear_image.shape

        # Generate atmospheric light if not provided
        if atmospheric_light is None:
            # A is typically bright (near 1.0 for each band)
            atmospheric_light = self.rng.uniform(0.7, 1.0, size=(C,)).astype(np.float32)

        # Transmission: T = exp(-beta * cloud_mask)
        # beta controls cloud thickness
        beta = self.rng.uniform(1.0, 4.0)
        transmission = np.exp(-beta * cloud_mask).astype(np.float32)  # [H, W]

        # Apply ASM: I = T * R + (1 - T) * A
        A = atmospheric_light[:, None, None]  # [C, 1, 1]
        T = transmission[None, :, :]          # [1, H, W]

        cloudy = T * clear_image + (1 - T) * A
        cloudy = np.clip(cloudy, 0, 1).astype(np.float32)

        return cloudy, transmission[None]  # [C, H, W], [1, H, W]

    def generate_pair(self, clear_image, num_clouds_range=(3, 15)):
        """Generate a synthetic cloudy/clear pair.
        
        Args:
            clear_image: [C, H, W] — clear image in [0, 1]
            num_clouds_range: tuple — range of number of cloud blobs
            
        Returns:
            cloudy: [C, H, W]
            clear: [C, H, W] (unchanged)
            transmission: [1, H, W]
            cloud_mask: [H, W]
        """
        C, H, W = clear_image.shape
        num_clouds = self.rng.randint(*num_clouds_range)
        cloud_mask = self.generate_cloud_mask(H, W, num_blobs=num_clouds)

        cloudy, transmission = self.apply_asm(clear_image, cloud_mask)

        return cloudy, clear_image, transmission, cloud_mask

    def generate_batch(self, clear_images, **kwargs):
        """Generate a batch of synthetic cloudy/clear pairs.
        
        Args:
            clear_images: [B, C, H, W] — batch of clear images
            
        Returns:
            dict with cloudy, clear, transmission, cloud_mask
        """
        B, C, H, W = clear_images.shape
        cloudy_list = []
        trans_list = []
        mask_list = []

        for i in range(B):
            clear_np = clear_images[i].numpy()
            cloudy, _, trans, mask = self.generate_pair(clear_np, **kwargs)
            cloudy_list.append(cloudy)
            trans_list.append(trans)
            mask_list.append(mask)

        return {
            'cloudy': torch.from_numpy(np.stack(cloudy_list)).float(),
            'clear': clear_images,
            'transmission': torch.from_numpy(np.stack(trans_list)).float(),
            'cloud_mask': torch.from_numpy(np.stack(mask_list)).float(),
        }
