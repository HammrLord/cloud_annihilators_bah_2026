"""
LISS-IV Dataset for Stage 2 training.

Reads paired cloudy/clear LISS-IV images + Sentinel-1 SAR + SRTM DEM.
Computes spectral indices on-the-fly.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio


class LISS4Dataset(Dataset):
    """LISS-IV dataset for cloud removal training.

    Directory structure:
        root/
            cloudy/       # Cloudy LISS-IV images (.tif, 3 bands)
            clear/        # Cloud-free LISS-IV images (.tif, 3 bands)
            sar/          # Sentinel-1 SAR (.tif, 2 bands: VV, VH)
            dem/          # SRTM DEM (.tif, 1 band: elevation)
            masks/        # Cloud masks (optional, .tif)
    """

    def __init__(self, root, crop_size=256, use_sar=True, use_dem=True,
                 augment=True, data_range=1.0):
        self.root = root
        self.crop_size = crop_size
        self.use_sar = use_sar
        self.use_dem = use_dem
        self.augment = augment
        self.data_range = data_range

        # Scan for images
        cloudy_dir = os.path.join(root, 'cloudy')
        clear_dir = os.path.join(root, 'clear')

        self.samples = []
        for fname in sorted(os.listdir(cloudy_dir)):
            if not fname.endswith('.tif'):
                continue
            cloudy_path = os.path.join(cloudy_dir, fname)
            clear_path = os.path.join(clear_dir, fname)
            if not os.path.exists(clear_path):
                continue

            sample = {
                'cloudy': cloudy_path,
                'clear': clear_path,
                'name': fname,
            }

            # Optional: SAR
            sar_dir = os.path.join(root, 'sar')
            sar_path = os.path.join(sar_dir, fname)
            if use_sar and os.path.exists(sar_path):
                sample['sar'] = sar_path

            # Optional: DEM
            dem_dir = os.path.join(root, 'dem')
            dem_path = os.path.join(dem_dir, fname)
            if use_dem and os.path.exists(dem_path):
                sample['dem'] = dem_path

            # Optional: cloud mask
            mask_dir = os.path.join(root, 'masks')
            mask_path = os.path.join(mask_dir, fname)
            if os.path.exists(mask_path):
                sample['mask'] = mask_path

            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def _read_tif(self, path):
        with rasterio.open(path) as src:
            return src.read().astype(np.float32)

    def _preprocess_optical(self, data):
        """Rescale optical to [0, 1]."""
        data = np.clip(data, 0, 10000)
        return data / 10000.0

    def _preprocess_sar(self, data):
        """Rescale SAR to [0, 1]."""
        data = np.clip(data, -25, 0)
        return (data + 25) / 25.0

    def _compute_dem_features(self, elevation):
        """Compute slope and aspect from elevation using Sobel filters."""
        from scipy.ndimage import sobel

        slope_x = sobel(elevation, axis=1)
        slope_y = sobel(elevation, axis=2)
        slope = np.sqrt(slope_x ** 2 + slope_y ** 2)
        aspect = np.arctan2(slope_y, slope_x)

        # Normalize to [0, 1]
        slope = slope / (slope.max() + 1e-6)
        aspect = (aspect + np.pi) / (2 * np.pi)

        return np.stack([elevation / 3000.0, slope, aspect], axis=0)

    def _random_crop(self, *arrays):
        _, h, w = arrays[0].shape
        if h < self.crop_size or w < self.crop_size:
            pad_h = max(self.crop_size - h, 0)
            pad_w = max(self.crop_size - w, 0)
            arrays = [np.pad(a, ((0, 0), (0, pad_h), (0, pad_w)),
                            mode='reflect') for a in arrays]
            _, h, w = arrays[0].shape

        top = np.random.randint(0, h - self.crop_size + 1)
        left = np.random.randint(0, w - self.crop_size + 1)
        arrays = [a[:, top:top + self.crop_size, left:left + self.crop_size]
                  for a in arrays]
        return arrays

    def _random_augment(self, *arrays):
        if not self.augment:
            return arrays
        if np.random.random() > 0.5:
            arrays = [a[:, :, ::-1].copy() for a in arrays]
        if np.random.random() > 0.5:
            arrays = [a[:, ::-1, :].copy() for a in arrays]
        if np.random.random() > 0.5:
            k = np.random.randint(1, 4)
            arrays = [np.rot90(a, k, axes=(1, 2)).copy() for a in arrays]
        return arrays

    def __getitem__(self, idx):
        sample = self.samples[idx]

        cloudy = self._preprocess_optical(self._read_tif(sample['cloudy']))
        clear = self._preprocess_optical(self._read_tif(sample['clear']))

        arrays = [cloudy, clear]

        sar = None
        if 'sar' in sample:
            sar = self._preprocess_sar(self._read_tif(sample['sar']))
            arrays.append(sar)

        dem = None
        if 'dem' in sample:
            dem_raw = self._read_tif(sample['dem'])
            dem = self._compute_dem_features(dem_raw)
            arrays.append(dem)

        arrays = self._random_crop(*arrays)
        arrays = self._random_augment(*arrays)

        cloudy = torch.from_numpy(arrays[0]).float()
        clear = torch.from_numpy(arrays[1]).float()

        result = {
            'cloudy': cloudy,
            'clear': clear,
            'name': sample['name'],
        }

        idx = 2
        if sar is not None:
            result['sar'] = torch.from_numpy(arrays[idx]).float()
            idx += 1
        if dem is not None:
            result['dem'] = torch.from_numpy(arrays[idx]).float()
            idx += 1

        if 'mask' in sample:
            mask = self._read_tif(sample['mask'])
            mask = np.clip(mask, 0, 1).astype(np.float32)
            if mask.ndim == 3:
                mask = mask[0:1]
            result['mask'] = torch.from_numpy(mask).float()

        return result
