"""
SEN12MS-CR Dataset for Stage 1 MSKT pretraining.

Reads paired Sentinel-1 SAR + Sentinel-2 optical patches.
Uses only B3/G, B4/R, B8/NIR (LISS-IV compatible bands).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio


class SEN12MSCRDataset(Dataset):
    """SEN12MS-CR dataset for pretraining MSKT.

    Directory structure:
        root/
            s1/           # Sentinel-1 SAR patches
            s2/           # Sentinel-2 cloud-free optical patches
            s2_cloudy/    # Sentinel-2 cloudy optical patches (optional)

    Each patch: .tif file with multi-band raster.
    """

    # Official train/val/test splits (ROI seed directories)
    TRAIN_ROIS = list(range(1, 113))  # ROIs 1-112
    VAL_ROIS = list(range(113, 123))  # ROIs 113-122
    TEST_ROIS = list(range(123, 133))  # ROIs 123-132

    # Sentinel-2 band indices for LISS-IV compatibility
    # B3 (Green, 560nm), B4 (Red, 665nm), B8 (NIR, 842nm)
    S2_BAND_INDICES = [2, 3, 7]  # 0-indexed: B3=idx2, B4=idx3, B8=idx7

    def __init__(self, root, split='train', crop_size=256,
                 sar_bands=(0, 1), data_range=1.0):
        """
        Args:
            root: Path to SEN12MS-CR dataset
            split: 'train', 'val', or 'test'
            crop_size: Random crop size
            sar_bands: SAR band indices (VV, VH)
            data_range: Rescale factor
        """
        self.root = root
        self.split = split
        self.crop_size = crop_size
        self.sar_bands = list(sar_bands)
        self.data_range = data_range

        # Select ROIs based on split
        if split == 'train':
            rois = self.TRAIN_ROIS
        elif split == 'val':
            rois = self.VAL_ROIS
        else:
            rois = self.TEST_ROIS

        # Scan for patch files
        self.patches = []
        s1_dir = os.path.join(root, 's1')
        s2_dir = os.path.join(root, 's2')

        for roi in rois:
            roi_s1 = os.path.join(s1_dir, f'ROI_{roi:03d}')
            roi_s2 = os.path.join(s2_dir, f'ROI_{roi:03d}')
            if not os.path.exists(roi_s1) or not os.path.exists(roi_s2):
                continue
            for fname in sorted(os.listdir(roi_s1)):
                if fname.endswith('.tif'):
                    s1_path = os.path.join(roi_s1, fname)
                    s2_path = os.path.join(roi_s2, fname)
                    if os.path.exists(s2_path):
                        self.patches.append({
                            's1': s1_path,
                            's2': s2_path,
                        })

    def __len__(self):
        return len(self.patches)

    def _read_tif(self, path, bands=None):
        """Read a .tif file and return as numpy array."""
        with rasterio.open(path) as src:
            if bands is not None:
                data = src.read(bands)
            else:
                data = src.read()
        return data.astype(np.float32)

    def _preprocess_sar(self, sar):
        """Preprocess SAR: clip to [-25, 0] dB, rescale to [0, 1]."""
        sar = np.clip(sar, -25, 0)
        sar = (sar + 25) / 25.0
        return sar

    def _preprocess_optical(self, optical):
        """Preprocess optical: clip to [0, 10000], rescale to [0, 1]."""
        optical = np.clip(optical, 0, 10000)
        optical = optical / 10000.0
        return optical

    def _random_crop(self, *arrays):
        """Apply consistent random crop to all arrays."""
        _, h, w = arrays[0].shape
        if h <= self.crop_size or w <= self.crop_size:
            # Pad if necessary
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

    def _random_flip(self, *arrays):
        """Apply consistent random flip to all arrays."""
        if np.random.random() > 0.5:
            arrays = [a[:, :, ::-1].copy() for a in arrays]
        if np.random.random() > 0.5:
            arrays = [a[:, ::-1, :].copy() for a in arrays]
        return arrays

    def __getitem__(self, idx):
        patch = self.patches[idx]

        # Read SAR (2 bands: VV, VH)
        sar = self._read_tif(patch['s1'], bands=self.sar_bands)
        sar = self._preprocess_sar(sar)

        # Read Sentinel-2 optical (all 13 bands)
        s2_all = self._read_tif(patch['s2'])
        s2_all = self._preprocess_optical(s2_all)

        # Extract LISS-IV compatible bands (B3, B4, B8)
        optical = s2_all[self.S2_BAND_INDICES]

        # Random crop
        sar, optical = self._random_crop(sar, optical)

        # Random flip
        sar, optical = self._random_flip(sar, optical)

        # To tensors
        sar = torch.from_numpy(sar).float()
        optical = torch.from_numpy(optical).float()
        s2_13band = torch.from_numpy(s2_all).float()

        return {
            'input': torch.cat([optical, sar], dim=0),  # [5, H, W]
            'target': s2_13band,  # [13, H, W]
            'optical': optical,   # [3, H, W]
            'sar': sar,          # [2, H, W]
        }
