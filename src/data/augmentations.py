"""
Data augmentation pipeline for cloud removal training.
Uses albumentations for consistent multi-modal augmentation.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_augmentations(crop_size=256):
    """Training augmentations (applied consistently to all modalities).
    
    Important:
        - NO color jitter (corrupts NDVI/NDWI computation)
        - NO geometric distortion (breaks SAR/optical co-registration)
        - Only spatial transforms that preserve pixel correspondence
    """
    return A.Compose([
        A.RandomCrop(height=crop_size, width=crop_size, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ], additional_targets={
        'image2': 'image',
        'image3': 'image',
        'mask': 'mask',
    })


def get_test_augmentations(crop_size=256):
    """Test augmentations: center crop only."""
    return A.Compose([
        A.CenterCrop(height=crop_size, width=crop_size, p=1.0),
    ], additional_targets={
        'image2': 'image',
        'image3': 'image',
        'mask': 'mask',
    })


def apply_augmentations(image, image2=None, image3=None, mask=None,
                        transform=None):
    """Apply augmentations consistently across all modalities.
    
    Args:
        image: [C, H, W] numpy array (optical)
        image2: [C, H, W] numpy array (SAR), optional
        image3: [C, H, W] numpy array (DEM), optional
        mask: [H, W] numpy array, optional
        transform: albumentations transform
        
    Returns:
        augmented arrays
    """
    # Albumentations expects [H, W, C] format
    inputs = {'image': image.transpose(1, 2, 0)}
    
    if image2 is not None:
        inputs['image2'] = image2.transpose(1, 2, 0)
    if image3 is not None:
        inputs['image3'] = image3.transpose(1, 2, 0)
    if mask is not None:
        inputs['mask'] = mask

    augmented = transform(**inputs)

    result = [augmented['image'].transpose(2, 0, 1)]
    if image2 is not None:
        result.append(augmented['image2'].transpose(2, 0, 1))
    if image3 is not None:
        result.append(augmented['image3'].transpose(2, 0, 1))
    if mask is not None:
        result.append(augmented['mask'])

    return result
