"""GIS export utilities for GeoTIFF and QGIS compatibility."""

import numpy as np


def export_geotiff(array, output_path, transform=None, crs=None,
                   dtype='float32'):
    """Export numpy array as GeoTIFF.
    
    Args:
        array: [C, H, W] or [H, W] numpy array
        output_path: path to output .tif file
        transform: affine transform (optional)
        crs: coordinate reference system (optional)
        dtype: output data type
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS

        if array.ndim == 2:
            array = array[np.newaxis, :, :]

        C, H, W = array.shape

        if transform is None:
            transform = from_bounds(0, 0, W, H, W, H)
        if crs is None:
            crs = CRS.from_epsg(4326)  # WGS84 default

        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=H, width=W, count=C,
            dtype=dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            for i in range(C):
                dst.write(array[i], i + 1)

        print(f"Exported GeoTIFF: {output_path}")
    except ImportError:
        print("rasterio not available. Saving as .npy instead.")
        np.save(output_path.replace('.tif', '.npy'), array)


def compute_confidence_map(pred, R_init):
    """Compute per-pixel confidence based on residual magnitude.
    
    High confidence where residual ΔR is small (physics estimate was good).
    Low confidence where residual is large (network corrected significantly).
    
    Returns:
        confidence: [1, H, W] in [0, 1]
    """
    residual = np.abs(pred - R_init)
    residual = np.mean(residual, axis=0)  # Average across bands
    
    # Normalize to [0, 1] with sigmoid
    confidence = 1.0 / (1.0 + np.exp(5 * (residual - 0.3)))
    return confidence[np.newaxis]
