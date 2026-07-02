"""Modality detection and sensor routing for multi-sensor support."""

import numpy as np


class SensorRouter:
    """Automatic sensor identification and routing.
    
    Detects input sensor type and routes to the appropriate model
    and preprocessing pipeline.
    """
    
    SENSOR_PROFILES = {
        'liss4': {
            'bands': 3,
            'resolution': 5.8,
            'band_names': ['Green', 'Red', 'NIR'],
            'wavelengths': [0.555, 0.650, 0.815],
        },
        'sentinel2': {
            'bands': 13,
            'resolution': 10,
            'band_names': ['B01', 'B02', 'B03', 'B04', 'B05', 'B06',
                           'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12'],
        },
        'sentinel1': {
            'bands': 2,
            'resolution': 10,
            'band_names': ['VV', 'VH'],
        },
        'landsat8': {
            'bands': 11,
            'resolution': 30,
        },
    }

    def detect_sensor(self, metadata):
        """Detect sensor type from metadata.
        
        Args:
            metadata: dict with keys like 'bands', 'resolution', 'crs'
            
        Returns:
            sensor_type: str
        """
        num_bands = metadata.get('bands', 0)
        resolution = metadata.get('resolution', 0)

        if num_bands == 3 and 5 <= resolution <= 7:
            return 'liss4'
        elif num_bands == 13:
            return 'sentinel2'
        elif num_bands == 2 and 5 <= resolution <= 25:
            return 'sentinel1'
        elif num_bands == 11:
            return 'landsat8'
        else:
            return 'unknown'

    def get_model_config(self, sensor_type):
        """Get appropriate model configuration for sensor type."""
        configs = {
            'liss4': {
                'model': 'psprnet',
                'in_channels': 5,  # 3 optical + 2 SAR
                'out_channels': 3,
                'preprocessing': 'liss4_standard',
            },
            'sentinel2': {
                'model': 'ecrformer',
                'in_channels': 15,  # 13 optical + 2 SAR
                'out_channels': 13,
                'preprocessing': 'sentinel2_standard',
            },
        }
        return configs.get(sensor_type, configs['liss4'])

    def route(self, metadata):
        """Full routing pipeline: detect → configure → preprocess.
        
        Args:
            metadata: dict with sensor metadata
            
        Returns:
            dict with sensor_info, model_config, preprocessing
        """
        sensor = self.detect_sensor(metadata)
        profile = self.SENSOR_PROFILES.get(sensor, {})
        model_config = self.get_model_config(sensor)

        return {
            'sensor_type': sensor,
            'profile': profile,
            'model_config': model_config,
        }
