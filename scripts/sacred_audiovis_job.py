"""Ray job script for sacred geometry audio-visual generation."""

import ray
from pathlib import Path
from typing import Dict, Any, Union
import json
import os
import yaml
import argparse
import numpy as np

from sacred_audiovis_complex import (
    AudioVisualParams, AudioParams, VisualizationParams,
    generate_complex_pattern
)

def convert_to_serializable(obj: Any) -> Any:
    """Convert numpy types to Python native types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.number):
        return obj.item()
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, Path):
        return str(obj)
    return obj

@ray.remote
class AudioVisualJob:
    """Ray actor for managing audio-visual generation job."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'data/audiovis/complex'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save job config
        with open(self.output_dir / 'job_config.json', 'w') as f:
            json.dump(convert_to_serializable(config), f, indent=2)
    
    def run(self) -> Dict[str, Any]:
        """Run the audio-visual generation job."""
        try:
            # Extract params from config
            params = self.config.get('params', {})
            
            # Generate patterns with specified configuration
            audio_path, frame_paths = generate_complex_pattern(
                pattern_type=params.get('pattern_type', 'mandala'),
                recursive_depth=params.get('recursive_depth', 3),
                batch_size=params.get('batch_size', 10)
            )
            
            result = {
                'status': 'success',
                'audio_path': str(audio_path),
                'frame_paths': frame_paths,
                'output_dir': str(self.output_dir),
                'config': self.config
            }
            
            # Convert to serializable types
            return convert_to_serializable(result)
            
        except Exception as e:
            error_result = {
                'status': 'error',
                'error': str(e),
                'output_dir': str(self.output_dir),
                'config': self.config
            }
            return convert_to_serializable(error_result)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def job_entrypoint(config: Dict[str, Any]) -> Dict[str, Any]:
    """Entrypoint function for Ray job submission."""
    # Initialize Ray if not already done
    if not ray.is_initialized():
        ray.init()
    
    try:
        # Create job actor
        job = AudioVisualJob.remote(config)
        
        # Run job and get results
        result = ray.get(job.run.remote())
        
        # Shutdown Ray
        ray.shutdown()
        
        return convert_to_serializable(result)
        
    except Exception as e:
        ray.shutdown()
        error_result = {
            'status': 'error',
            'error': str(e),
            'config': config
        }
        return convert_to_serializable(error_result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sacred geometry audio-visual generation job')
    parser.add_argument('--config', type=str, help='Path to YAML configuration file')
    args = parser.parse_args()
    
    if args.config:
        # Load config from YAML
        config = load_config(args.config)
    else:
        # Use default configuration
        config = {
            'params': {
                'pattern_type': 'mandala',
                'recursive_depth': 3,
                'batch_size': 10,
                'output_dir': 'data/audiovis/complex',
                'audio_params': {
                    'sample_rate': 44100,
                    'duration': 15.0,
                    'amplitude': 0.9
                },
                'visual_params': {
                    'width': 120,
                    'height': 60,
                    'density': 0.7,
                    'style': 'mystical',
                    'symmetry': 12,
                    'layers': 7
                }
            }
        }
    
    # Run job
    result = job_entrypoint(config)
    print(json.dumps(convert_to_serializable(result), indent=2)) 