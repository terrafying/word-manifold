#!/usr/bin/env python3
"""
Ray Worker Management Script

Manages remote Ray workers, including registration, health monitoring,
and resource configuration. Supports both local network and Ray Cloud workers.
"""

import ray
import click
import logging
import yaml
import time
import socket
import psutil
import sys
from pathlib import Path
from typing import Dict, Optional
import json
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WorkerManager:
    """Manages Ray worker nodes."""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self._load_config()
        self.worker_info: Dict[str, Dict] = {}
        
    def _load_config(self) -> Dict:
        """Load Ray configuration."""
        try:
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            sys.exit(1)
    
    def get_system_resources(self) -> Dict:
        """Get available system resources."""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024 ** 3),
            "disk_gb": psutil.disk_usage('/').total / (1024 ** 3),
            "gpu_count": self._get_gpu_count()
        }
    
    def _get_gpu_count(self) -> int:
        """Get number of available GPUs."""
        try:
            import torch
            return torch.cuda.device_count()
        except ImportError:
            return 0
    
    def register_worker(self, head_address: str, resources: Optional[Dict] = None) -> bool:
        """Register this machine as a Ray worker."""
        try:
            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)
            
            # Get system resources if not provided
            if not resources:
                resources = self.get_system_resources()
            
            # Initialize Ray worker
            ray.init(
                address=head_address,
                runtime_env=self.config['runtime_env'],
                resources={
                    "CPU": resources['cpu_count'],
                    "memory": resources['memory_gb'] * 1024 * 1024 * 1024,  # Convert to bytes
                    "GPU": resources['gpu_count']
                }
            )
            
            # Store worker information
            self.worker_info[hostname] = {
                "ip_address": ip_address,
                "resources": resources,
                "start_time": datetime.now(),
                "head_address": head_address
            }
            
            logger.info(f"Registered worker {hostname} ({ip_address}) with Ray cluster")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register worker: {e}")
            return False
    
    def start_health_monitor(self, interval: int = 60):
        """Start worker health monitoring."""
        try:
            while True:
                self._check_health()
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Health monitoring stopped")
    
    def _check_health(self):
        """Check worker health metrics."""
        try:
            hostname = socket.gethostname()
            if hostname not in self.worker_info:
                logger.warning("Worker not registered")
                return
            
            # Collect health metrics
            metrics = {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "network": psutil.net_io_counters()._asdict(),
                "timestamp": datetime.now().isoformat()
            }
            
            # Update worker info
            self.worker_info[hostname]["last_health_check"] = metrics
            
            # Log warnings for high resource usage
            if metrics["cpu_percent"] > 90:
                logger.warning("High CPU usage: %.1f%%", metrics["cpu_percent"])
            if metrics["memory_percent"] > 90:
                logger.warning("High memory usage: %.1f%%", metrics["memory_percent"])
            if metrics["disk_percent"] > 90:
                logger.warning("High disk usage: %.1f%%", metrics["disk_percent"])
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    def save_worker_info(self, output_path: Path):
        """Save worker information to file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(self.worker_info, f, indent=2, default=str)
            logger.info(f"Saved worker info to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save worker info: {e}")

@click.group()
def cli():
    """Ray worker management tools."""
    pass

@cli.command()
@click.option('--head-address', required=True, help='Ray head node address')
@click.option('--config', type=click.Path(exists=True, path_type=Path), default='ray_config.yaml', help='Path to Ray configuration')
@click.option('--cpu-limit', type=int, help='Limit CPU cores available to Ray')
@click.option('--memory-limit', type=float, help='Limit memory (GB) available to Ray')
@click.option('--gpu-limit', type=int, help='Limit GPUs available to Ray')
def register(head_address: str, config: Path, cpu_limit: Optional[int], memory_limit: Optional[float], gpu_limit: Optional[int]):
    """Register this machine as a Ray worker."""
    manager = WorkerManager(config)
    
    # Override system resources if limits provided
    resources = manager.get_system_resources()
    if cpu_limit:
        resources['cpu_count'] = min(cpu_limit, resources['cpu_count'])
    if memory_limit:
        resources['memory_gb'] = min(memory_limit, resources['memory_gb'])
    if gpu_limit:
        resources['gpu_count'] = min(gpu_limit, resources['gpu_count'])
    
    if manager.register_worker(head_address, resources):
        click.echo("Worker registered successfully")
        
        # Save worker info
        info_path = Path('worker_info.json')
        manager.save_worker_info(info_path)
        
        # Start health monitoring
        click.echo("Starting health monitoring (Ctrl+C to stop)")
        manager.start_health_monitor()
    else:
        sys.exit(1)

@cli.command()
@click.option('--config', type=click.Path(exists=True, path_type=Path), default='ray_config.yaml', help='Path to Ray configuration')
def status(config: Path):
    """Check worker status."""
    manager = WorkerManager(config)
    resources = manager.get_system_resources()
    
    click.echo("\nSystem Resources:")
    click.echo(f"CPU Cores: {resources['cpu_count']}")
    click.echo(f"Memory: {resources['memory_gb']:.1f} GB")
    click.echo(f"Disk: {resources['disk_gb']:.1f} GB")
    click.echo(f"GPUs: {resources['gpu_count']}")
    
    click.echo("\nCurrent Usage:")
    click.echo(f"CPU: {psutil.cpu_percent()}%")
    click.echo(f"Memory: {psutil.virtual_memory().percent}%")
    click.echo(f"Disk: {psutil.disk_usage('/').percent}%")
    
    # Check Ray connection
    try:
        if ray.is_initialized():
            click.echo("\nRay Status: Connected")
            click.echo(f"Ray Address: {ray.get_runtime_context().gcs_address}")
        else:
            click.echo("\nRay Status: Not connected")
    except Exception as e:
        click.echo(f"\nRay Status: Error - {e}")

if __name__ == '__main__':
    cli() 