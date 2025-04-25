"""
Ray Cloud integration and autoscaling configuration.

Provides utilities for deploying and managing Ray clusters on Ray Cloud,
with autoscaling and monitoring integration.
"""

import ray
from ray import air, tune
from ray.air import session
from ray.air.config import ScalingConfig
from ray.train.torch import TorchTrainer
from typing import Dict, Any, Optional, List
import yaml
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class RayCloudManager:
    """Manages Ray Cloud deployment and scaling."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        project_id: Optional[str] = None,
        cluster_name: str = "wordmanifold-cluster"
    ):
        self.config_path = config_path or "ray_config.yaml"
        self.project_id = project_id or os.getenv("RAY_PROJECT_ID")
        self.cluster_name = cluster_name
        
        if not self.project_id:
            raise ValueError("Ray Cloud project ID not found. Set RAY_PROJECT_ID env var.")
        
        self.config = self._load_config()
        self._validate_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load Ray cluster configuration."""
        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            raise
    
    def _validate_config(self):
        """Validate Ray cluster configuration."""
        required_fields = ['head_node', 'worker_nodes', 'system_config']
        missing = [f for f in required_fields if f not in self.config]
        if missing:
            raise ValueError(f"Missing required config fields: {missing}")
    
    def get_scaling_config(self) -> ScalingConfig:
        """Create Ray AIR scaling configuration."""
        worker_config = self.config['visualization_workers']
        return ScalingConfig(
            num_workers=worker_config['initial_workers'],
            use_gpu=False,  # Configure based on needs
            resources_per_worker={
                "CPU": worker_config['resources_per_worker']['CPU'],
                "memory": worker_config['resources_per_worker']['memory']
            },
            scaling_config={
                "min_workers": worker_config['min_workers'],
                "max_workers": worker_config['max_workers'],
                "target_utilization_fraction": 0.8,
                "seconds_to_wait_between_scales": 60
            }
        )
    
    def setup_cloud_cluster(self) -> bool:
        """Initialize and configure Ray Cloud cluster."""
        try:
            # Initialize Ray with cloud configuration
            ray.init(
                address=f"anyscale://{self.cluster_name}",
                runtime_env={
                    "working_dir": ".",
                    "pip": self.config['runtime_env']['pip'],
                    "env_vars": self.config['runtime_env']['env_vars']
                }
            )
            
            # Configure autoscaling
            scaling_config = self.get_scaling_config()
            
            # Set up monitoring
            if self.config['monitoring']['enabled']:
                from ray.util.metrics import Counter, Gauge, Histogram
                metrics_config = {
                    "metrics_export_port": self.config['monitoring']['metrics_export_port'],
                    "prometheus": self.config['monitoring']['prometheus_enabled'],
                    "dashboard": self.config['monitoring']['dashboard_enabled']
                }
                ray.init(metrics_export_port=metrics_config["metrics_export_port"])
            
            logger.info(f"Successfully initialized Ray Cloud cluster: {self.cluster_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Ray Cloud cluster: {e}")
            return False
    
    def setup_autoscaling(self, metrics_collector: ray.ObjectRef):
        """Configure autoscaling based on metrics."""
        try:
            def scaling_func(metrics: Dict[str, Any]) -> int:
                """Dynamic scaling based on metrics."""
                worker_metrics = metrics['workers']
                current_load = worker_metrics['avg_load']
                current_workers = worker_metrics['active_count']
                
                # Scale up if load is high
                if current_load > 80:
                    return min(
                        current_workers + 2,
                        self.config['visualization_workers']['max_workers']
                    )
                # Scale down if load is low
                elif current_load < 30 and current_workers > self.config['visualization_workers']['min_workers']:
                    return current_workers - 1
                return current_workers
            
            # Set up autoscaling policy
            tune.run(
                scaling_config=self.get_scaling_config(),
                scaling_strategy=scaling_func,
                stop_condition={"training_iteration": 999999}  # Run indefinitely
            )
            
            logger.info("Autoscaling configured successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure autoscaling: {e}")
            return False
    
    def setup_cloud_monitoring(self):
        """Configure cloud monitoring and metrics export."""
        try:
            from ray.util.metrics import Counter, Gauge, Histogram
            from ray.util.metrics.metrics_exporter import PrometheusMetricsExporter
            
            # Configure Prometheus metrics export
            if self.config['monitoring']['prometheus_enabled']:
                exporter = PrometheusMetricsExporter(
                    port=self.config['monitoring']['metrics_export_port'],
                    addr="0.0.0.0"  # Allow external access
                )
                ray.util.metrics.metrics_export_handler.set_exporter(exporter)
            
            # Enable Ray dashboard
            if self.config['monitoring']['dashboard_enabled']:
                ray.init(
                    dashboard_host="0.0.0.0",
                    dashboard_port=self.config['monitoring']['dashboard_port']
                )
            
            logger.info("Cloud monitoring configured successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure cloud monitoring: {e}")
            return False
    
    def shutdown(self):
        """Shutdown Ray Cloud cluster."""
        try:
            if ray.is_initialized():
                ray.shutdown()
                logger.info("Ray Cloud cluster shutdown complete")
        except Exception as e:
            logger.error(f"Error during cluster shutdown: {e}")

def init_ray_cloud(config_path: Optional[str] = None) -> RayCloudManager:
    """Initialize Ray Cloud deployment."""
    manager = RayCloudManager(config_path=config_path)
    if manager.setup_cloud_cluster():
        if manager.setup_cloud_monitoring():
            return manager
    raise RuntimeError("Failed to initialize Ray Cloud") 