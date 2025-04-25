"""
Ray Debug Utilities

Provides tools for debugging Ray clusters, service discovery, and monitoring.
Includes dashboard setup and service health checks.
"""

import ray
from ray import serve
import logging
import time
from typing import Dict, Optional, List, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import psutil
import json
from pathlib import Path
import requests
from functools import wraps
import threading
import queue

logger = logging.getLogger(__name__)

@dataclass
class RayServiceInfo:
    """Information about a Ray service."""
    name: str
    status: str
    host: str
    port: int
    pid: int
    cpu_percent: float
    memory_percent: float
    last_heartbeat: datetime
    metadata: Dict[str, Any]

class RayDebugMonitor:
    """Monitors Ray services and provides debugging information."""
    
    def __init__(self, dashboard_port: int = 8265, log_dir: Optional[Path] = None):
        self.dashboard_port = dashboard_port
        self.log_dir = log_dir or Path("logs/ray")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.services: Dict[str, RayServiceInfo] = {}
        self._monitor_thread = None
        self._stop_event = threading.Event()
        self._metrics_queue = queue.Queue()
        
    def start(self):
        """Start the Ray debug monitor."""
        if not ray.is_initialized():
            ray.init(
                dashboard_port=self.dashboard_port,
                include_dashboard=True,
                logging_level=logging.DEBUG
            )
            
        # Start monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        
        logger.info(f"Ray debug monitor started. Dashboard available at http://localhost:{self.dashboard_port}")
        
    def stop(self):
        """Stop the Ray debug monitor."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join()
        logger.info("Ray debug monitor stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self._stop_event.is_set():
            try:
                self._collect_metrics()
                self._process_metrics()
                self._check_service_health()
                time.sleep(1)  # Update frequency
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}", exc_info=True)
                
    def _collect_metrics(self):
        """Collect metrics from Ray services."""
        for service_name, service in self.services.items():
            try:
                process = psutil.Process(service.pid)
                metrics = {
                    'cpu_percent': process.cpu_percent(),
                    'memory_percent': process.memory_percent(),
                    'threads': len(process.threads()),
                    'connections': len(process.connections()),
                    'timestamp': datetime.now().isoformat()
                }
                self._metrics_queue.put((service_name, metrics))
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.warning(f"Could not collect metrics for {service_name}: {e}")
                
    def _process_metrics(self):
        """Process collected metrics."""
        while not self._metrics_queue.empty():
            try:
                service_name, metrics = self._metrics_queue.get_nowait()
                # Save metrics to log file
                log_file = self.log_dir / f"{service_name}_metrics.jsonl"
                with open(log_file, 'a') as f:
                    json.dump(metrics, f)
                    f.write('\n')
            except queue.Empty:
                break
                
    def _check_service_health(self):
        """Check health of registered services."""
        for service_name, service in list(self.services.items()):
            try:
                # Check if service is responding
                response = requests.get(
                    f"http://{service.host}:{service.port}/health",
                    timeout=1
                )
                if response.status_code == 200:
                    service.last_heartbeat = datetime.now()
                else:
                    logger.warning(f"Service {service_name} returned status {response.status_code}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Service {service_name} health check failed: {e}")
                
    def register_service(
        self,
        name: str,
        host: str,
        port: int,
        pid: int,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Register a service for monitoring."""
        self.services[name] = RayServiceInfo(
            name=name,
            status="starting",
            host=host,
            port=port,
            pid=pid,
            cpu_percent=0.0,
            memory_percent=0.0,
            last_heartbeat=datetime.now(),
            metadata=metadata or {}
        )
        logger.info(f"Registered service {name} at {host}:{port}")
        
    def unregister_service(self, name: str) -> None:
        """Unregister a service from monitoring."""
        if name in self.services:
            del self.services[name]
            logger.info(f"Unregistered service {name}")
            
    def get_service_info(self, name: str) -> Optional[RayServiceInfo]:
        """Get information about a registered service."""
        return self.services.get(name)
        
    def get_all_services(self) -> List[RayServiceInfo]:
        """Get information about all registered services."""
        return list(self.services.values())

def with_ray_monitoring(monitor: RayDebugMonitor):
    """Decorator to add Ray monitoring to a service."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get service info from the function
            service_name = func.__name__
            if hasattr(func, 'service_info'):
                service_info = func.service_info
                monitor.register_service(
                    name=service_info.get('name', service_name),
                    host=service_info.get('host', 'localhost'),
                    port=service_info.get('port'),
                    pid=service_info.get('pid', psutil.Process().pid),
                    metadata=service_info.get('metadata', {})
                )
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                if hasattr(func, 'service_info'):
                    monitor.unregister_service(service_info.get('name', service_name))
                    
        return wrapper
    return decorator

# Example usage:
"""
# Initialize monitor
monitor = RayDebugMonitor(dashboard_port=8265)
monitor.start()

@ray.remote
@with_ray_monitoring(monitor)
def my_service():
    my_service.service_info = {
        'name': 'my_service',
        'host': 'localhost',
        'port': 8000,
        'metadata': {'version': '1.0'}
    }
    # Service logic here
    
# Stop monitoring when done
monitor.stop()
""" 