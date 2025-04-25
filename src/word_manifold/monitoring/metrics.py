"""
Metrics collection and monitoring for word-manifold system.

Provides metrics collection, Prometheus integration, and Ray dashboard integration.
"""

import ray
from ray.util import metrics
from typing import Dict, Any, Optional
import logging
import psutil
import time
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System-level metrics."""
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_io_counters: Dict[str, int]
    timestamp: datetime

@ray.remote
class MetricsCollector:
    """Collects and aggregates metrics from all system components."""
    
    def __init__(self):
        # Register metrics with Ray
        self.visualization_requests = metrics.Counter(
            "visualization_requests_total",
            description="Total number of visualization requests"
        )
        self.visualization_errors = metrics.Counter(
            "visualization_errors_total",
            description="Total number of visualization errors"
        )
        self.processing_time = metrics.Histogram(
            "visualization_processing_seconds",
            description="Time spent processing visualizations",
            boundaries=[0.1, 0.5, 1.0, 2.0, 5.0]  # buckets in seconds
        )
        self.worker_load = metrics.Gauge(
            "worker_load",
            description="Current worker load percentage"
        )
        self.cache_size = metrics.Gauge(
            "cache_size",
            description="Current cache size"
        )
        self.active_workers = metrics.Gauge(
            "active_workers",
            description="Number of active workers"
        )
        
        # Initialize internal state
        self.start_time = time.time()
        self.last_metrics: Optional[SystemMetrics] = None
    
    def record_request(self, duration: float, error: bool = False):
        """Record a visualization request."""
        self.visualization_requests.inc()
        if error:
            self.visualization_errors.inc()
        self.processing_time.observe(duration)
    
    def update_worker_metrics(self, num_workers: int, avg_load: float):
        """Update worker-related metrics."""
        self.active_workers.set(num_workers)
        self.worker_load.set(avg_load)
    
    def update_cache_metrics(self, size: int):
        """Update cache-related metrics."""
        self.cache_size.set(size)
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect system-level metrics."""
        try:
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory().percent
            disk = psutil.disk_usage('/').percent
            network = psutil.net_io_counters()._asdict()
            
            metrics = SystemMetrics(
                cpu_percent=cpu,
                memory_percent=memory,
                disk_usage_percent=disk,
                network_io_counters=network,
                timestamp=datetime.now()
            )
            
            self.last_metrics = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return self.last_metrics if self.last_metrics else SystemMetrics(
                cpu_percent=0,
                memory_percent=0,
                disk_usage_percent=0,
                network_io_counters={},
                timestamp=datetime.now()
            )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        system_metrics = self.collect_system_metrics()
        
        return {
            'system': {
                'cpu_percent': system_metrics.cpu_percent,
                'memory_percent': system_metrics.memory_percent,
                'disk_usage_percent': system_metrics.disk_usage_percent,
                'network': system_metrics.network_io_counters
            },
            'visualization': {
                'total_requests': self.visualization_requests.get_value(),
                'error_rate': (
                    self.visualization_errors.get_value() / 
                    max(1, self.visualization_requests.get_value())
                ) * 100,
                'avg_processing_time': self.processing_time.get_value().mean
            },
            'workers': {
                'active_count': self.active_workers.get_value(),
                'avg_load': self.worker_load.get_value()
            },
            'cache': {
                'size': self.cache_size.get_value()
            },
            'uptime': time.time() - self.start_time
        }

@ray.remote
class MetricsExporter:
    """Exports metrics to various backends (Prometheus, CloudWatch, etc.)."""
    
    def __init__(self, collector: ray.ObjectRef, export_interval: int = 60):
        self.collector = collector
        self.export_interval = export_interval
        self.running = False
        
        # Initialize Prometheus client if available
        try:
            from prometheus_client import start_http_server, Gauge, Counter, Histogram
            self.prometheus_enabled = True
            
            # Create Prometheus metrics
            self.prom_cpu = Gauge('system_cpu_percent', 'System CPU usage')
            self.prom_memory = Gauge('system_memory_percent', 'System memory usage')
            self.prom_requests = Counter('visualization_requests_total', 'Total visualization requests')
            self.prom_errors = Counter('visualization_errors_total', 'Total visualization errors')
            self.prom_processing_time = Histogram(
                'visualization_processing_seconds',
                'Visualization processing time',
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
            )
            
            # Start Prometheus HTTP server
            start_http_server(8000)
            logger.info("Started Prometheus metrics server on port 8000")
            
        except ImportError:
            self.prometheus_enabled = False
            logger.warning("Prometheus client not available, metrics export disabled")
    
    def start(self):
        """Start metrics export loop."""
        self.running = True
        while self.running:
            try:
                # Get metrics from collector
                metrics = ray.get(self.collector.get_metrics_summary.remote())
                
                # Export to Prometheus if enabled
                if self.prometheus_enabled:
                    self.prom_cpu.set(metrics['system']['cpu_percent'])
                    self.prom_memory.set(metrics['system']['memory_percent'])
                    self.prom_requests.inc(metrics['visualization']['total_requests'])
                    self.prom_errors.inc(metrics['visualization']['error_rate'])
                    
                # Could add other export backends here (CloudWatch, etc.)
                
            except Exception as e:
                logger.error(f"Error exporting metrics: {e}")
            
            time.sleep(self.export_interval)
    
    def stop(self):
        """Stop metrics export."""
        self.running = False 