"""
Ray Service Discovery

Enables automatic discovery of Ray head nodes and worker initialization
using Bonjour/Zeroconf service discovery.
"""

import ray
from zeroconf import Zeroconf, ServiceInfo, ServiceBrowser, ServiceStateChange
import socket
import json
import logging
import threading
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import psutil
from pathlib import Path

from .ray_debug import RayDebugMonitor

logger = logging.getLogger(__name__)

@dataclass
class RayNodeInfo:
    """Information about a Ray node."""
    node_id: str
    host: str
    port: int
    is_head: bool
    redis_port: int
    dashboard_port: Optional[int]
    resources: Dict[str, float]
    last_heartbeat: datetime

class RayServiceDiscovery:
    """Discovers and connects to Ray clusters using Zeroconf."""
    
    SERVICE_TYPE = "_ray-cluster._tcp.local."
    
    def __init__(self, 
                 node_id: Optional[str] = None,
                 monitor: Optional[RayDebugMonitor] = None):
        """Initialize Ray service discovery."""
        self.node_id = node_id or socket.gethostname()
        self.zeroconf = Zeroconf()
        self.monitor = monitor or RayDebugMonitor()
        self.nodes: Dict[str, RayNodeInfo] = {}
        self.browser = None
        self.info = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        
    def start_head_node(self, 
                       port: int = 6379,
                       dashboard_port: int = 8265,
                       resources: Optional[Dict[str, float]] = None) -> None:
        """Start a Ray head node and advertise it via Zeroconf."""
        try:
            # Initialize Ray head node
            ray.init(
                _system_config={"worker_port": port},  # Use _system_config for port configuration
                include_dashboard=True,
                dashboard_port=dashboard_port,
                resources=resources,
                num_cpus=None  # Let Ray auto-detect
            )
            
            # Get node info
            node_info = ray.nodes()[0]
            
            # Register service
            self.info = ServiceInfo(
                self.SERVICE_TYPE,
                f"{self.node_id}.{self.SERVICE_TYPE}",
                addresses=[socket.inet_aton(socket.gethostbyname(socket.gethostname()))],
                port=port,
                properties={
                    'node_id': self.node_id,
                    'is_head': 'true',
                    'redis_port': str(port),
                    'dashboard_port': str(dashboard_port),
                    'resources': json.dumps(resources or {}),
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            self.zeroconf.register_service(self.info)
            
            # Start monitoring
            if self.monitor:
                self.monitor.start()
            
            logger.info(f"Started Ray head node on port {port}")
            
            # Start discovery for other nodes
            self._start_discovery()
            
        except Exception as e:
            logger.error(f"Failed to start Ray head node: {e}")
            self.stop()
            raise
            
    def _start_discovery(self) -> None:
        """Start browsing for Ray services."""
        self.browser = ServiceBrowser(
            self.zeroconf,
            self.SERVICE_TYPE,
            handlers=[self._handle_service_state_change]
        )
        
    def _handle_service_state_change(self,
                                   zeroconf: Zeroconf,
                                   service_type: str,
                                   name: str,
                                   state_change: ServiceStateChange) -> None:
        """Handle service discovery events."""
        if state_change is ServiceStateChange.Added:
            info = zeroconf.get_service_info(service_type, name)
            if info:
                try:
                    properties = {
                        k.decode(): v.decode() if isinstance(v, bytes) else v
                        for k, v in info.properties.items()
                    }
                    
                    node = RayNodeInfo(
                        node_id=properties['node_id'],
                        host=socket.inet_ntoa(info.addresses[0]),
                        port=info.port,
                        is_head=properties['is_head'].lower() == 'true',
                        redis_port=int(properties['redis_port']),
                        dashboard_port=int(properties.get('dashboard_port', 0)) or None,
                        resources=json.loads(properties.get('resources', '{}')),
                        last_heartbeat=datetime.fromisoformat(properties['timestamp'])
                    )
                    
                    with self._lock:
                        self.nodes[node.node_id] = node
                    
                    # If we're not a head node and we discover one, connect as worker
                    if node.is_head and not ray.is_initialized():
                        self._connect_as_worker(node)
                        
                    logger.info(f"Discovered Ray node: {node.node_id}")
                    
                except Exception as e:
                    logger.error(f"Error processing discovered node: {e}")
                    
        elif state_change is ServiceStateChange.Removed:
            node_id = name.replace(f".{service_type}", "")
            with self._lock:
                if node_id in self.nodes:
                    del self.nodes[node_id]
                    logger.info(f"Ray node left: {node_id}")
                    
    def _connect_as_worker(self, head_node: RayNodeInfo) -> None:
        """Connect to a head node as a worker."""
        try:
            # Initialize Ray in worker mode
            ray.init(
                address=f"ray://{head_node.host}:{head_node.redis_port}",
                runtime_env={"working_dir": "."}, # Include current directory
                resources=self._get_worker_resources()
            )
            
            logger.info(f"Connected to Ray head node at {head_node.host}:{head_node.redis_port}")
            
            # Start monitoring if available
            if self.monitor:
                self.monitor.start()
                
        except Exception as e:
            logger.error(f"Failed to connect as worker: {e}")
            
    def _get_worker_resources(self) -> Dict[str, float]:
        """Get available resources for worker node."""
        resources = {}
        
        try:
            cpu_count = psutil.cpu_count()
            memory = psutil.virtual_memory()
            
            resources.update({
                "CPU": max(cpu_count - 1, 1),  # Leave one CPU for system
                "memory": memory.available / (1024 * 1024 * 1024),  # Convert to GB
                "node_type": "worker",
                "custom_resource": 1.0  # Example custom resource
            })
            
        except Exception as e:
            logger.warning(f"Error getting worker resources: {e}")
            
        return resources
    
    def start_worker_node(self) -> None:
        """Start as a worker node and discover head nodes."""
        # Start service discovery
        self._start_discovery()
        
        # Wait for head node discovery
        while not self._stop_event.is_set():
            head_nodes = [n for n in self.nodes.values() if n.is_head]
            if head_nodes:
                self._connect_as_worker(head_nodes[0])
                break
            time.sleep(1)
            
    def stop(self) -> None:
        """Stop service discovery and Ray processes."""
        self._stop_event.set()
        
        if self.info:
            self.zeroconf.unregister_service(self.info)
            
        if self.browser:
            self.browser.cancel()
            
        self.zeroconf.close()
        
        if ray.is_initialized():
            ray.shutdown()
            
        if self.monitor:
            self.monitor.stop()
            
        logger.info("Stopped Ray service discovery") 