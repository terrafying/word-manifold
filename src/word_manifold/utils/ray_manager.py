"""
Unified Ray Management

Provides centralized management of Ray clusters, including service discovery,
debugging, monitoring, and resource management.
"""

import ray
import logging
import asyncio
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import json
from datetime import datetime

from .ray_debug import RayDebugMonitor
from .ray_service import RayServiceDiscovery

logger = logging.getLogger(__name__)

class RayManager:
    """Centralized manager for Ray functionality."""
    
    def __init__(self,
                 node_id: Optional[str] = None,
                 log_dir: Optional[Path] = None,
                 dashboard_port: int = 8265):
        """Initialize Ray manager."""
        self.monitor = RayDebugMonitor(
            dashboard_port=dashboard_port,
            log_dir=log_dir
        )
        self.service_discovery = RayServiceDiscovery(
            node_id=node_id,
            monitor=self.monitor
        )
        self.is_head = False
        
    async def start(self,
                   mode: str = 'auto',
                   port: int = 6379,
                   resources: Optional[Dict[str, float]] = None) -> None:
        """
        Start Ray with service discovery and monitoring.
        
        Args:
            mode: One of 'auto', 'head', or 'worker'
            port: Port for Ray head node
            resources: Resource specifications
        """
        try:
            if mode == 'head':
                await self._start_head(port, resources)
            elif mode == 'worker':
                await self._start_worker(resources)
            else:  # auto mode
                try:
                    await self._start_head(port, resources)
                except Exception as e:
                    logger.info(f"Could not start as head node, switching to worker mode: {e}")
                    await self._start_worker(resources)
                    
        except Exception as e:
            logger.error(f"Failed to start Ray manager: {e}")
            await self.stop()
            raise
            
    async def _start_head(self, port: int, resources: Optional[Dict[str, float]] = None) -> None:
        """Start as head node."""
        self.service_discovery.start_head_node(
            port=port,
            dashboard_port=self.monitor.dashboard_port,
            resources=resources
        )
        self.is_head = True
        
    async def _start_worker(self, resources: Optional[Dict[str, float]] = None) -> None:
        """Start as worker node."""
        self.service_discovery.start_worker_node()
        self.is_head = False
        
    async def stop(self) -> None:
        """Stop Ray manager and all services."""
        self.service_discovery.stop()
        self.monitor.stop()
        
    def get_node_info(self) -> Dict[str, Any]:
        """Get information about the current node."""
        return {
            'is_head': self.is_head,
            'nodes': len(self.service_discovery.nodes),
            'resources': ray.available_resources() if ray.is_initialized() else {},
            'timestamp': datetime.now().isoformat()
        }
        
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get information about the entire cluster."""
        cluster_info = {
            'nodes': [
                {
                    'node_id': node.node_id,
                    'host': node.host,
                    'port': node.port,
                    'is_head': node.is_head,
                    'resources': node.resources,
                    'last_heartbeat': node.last_heartbeat.isoformat()
                }
                for node in self.service_discovery.nodes.values()
            ],
            'total_resources': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Aggregate resources
        for node in cluster_info['nodes']:
            for resource, amount in node['resources'].items():
                if resource in cluster_info['total_resources']:
                    cluster_info['total_resources'][resource] += amount
                else:
                    cluster_info['total_resources'][resource] = amount
                    
        return cluster_info
        
    async def wait_for_nodes(self, 
                           min_nodes: int,
                           timeout: float = 60.0) -> bool:
        """
        Wait for a minimum number of nodes to join.
        
        Args:
            min_nodes: Minimum number of nodes to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if minimum nodes joined, False if timeout
        """
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < timeout:
            if len(self.service_discovery.nodes) >= min_nodes:
                return True
            await asyncio.sleep(1)
        return False
        
    @staticmethod
    def is_ray_initialized() -> bool:
        """Check if Ray is initialized."""
        return ray.is_initialized()
        
    @staticmethod
    def get_runtime_context() -> Optional[ray.runtime_context.RuntimeContext]:
        """Get Ray runtime context if available."""
        try:
            return ray.get_runtime_context()
        except Exception:
            return None

# Global Ray manager instance
_ray_manager: Optional[RayManager] = None

def get_ray_manager(
    node_id: Optional[str] = None,
    log_dir: Optional[Path] = None,
    dashboard_port: int = 8265
) -> RayManager:
    """Get or create global Ray manager instance."""
    global _ray_manager
    if _ray_manager is None:
        _ray_manager = RayManager(
            node_id=node_id,
            log_dir=log_dir,
            dashboard_port=dashboard_port
        )
    return _ray_manager 