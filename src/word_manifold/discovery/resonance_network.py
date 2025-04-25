"""
Resonance Network Discovery

Enables automatic discovery of resonant mandala nodes and field synchronization
using Bonjour/Zeroconf. Allows mandalas to find and interact with each other
across the local network or Ray cluster.
"""

import ray
from zeroconf import Zeroconf, ServiceInfo, ServiceBrowser, ServiceStateChange
import socket
import json
import asyncio
import logging
from typing import Dict, Set, Optional, Callable, Any, List, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from word_manifold.visualization.engines.resonant_mandala import ResonanceField
else:
    ResonanceField = Any

from word_manifold.utils.ray_manager import get_ray_manager

logger = logging.getLogger(__name__)

# Initialize Ray manager
ray_manager = get_ray_manager(dashboard_port=8265)

@dataclass
class ResonanceNode:
    """Information about a resonance network node."""
    node_id: str
    host: str
    port: int
    active_fields: Set[ResonanceField]
    field_strength: float
    last_evolution: datetime
    metadata: Dict[str, Any]

class ResonanceDiscovery:
    """Discovers and tracks resonant mandala nodes."""
    
    SERVICE_TYPE = "_mandala-resonance._tcp.local."
    
    def __init__(self, node_id: Optional[str] = None):
        self.node_id = node_id or socket.gethostname()
        self.zeroconf = Zeroconf()
        self.nodes: Dict[str, ResonanceNode] = {}
        self.browser = None
        self.info = None
        self._lock = threading.Lock()
        
        # Initialize Ray service discovery
        self.ray_discovery = ray_manager.service_discovery
        
        # Callbacks for node events
        self._node_join_callback: Optional[Callable[[ResonanceNode], None]] = None
        self._node_leave_callback: Optional[Callable[[str], None]] = None
        self._field_update_callback: Optional[Callable[[str, Set[ResonanceField]], None]] = None
        
    @property
    def on_node_join(self) -> Optional[Callable[[ResonanceNode], None]]:
        return self._node_join_callback
        
    @on_node_join.setter
    def on_node_join(self, callback: Optional[Callable[[ResonanceNode], None]]) -> None:
        self._node_join_callback = callback
        
    @property
    def on_node_leave(self) -> Optional[Callable[[str], None]]:
        return self._node_leave_callback
        
    @on_node_leave.setter
    def on_node_leave(self, callback: Optional[Callable[[str], None]]) -> None:
        self._node_leave_callback = callback
        
    @property
    def on_field_update(self) -> Optional[Callable[[str, Set[ResonanceField]], None]]:
        return self._field_update_callback
        
    @on_field_update.setter
    def on_field_update(self, callback: Optional[Callable[[str, Set[ResonanceField]], None]]) -> None:
        self._field_update_callback = callback
    
    async def start(self, port: int, active_fields: Set[ResonanceField], metadata: Dict[str, Any] = None):
        """Start service discovery and broadcasting."""
        try:
            # Start Ray in auto mode
            await ray_manager.start(mode='auto', port=6379)
            
            # Register our mandala service
            self.info = ServiceInfo(
                self.SERVICE_TYPE,
                f"{self.node_id}.{self.SERVICE_TYPE}",
                addresses=[socket.inet_aton(socket.gethostbyname(socket.gethostname()))],
                port=port,
                properties={
                    'node_id': self.node_id,
                    'active_fields': json.dumps([f.value for f in active_fields]),
                    'field_strength': str(metadata.get('field_strength', 0.0)),
                    'last_evolution': datetime.now().isoformat(),
                    'metadata': json.dumps(metadata or {})
                }
            )
            self.zeroconf.register_service(self.info)
            
            # Start browsing for other nodes
            self.browser = ServiceBrowser(
                self.zeroconf,
                self.SERVICE_TYPE,
                handlers=[self._handle_service_state_change]
            )
            
            logger.info(f"Started resonance discovery on port {port}")
            
        except Exception as e:
            logger.error(f"Failed to start resonance discovery: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop service discovery."""
        if self.info:
            self.zeroconf.unregister_service(self.info)
        if self.browser:
            self.browser.cancel()
        self.zeroconf.close()
        
        # Stop Ray service discovery
        await ray_manager.stop()
        
        logger.info("Stopped resonance discovery")
    
    def _handle_service_state_change(
        self,
        zeroconf: Zeroconf,
        service_type: str,
        name: str,
        state_change: ServiceStateChange
    ):
        """Handle service discovery events."""
        if state_change is ServiceStateChange.Added:
            info = zeroconf.get_service_info(service_type, name)
            if info:
                try:
                    properties = {
                        k.decode(): v.decode() if isinstance(v, bytes) else v
                        for k, v in info.properties.items()
                    }
                    
                    node = ResonanceNode(
                        node_id=properties['node_id'],
                        host=socket.inet_ntoa(info.addresses[0]),
                        port=info.port,
                        active_fields={
                            ResonanceField(f) for f in json.loads(properties['active_fields'])
                        },
                        field_strength=float(properties['field_strength']),
                        last_evolution=datetime.fromisoformat(properties['last_evolution']),
                        metadata=json.loads(properties['metadata'])
                    )
                    
                    with self._lock:
                        self.nodes[node.node_id] = node
                    
                    if callable(self._node_join_callback):
                        self._node_join_callback(node)
                        
                    logger.info(f"Discovered resonance node: {node.node_id}")
                    
                except Exception as e:
                    logger.error(f"Error processing discovered node: {e}")
                    
        elif state_change is ServiceStateChange.Removed:
            node_id = name.replace(f".{service_type}", "")
            with self._lock:
                if node_id in self.nodes:
                    del self.nodes[node_id]
                    if callable(self._node_leave_callback):
                        self._node_leave_callback(node_id)
                    logger.info(f"Resonance node left: {node_id}")

@ray.remote
class ResonanceNetworkManager:
    """Manages resonance network across Ray cluster."""
    
    def __init__(self, discovery: ResonanceDiscovery):
        self.discovery = discovery
        self.field_registry: Dict[ResonanceField, Set[str]] = {
            field: set() for field in ResonanceField
        }
        
        # Set service info for monitoring
        self.service_info = {
            'name': f'resonance_manager_{discovery.node_id}',
            'host': socket.gethostname(),
            'port': discovery.info.port if discovery.info else 0,
            'metadata': {
                'node_id': discovery.node_id,
                'active_fields': len(discovery.nodes),
                'version': '1.0'
            }
        }
    
    def register_field(self, node_id: str, field: ResonanceField):
        """Register a node's interest in a resonance field."""
        self.field_registry[field].add(node_id)
        # Update monitoring metadata
        self.service_info['metadata']['active_fields'] = sum(len(nodes) for nodes in self.field_registry.values())
    
    def unregister_field(self, node_id: str, field: ResonanceField):
        """Unregister a node's interest in a field."""
        self.field_registry[field].discard(node_id)
        # Update monitoring metadata
        self.service_info['metadata']['active_fields'] = sum(len(nodes) for nodes in self.field_registry.values())
    
    def get_field_participants(self, field: ResonanceField) -> Set[str]:
        """Get all nodes participating in a field."""
        return self.field_registry[field].copy()
    
    async def broadcast_evolution(
        self,
        source_node: str,
        field: ResonanceField,
        pattern: np.ndarray,
        metrics: Dict[str, Any]
    ) -> List[Any]:
        """Broadcast an evolution event to all nodes in a field."""
        participants = self.field_registry[field]
        futures = []
        
        for node_id in participants:
            if node_id != source_node:
                node = self.discovery.nodes.get(node_id)
                if node:
                    # Could implement pattern sharing here
                    pass
        
        return await asyncio.gather(*futures)

async def create_resonance_network(
    port: int,
    active_fields: Set[ResonanceField],
    metadata: Dict[str, Any] = None
) -> Tuple[ResonanceDiscovery, ray.ObjectRef]:
    """Create a resonance network node."""
    
    # Initialize discovery
    discovery = ResonanceDiscovery()
    await discovery.start(port, active_fields, metadata)
    
    # Create network manager
    manager = ResonanceNetworkManager.remote(discovery)
    
    # Register our fields
    for field in active_fields:
        await manager.register_field.remote(discovery.node_id, field)
    
    return discovery, manager 