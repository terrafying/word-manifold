"""
Visualization Server.

This module provides a Flask server for vector operations, rendering engine functionality,
and data processing for visualizations. It maintains a vector database for efficient
term operations and provides endpoints for visualization generation.
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import logging
import numpy as np
from typing import Dict, Any, List, Optional
import json
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import ray
from functools import wraps
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

from word_manifold.embeddings.word_embeddings import WordEmbeddings
from word_manifold.visualization.engines.timeseries import TimeSeriesEngine, PatternType
from word_manifold.visualization.renderers.timeseries import TimeSeriesRenderer
from word_manifold.manifold.vector_manifold import VectorManifold
from word_manifold.discovery.service_registry import ServiceRegistry
from word_manifold.core.worker_pool import WorkerPool
from word_manifold.monitoring.metrics import MetricsCollector, MetricsExporter
from word_manifold.cloud.ray_cloud import RayCloudManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisualizationServer:
    """Manages the visualization server with Ray integration."""
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 5000,
        num_workers: int = 4,
        ray_address: Optional[str] = None,
        use_cloud: bool = False,
        config_path: Optional[str] = None
    ):
        self.host = host
        self.port = port
        self.use_cloud = use_cloud
        
        # Initialize Ray and cloud services
        if use_cloud:
            self.cloud_manager = RayCloudManager(config_path=config_path)
            self.cloud_manager.setup_cloud_cluster()
        else:
            # Initialize Ray locally if needed
            if not ray.is_initialized():
                ray.init(address=ray_address)
                logger.info(f"Connected to Ray cluster at {ray_address}" if ray_address else "Initialized Ray locally")
        
        # Initialize metrics collection
        self.metrics_collector = MetricsCollector.remote()
        self.metrics_exporter = MetricsExporter.remote(self.metrics_collector)
        
        # Start metrics export in background
        ray.get(self.metrics_exporter.start.remote())
        
        # Initialize service registry
        self.registry = ServiceRegistry.remote()
        
        # Initialize worker pool
        self.worker_pool = WorkerPool(num_workers)
        
        # Initialize Flask app with SocketIO
        self.app = Flask(__name__)
        CORS(self.app)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Initialize embeddings with default terms
        self.embeddings = WordEmbeddings(load_default_terms=True)
        logger.info(f"Loaded {len(self.embeddings.get_terms())} default terms")
        
        # Initialize manifold
        self.vector_manifold = VectorManifold(self.embeddings)
        
        # Initialize core components
        self.timeseries_engine = TimeSeriesEngine(self.embeddings)
        
        # Vector database cache
        self.vector_cache: Dict[str, np.ndarray] = {}
        self.last_update: Dict[str, datetime] = {}
        self.CACHE_DURATION = timedelta(hours=1)
        
        # Active visualization sessions
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Register routes and WebSocket handlers
        self._register_routes()
        self._register_socket_handlers()
        
        # Configure autoscaling if using cloud
        if use_cloud:
            self.cloud_manager.setup_autoscaling(self.metrics_collector)
    
    def _register_routes(self):
        """Register Flask routes with Ray worker handling and metrics."""
        
        def with_metrics(f):
            """Decorator to record request metrics."""
            @wraps(f)
            async def wrapped(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await f(*args, **kwargs)
                    duration = time.time() - start_time
                    ray.get(self.metrics_collector.record_request.remote(duration))
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    ray.get(self.metrics_collector.record_request.remote(duration, error=True))
                    raise
            return wrapped
        
        def with_worker(f):
            """Decorator to handle requests with Ray workers."""
            @wraps(f)
            @with_metrics
            async def wrapped(*args, **kwargs):
                worker = await self.worker_pool.get_available_worker()
                if worker is None:
                    return jsonify({'error': 'No workers available'}), 503
                return await ray.get(worker.process_visualization.remote(request.json))
            return wrapped
        
        @self.app.route('/health')
        def health_check():
            """Health check endpoint."""
            return jsonify({'status': 'healthy'})
        
        @self.app.route('/metrics')
        async def metrics():
            """Get server metrics."""
            metrics_summary = await ray.get(self.metrics_collector.get_metrics_summary.remote())
            return jsonify(metrics_summary)
        
        # Add visualization endpoints with metrics and worker handling
        @self.app.route('/api/v1/visualize', methods=['POST'])
        @with_worker
        async def visualize():
            """Handle visualization request."""
            return await self._handle_visualization(request.json)
    
    async def _handle_visualization(self, data: Dict[str, Any]):
        """Process visualization request with metrics tracking."""
        try:
            # Update worker metrics
            workers = len(self.worker_pool.workers)
            loads = await ray.get([w.is_busy.remote() for w in self.worker_pool.workers])
            avg_load = sum(loads) / max(1, len(loads)) * 100
            
            ray.get(self.metrics_collector.update_worker_metrics.remote(workers, avg_load))
            ray.get(self.metrics_collector.update_cache_metrics.remote(len(self.vector_cache)))
            
            # Process visualization
            # ... visualization logic here ...
            
            return jsonify({'status': 'success'})
            
        except Exception as e:
            logger.error(f"Error processing visualization: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _register_socket_handlers(self):
        """Register WebSocket event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            session_id = request.sid
            self.active_sessions[session_id] = {
                'start_time': datetime.now(),
                'last_update': datetime.now(),
                'visualizations': {}
            }
            emit('connected', {'session_id': session_id})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            session_id = request.sid
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
        
        @self.socketio.on('visualization_request')
        def handle_visualization_request(data):
            """Handle real-time visualization request."""
            session_id = request.sid
            if session_id not in self.active_sessions:
                emit('error', {'message': 'Invalid session'})
                return
            
            try:
                # Process visualization request
                result = self._process_visualization_request(data)
                
                # Update session
                self.active_sessions[session_id]['last_update'] = datetime.now()
                self.active_sessions[session_id]['visualizations'][data.get('id')] = result
                
                # Emit result
                emit('visualization_update', result)
                
            except Exception as e:
                logger.error(f"Error processing visualization request: {e}")
                emit('error', {'message': str(e)})
        
        @self.socketio.on('visualization_update')
        def handle_visualization_update(data):
            """Handle visualization parameter updates."""
            session_id = request.sid
            if session_id not in self.active_sessions:
                emit('error', {'message': 'Invalid session'})
                return
            
            try:
                # Update visualization parameters
                result = self._update_visualization(data)
                
                # Emit updated visualization
                emit('visualization_update', result)
                
            except Exception as e:
                logger.error(f"Error updating visualization: {e}")
                emit('error', {'message': str(e)})
    
    def _process_visualization_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a visualization request.
        
        Args:
            data: Visualization request data
            
        Returns:
            Processed visualization data
        """
        # Get worker
        worker = self.worker_pool.get_available_worker()
        if worker is None:
            raise RuntimeError("No workers available")
        
        # Process request
        result = ray.get(worker.process_visualization.remote(data))
        
        # Update metrics
        ray.get(self.metrics_collector.record_request.remote(
            time.time() - self.active_sessions[request.sid]['start_time'].timestamp()
        ))
        
        return result
    
    def _update_visualization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update visualization parameters.
        
        Args:
            data: Update parameters
            
        Returns:
            Updated visualization data
        """
        session_id = request.sid
        viz_id = data.get('id')
        
        if viz_id not in self.active_sessions[session_id]['visualizations']:
            raise ValueError(f"Visualization {viz_id} not found")
        
        # Get current visualization
        current = self.active_sessions[session_id]['visualizations'][viz_id]
        
        # Update parameters
        current.update(data.get('parameters', {}))
        
        # Reprocess visualization
        return self._process_visualization_request(current)
    
    def run(self):
        """Run the server."""
        self.socketio.run(self.app, host=self.host, port=self.port)
    
    def cleanup(self):
        """Clean up server resources."""
        # Stop metrics export
        ray.get(self.metrics_exporter.stop.remote())
        
        # Clean up worker pool
        self.worker_pool.cleanup()
        
        # Clean up cloud resources if used
        if self.use_cloud:
            self.cloud_manager.cleanup()
        
        # Clear caches
        self.vector_cache.clear()
        self.active_sessions.clear()

def run_server(
    host: str = 'localhost',
    port: int = 5000,
    debug: bool = False,
    use_cloud: bool = False,
    config_path: Optional[str] = None
):
    """Run the visualization server."""
    server = VisualizationServer(
        host=host,
        port=port,
        use_cloud=use_cloud,
        config_path=config_path
    )
    server.run() 