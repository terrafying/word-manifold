"""
Test script for Ray cluster operations.
Runs a continuous loop of tests to verify cluster functionality.
"""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import ray
from ray import serve
from rich.console import Console
from rich.table import Table

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/cluster_tests.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@serve.deployment(name="echo", num_replicas=2)
class EchoModel:
    def __init__(self):
        self._counter = 0

    async def __call__(self, request) -> Dict:
        self._counter += 1
        return {
            "echo": request.query_params.get("message", "Hello!"),
            "counter": self._counter
        }

class ClusterTester:
    def __init__(self):
        self.console = Console()
        self.test_results: List[Dict] = []
        self.current_iteration = 0
        
    async def run_tests(self):
        """Run all cluster tests in a continuous loop."""
        while True:
            self.current_iteration += 1
            iteration_start = datetime.now()
            
            self.console.print(f"\n[bold blue]Starting test iteration {self.current_iteration}[/bold blue]")
            
            try:
                # Initialize Ray if not already running
                if not ray.is_initialized():
                    # Start Ray with specific resources to ensure worker creation
                    ray.init(
                        num_cpus=4,  # Request more CPUs to ensure worker creation
                        resources={"worker": 2},  # Add custom resource to force worker allocation
                        _system_config={}
                        
                    )
                
                # Run individual tests
                await self.test_head_node_startup()
                await self.test_worker_node_startup()
                await self.test_task_execution()
                await self.test_resource_management()
                await self.test_serve_deployment()
                
                # Display results
                self.display_results()
                
                # Clear results for next iteration
                self.test_results.clear()
                
            except Exception as e:
                logger.error(f"Error in test iteration {self.current_iteration}: {str(e)}")
                
            finally:
                iteration_time = (datetime.now() - iteration_start).total_seconds()
                self.console.print(f"[bold green]Test iteration completed in {iteration_time:.2f} seconds[/bold green]")
                
                # Wait before next iteration
                await asyncio.sleep(60)  # 1 minute between iterations
    
    async def test_head_node_startup(self):
        """Test Ray head node startup."""
        start_time = time.time()
        success = False
        error_msg = None
        
        try:
            # Verify head node is running
            nodes = ray.nodes()
            if len(nodes) > 0 and any(node.get("alive") for node in nodes):
                success = True
            else:
                error_msg = "Head node not found or not alive"
                
        except Exception as e:
            error_msg = str(e)
            
        finally:
            duration = time.time() - start_time
            self.test_results.append({
                "test_name": "Head Node Startup",
                "success": success,
                "duration": duration,
                "error": error_msg
            })
    
    async def test_worker_node_startup(self):
        """Test worker node startup."""
        start_time = time.time()
        success = False
        error_msg = None
        
        try:
            # Wait briefly for workers to start
            await asyncio.sleep(1)
            
            # Check for worker nodes using multiple methods
            nodes = ray.nodes()
            worker_count = 0
            
            for node in nodes:
                # A node is a worker if:
                # 1. It's alive
                # 2. Has worker resources
                # 3. Not marked as head
                if (node.get("alive") and 
                    node.get("Resources", {}).get("worker", 0) > 0 and
                    not node.get("RayletSocketName", "").endswith("raylet")):
                    worker_count += 1
            
            if worker_count > 0:
                success = True
                logger.info(f"Found {worker_count} worker nodes")
            else:
                error_msg = "No worker nodes found"
                logger.debug(f"Node details: {nodes}")
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error checking worker nodes: {e}")
            
        finally:
            duration = time.time() - start_time
            self.test_results.append({
                "test_name": "Worker Node Startup",
                "success": success,
                "duration": duration,
                "error": error_msg
            })
    
    async def test_task_execution(self):
        """Test basic task execution on the cluster."""
        start_time = time.time()
        success = False
        error_msg = None
        
        try:
            @ray.remote
            def dummy_task(x: int) -> int:
                return x * 2
            
            # Submit multiple tasks
            futures = [dummy_task.remote(i) for i in range(5)]
            results = ray.get(futures)
            
            if all(result == i * 2 for i, result in enumerate(results)):
                success = True
            else:
                error_msg = "Task results did not match expected values"
                
        except Exception as e:
            error_msg = str(e)
            
        finally:
            duration = time.time() - start_time
            self.test_results.append({
                "test_name": "Task Execution",
                "success": success,
                "duration": duration,
                "error": error_msg
            })
    
    async def test_resource_management(self):
        """Test cluster resource management."""
        start_time = time.time()
        success = False
        error_msg = None
        
        try:
            resources = ray.available_resources()
            if resources and resources.get("CPU", 0) > 0:
                success = True
            else:
                error_msg = "No CPU resources available"
                
        except Exception as e:
            error_msg = str(e)
            
        finally:
            duration = time.time() - start_time
            self.test_results.append({
                "test_name": "Resource Management",
                "success": success,
                "duration": duration,
                "error": error_msg
            })

    async def test_serve_deployment(self):
        """Test Ray Serve deployment and endpoint functionality."""
        start_time = time.time()
        success = False
        error_msg = None
        
        try:
            # Start Ray Serve if not already running
            try:
                serve.start(detached=True, http_options={"host": "127.0.0.1", "port": 8000})
                logger.info("Ray Serve started successfully")
            except Exception as e:
                error_msg = f"Failed to start Ray Serve: {str(e)}"
                raise
            
            # Deploy the model
            try:
                # Create deployment using bind()
                deployment = EchoModel.bind()
                logger.info("Echo model bound successfully")
            except Exception as e:
                error_msg = f"Failed to bind model: {str(e)}"
                raise
            
            # Wait for deployment to be ready
            try:
                await asyncio.sleep(2)  # Give deployment time to start
                logger.info("Deployment should be ready")
            except Exception as e:
                error_msg = f"Failed during deployment wait: {str(e)}"
                raise
            
            # Test the deployment
            try:
                result = await deployment.remote(message="test")
                logger.info(f"Got result from deployment: {result}")
                
                # Verify the result
                if result["echo"] == "test" and isinstance(result["counter"], int):
                    success = True
                else:
                    error_msg = f"Unexpected response format: {result}"
            except Exception as e:
                error_msg = f"Failed to test deployment: {str(e)}"
                raise
                
        except Exception as e:
            if not error_msg:
                error_msg = str(e)
            logger.error(f"Serve deployment test failed: {error_msg}")
            
        finally:
            duration = time.time() - start_time
            self.test_results.append({
                "test_name": "Serve Deployment",
                "success": success,
                "duration": duration,
                "error": error_msg
            })
            
            # Cleanup
            try:
                serve.shutdown()
                logger.info("Ray Serve shutdown complete")
            except Exception as e:
                logger.error(f"Failed to shutdown Ray Serve: {e}")
    
    def display_results(self):
        """Display test results in a formatted table."""
        table = Table(title=f"Test Results - Iteration {self.current_iteration}")
        
        table.add_column("Test Name", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Duration (s)", justify="right")
        table.add_column("Error", style="red")
        
        for result in self.test_results:
            status = "[green]✓[/green]" if result["success"] else "[red]✗[/red]"
            table.add_row(
                result["test_name"],
                status,
                f"{result['duration']:.3f}",
                result.get("error", "")
            )
        
        self.console.print(table)

async def main():
    """Main entry point for the test script."""
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    logger.info("Starting cluster test loop")
    tester = ClusterTester()
    
    try:
        await tester.run_tests()
    except KeyboardInterrupt:
        logger.info("Test loop interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error in test loop: {str(e)}")
    finally:
        # Ensure serve is shutdown
        try:
            serve.shutdown()
            logger.info("Ray Serve shutdown complete")
        except Exception as e:
            logger.error(f"Failed to shutdown Ray Serve: {e}")
            
        # Shutdown ray
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray shutdown complete")
        
        logger.info("Test loop terminated")

if __name__ == "__main__":
    asyncio.run(main()) 