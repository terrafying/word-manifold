#!/usr/bin/env python3
"""
Ray Cluster Management Script

Starts and manages a Ray cluster for distributed model hosting.
"""

import ray
import yaml
import click
import logging
from pathlib import Path
import time
import socket
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_local_ip() -> str:
    """Get local IP address."""
    return "0.0.0.0"

def load_config(config_path: Path) -> dict:
    """Load Ray configuration."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Update head node IP if set to localhost
    if config["head_node"]["node_ip"] == "localhost":
        config["head_node"]["node_ip"] = get_local_ip()
    
    return config

def wait_for_nodes(expected_nodes: int, timeout: int = 60) -> bool:
    """Wait for worker nodes to connect."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        nodes = ray.nodes()
        alive_nodes = len([n for n in nodes if n["alive"]])
        if alive_nodes >= expected_nodes:
            return True
        time.sleep(1)
        logger.info(f"Waiting for nodes... ({alive_nodes}/{expected_nodes})")
    return False

@click.group()
def cli():
    """Ray cluster management CLI."""
    pass

@cli.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), default="ray_config.yaml", help="Path to Ray configuration file")
@click.option("--head/--worker", default=True, help="Start as head node or worker node")
@click.option("--block/--no-block", default=True, help="Block the script from exiting")
def start(config: Path, head: bool, block: bool):
    """Start Ray node (head or worker)."""
    try:
        ray_config = load_config(config)
        
        if head:
            logger.info("Starting Ray head node...")
            ray.init(
                _system_config=ray_config["system_config"],
                runtime_env=ray_config["runtime_env"]
            )
            
            # Wait for worker nodes
            num_workers = len(ray_config["worker_nodes"])
            if num_workers > 0:
                if wait_for_nodes(num_workers + 1):  # +1 for head node
                    logger.info("All worker nodes connected")
                else:
                    logger.warning("Timeout waiting for worker nodes")
            
            if block:
                logger.info("Head node running. Press Ctrl+C to exit.")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
                    
        else:
            # Connect to existing cluster
            logger.info("Connecting to Ray cluster as worker...")
            head_ip = ray_config["head_node"]["node_ip"]
            head_port = ray_config["head_node"]["redis_port"]
            password = ray_config["head_node"]["redis_password"]
            
            ray.init(
                address=f"ray://{head_ip}:{head_port}",
                _redis_password=password,
                runtime_env=ray_config["runtime_env"]
            )
            
            if block:
                logger.info("Worker node running. Press Ctrl+C to exit.")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
    
    except Exception as e:
        logger.error(f"Error starting Ray node: {e}")
        sys.exit(1)
    finally:
        if ray.is_initialized():
            ray.shutdown()

@cli.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), default="ray_config.yaml", help="Path to Ray configuration file")
def status(config: Path):
    """Check Ray cluster status."""
    try:
        ray_config = load_config(config)
        
        # Try to connect to cluster
        try:
            head_ip = ray_config["head_node"]["node_ip"]
            head_port = ray_config["head_node"]["redis_port"]
            password = ray_config["head_node"]["redis_password"]
            
            ray.init(
                address=f"ray://{head_ip}:{head_port}",
                _redis_password=password,
                runtime_env=ray_config["runtime_env"]
            )
            
            # Get cluster info
            nodes = ray.nodes()
            alive_nodes = [n for n in nodes if n["alive"]]
            dead_nodes = [n for n in nodes if not n["alive"]]
            
            click.echo("\nCluster Status:")
            click.echo(f"Total Nodes: {len(nodes)}")
            click.echo(f"Alive Nodes: {len(alive_nodes)}")
            click.echo(f"Dead Nodes: {len(dead_nodes)}")
            
            # Show node details
            click.echo("\nNode Details:")
            for node in nodes:
                status = "ALIVE" if node["alive"] else "DEAD"
                click.echo(f"\nNode {node['NodeID']}:")
                click.echo(f"  Status: {status}")
                click.echo(f"  IP: {node['NodeManagerAddress']}")
                click.echo(f"  Resources: {node['Resources']}")
                
        except Exception as e:
            click.echo("Cluster is not running")
            
    except Exception as e:
        logger.error(f"Error checking cluster status: {e}")
        sys.exit(1)
    finally:
        if ray.is_initialized():
            ray.shutdown()

if __name__ == "__main__":
    cli() 