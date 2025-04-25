"""
CLI tool for managing Ray clusters.
"""

import click
import asyncio
import logging
from pathlib import Path
from typing import Optional
import json
from rich.console import Console
from rich.table import Table
from rich.live import Live
import time
import functools

from ...utils.ray_manager import get_ray_manager

console = Console()
logger = logging.getLogger(__name__)

def coro(f):
    """Decorator to run async functions in click commands."""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper

@click.group()
def cli():
    """Ray cluster management commands."""
    pass

@cli.command()
@click.option('--mode', type=click.Choice(['auto', 'head', 'worker']), default='auto',
              help='Node mode (auto detects if not specified)')
@click.option('--port', default=6379, help='Port for Ray head node')
@click.option('--dashboard-port', default=8265, help='Port for Ray dashboard')
@click.option('--log-dir', type=click.Path(), help='Directory for logs')
@click.option('--resources', type=str, help='JSON string of resource specifications')
@coro
async def start(mode: str, port: int, dashboard_port: int, log_dir: Optional[str], resources: Optional[str]):
    """Start a Ray node."""
    try:
        # Parse resources if provided
        resource_dict = json.loads(resources) if resources else None
        
        # Initialize Ray manager
        manager = get_ray_manager(
            log_dir=Path(log_dir) if log_dir else None,
            dashboard_port=dashboard_port
        )
        
        # Start node
        await manager.start(
            mode=mode,
            port=port,
            resources=resource_dict
        )
        
        # Display status
        console.print(f"\n[green]Started Ray node in {mode} mode[/green]")
        console.print("\nNode Information:")
        node_info = manager.get_node_info()
        for key, value in node_info.items():
            console.print(f"  {key}: {value}")
            
        # Keep running and display cluster info
        with Live(refresh_per_second=1) as live:
            try:
                while True:
                    table = Table(title="Cluster Status")
                    table.add_column("Node ID")
                    table.add_column("Role")
                    table.add_column("Resources")
                    table.add_column("Last Heartbeat")
                    
                    cluster_info = manager.get_cluster_info()
                    for node in cluster_info['nodes']:
                        table.add_row(
                            node['node_id'],
                            "Head" if node['is_head'] else "Worker",
                            str(node['resources']),
                            node['last_heartbeat']
                        )
                    
                    live.update(table)
                    await asyncio.sleep(1)
                    
            except KeyboardInterrupt:
                await manager.stop()
                console.print("\n[yellow]Stopped Ray node[/yellow]")
                
    except Exception as e:
        logger.error(f"Error starting Ray node: {e}")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--wait/--no-wait', default=True, help='Wait for nodes to stop')
@coro
async def stop(wait: bool):
    """Stop Ray nodes."""
    try:
        manager = get_ray_manager()
        await manager.stop()
        
        if wait:
            with console.status("Waiting for nodes to stop..."):
                await asyncio.sleep(2)  # Give nodes time to clean up
                
        console.print("[green]Successfully stopped Ray nodes[/green]")
        
    except Exception as e:
        logger.error(f"Error stopping Ray nodes: {e}")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--watch/--no-watch', default=False, help='Watch cluster status')
@coro
async def status(watch: bool):
    """Show Ray cluster status."""
    try:
        manager = get_ray_manager()
        
        if watch:
            with Live(refresh_per_second=1) as live:
                while True:
                    tables = []
                    
                    # Node status table
                    node_table = Table(title="Node Status")
                    node_table.add_column("Node ID")
                    node_table.add_column("Role")
                    node_table.add_column("Resources")
                    node_table.add_column("Status")
                    
                    cluster_info = manager.get_cluster_info()
                    for node in cluster_info['nodes']:
                        node_table.add_row(
                            node['node_id'],
                            "Head" if node['is_head'] else "Worker",
                            str(node['resources']),
                            "Active"
                        )
                    
                    tables.append(node_table)
                    
                    # Resource summary table
                    resource_table = Table(title="Resource Summary")
                    resource_table.add_column("Resource")
                    resource_table.add_column("Total")
                    resource_table.add_column("Available")
                    
                    for resource, amount in cluster_info['total_resources'].items():
                        resource_table.add_row(
                            resource,
                            str(amount),
                            str(amount)  # TODO: Add actual available resources
                        )
                    
                    tables.append(resource_table)
                    
                    # Update display
                    live.update(
                        "\n".join(str(table) for table in tables)
                    )
                    
                    await asyncio.sleep(1)
        else:
            # Show one-time status
            cluster_info = manager.get_cluster_info()
            console.print("\n[bold]Cluster Status[/bold]")
            
            console.print("\nNodes:")
            for node in cluster_info['nodes']:
                console.print(f"  • {node['node_id']} ({'Head' if node['is_head'] else 'Worker'})")
                console.print(f"    Resources: {node['resources']}")
                console.print(f"    Last Heartbeat: {node['last_heartbeat']}")
            
            console.print("\nTotal Resources:")
            for resource, amount in cluster_info['total_resources'].items():
                console.print(f"  • {resource}: {amount}")
                
    except Exception as e:
        logger.error(f"Error showing status: {e}")
        raise click.ClickException(str(e))

def run():
    """Entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise click.ClickException(str(e))

if __name__ == '__main__':
    run() 