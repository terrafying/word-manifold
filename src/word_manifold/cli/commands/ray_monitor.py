"""
CLI tool for managing and monitoring Ray services.
"""

import click
import ray
import logging
import json
from pathlib import Path
from typing import Optional
import webbrowser
import time
from ...utils.ray_debug import RayDebugMonitor
from rich.console import Console
from rich.table import Table
from rich.live import Live
from datetime import datetime

console = Console()
logger = logging.getLogger(__name__)

@click.group()
def ray_cli():
    """Ray service management and monitoring commands."""
    pass

@ray_cli.command()
@click.option('--dashboard-port', default=8265, help='Port for Ray dashboard')
@click.option('--log-dir', type=click.Path(), help='Directory for monitoring logs')
@click.option('--open-browser/--no-open-browser', default=True, help='Open dashboard in browser')
def start(dashboard_port: int, log_dir: Optional[str], open_browser: bool):
    """Start Ray monitoring dashboard."""
    try:
        monitor = RayDebugMonitor(
            dashboard_port=dashboard_port,
            log_dir=Path(log_dir) if log_dir else None
        )
        monitor.start()
        
        if open_browser:
            webbrowser.open(f"http://localhost:{dashboard_port}")
            
        # Display live service status
        table = Table(title="Ray Services")
        table.add_column("Service")
        table.add_column("Status")
        table.add_column("CPU %")
        table.add_column("Memory %")
        table.add_column("Last Heartbeat")
        
        with Live(table, refresh_per_second=2) as live:
            try:
                while True:
                    table.rows.clear()
                    
                    for service in monitor.get_all_services():
                        table.add_row(
                            service.name,
                            service.status,
                            f"{service.cpu_percent:.1f}%",
                            f"{service.memory_percent:.1f}%",
                            f"{(datetime.now() - service.last_heartbeat).seconds}s ago"
                        )
                    
                    live.update(table)
                    time.sleep(0.5)
                    
            except KeyboardInterrupt:
                console.print("\nStopping monitor...")
                monitor.stop()
                
    except Exception as e:
        logger.error(f"Error starting Ray monitor: {e}")
        raise click.ClickException(str(e))

@ray_cli.command()
@click.option('--service', help='Service name to show details for')
@click.option('--log-dir', type=click.Path(), help='Directory containing monitoring logs')
def status(service: Optional[str], log_dir: Optional[str]):
    """Show detailed status of Ray services."""
    try:
        monitor = RayDebugMonitor(
            log_dir=Path(log_dir) if log_dir else None
        )
        
        if service:
            # Show detailed info for specific service
            service_info = monitor.get_service_info(service)
            if service_info:
                console.print(f"\n[bold]Service: {service}[/bold]")
                console.print(f"Status: {service_info.status}")
                console.print(f"Host: {service_info.host}")
                console.print(f"Port: {service_info.port}")
                console.print(f"PID: {service_info.pid}")
                console.print(f"CPU Usage: {service_info.cpu_percent:.1f}%")
                console.print(f"Memory Usage: {service_info.memory_percent:.1f}%")
                console.print(f"Last Heartbeat: {service_info.last_heartbeat}")
                console.print("\nMetadata:")
                console.print(json.dumps(service_info.metadata, indent=2))
                
                # Show recent metrics
                if log_dir:
                    log_file = Path(log_dir) / f"{service}_metrics.jsonl"
                    if log_file.exists():
                        console.print("\n[bold]Recent Metrics:[/bold]")
                        with open(log_file) as f:
                            for line in f.readlines()[-10:]:  # Show last 10 entries
                                metrics = json.loads(line)
                                console.print(json.dumps(metrics, indent=2))
            else:
                console.print(f"[red]Service {service} not found[/red]")
        else:
            # Show summary of all services
            table = Table(title="Ray Services Summary")
            table.add_column("Service")
            table.add_column("Status")
            table.add_column("Resources")
            table.add_column("Active Since")
            
            for service in monitor.get_all_services():
                table.add_row(
                    service.name,
                    service.status,
                    f"CPU: {service.cpu_percent:.1f}% | Mem: {service.memory_percent:.1f}%",
                    service.last_heartbeat.strftime("%Y-%m-%d %H:%M:%S")
                )
            
            console.print(table)
            
    except Exception as e:
        logger.error(f"Error showing status: {e}")
        raise click.ClickException(str(e))

@ray_cli.command()
@click.argument('service')
def stop(service: str):
    """Stop a Ray service."""
    try:
        monitor = RayDebugMonitor()
        service_info = monitor.get_service_info(service)
        
        if service_info:
            monitor.unregister_service(service)
            console.print(f"[green]Successfully stopped service {service}[/green]")
        else:
            console.print(f"[red]Service {service} not found[/red]")
            
    except Exception as e:
        logger.error(f"Error stopping service: {e}")
        raise click.ClickException(str(e))

if __name__ == '__main__':
    ray_cli() 