"""
Development watch script for Ray cluster operations.
Auto-reloads when relevant files change.
"""

import sys
import time
import logging
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import psutil
import signal
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ClusterWatcher(FileSystemEventHandler):
    """Watches for changes in cluster-related files and restarts processes."""
    
    def __init__(self):
        self.cluster_process: Optional[subprocess.Popen] = None
        self.last_restart = 0
        self.restart_cooldown = 1.0  # Minimum seconds between restarts
        
    def on_modified(self, event):
        if event.src_path.endswith('.py'):
            rel_path = Path(event.src_path).relative_to(Path.cwd())
            if self._is_cluster_related(rel_path):
                self._restart_cluster()
                
    def _is_cluster_related(self, path: Path) -> bool:
        """Check if the modified file is related to cluster operations."""
        cluster_paths = [
            'src/word_manifold/cli/commands/ray_cluster.py',
            'src/word_manifold/utils/ray_service.py',
            'src/word_manifold/utils/ray_manager.py',
            'src/word_manifold/utils/ray_debug.py'
        ]
        return str(path) in cluster_paths
        
    def _restart_cluster(self):
        """Restart the cluster process with cooldown."""
        now = time.time()
        if now - self.last_restart < self.restart_cooldown:
            return
            
        logger.info("Restarting cluster process...")
        self.last_restart = now
        
        # Stop existing process
        if self.cluster_process:
            try:
                # Try graceful shutdown first
                self.cluster_process.terminate()
                try:
                    self.cluster_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.cluster_process.kill()
                    
                # Clean up any Ray processes
                self._cleanup_ray_processes()
                
            except Exception as e:
                logger.error(f"Error stopping process: {e}")
                
        # Start new process
        try:
            self.cluster_process = subprocess.Popen(
                ['word-manifold-cluster', 'start'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            logger.info("Started new cluster process")
            
        except Exception as e:
            logger.error(f"Error starting process: {e}")
            
    def _cleanup_ray_processes(self):
        """Clean up any remaining Ray processes."""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if 'ray' in proc.name().lower() or \
                       (proc.cmdline() and any('ray' in cmd.lower() for cmd in proc.cmdline())):
                        proc.terminate()
                        try:
                            proc.wait(timeout=3)
                        except psutil.TimeoutExpired:
                            proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.error(f"Error cleaning up Ray processes: {e}")

def main():
    """Main entry point for the watch script."""
    src_path = Path('src/word_manifold')
    if not src_path.exists():
        logger.error(f"Source directory not found: {src_path}")
        sys.exit(1)
        
    # Set up file watcher
    watcher = ClusterWatcher()
    observer = Observer()
    observer.schedule(watcher, str(src_path), recursive=True)
    observer.start()
    
    logger.info("Starting cluster dev watch...")
    watcher._restart_cluster()  # Initial start
    
    try:
        while True:
            time.sleep(1)
            
            # Check if process is still running
            if watcher.cluster_process and watcher.cluster_process.poll() is not None:
                stdout, stderr = watcher.cluster_process.communicate()
                if stdout:
                    print("Process output:", stdout)
                if stderr:
                    print("Process errors:", stderr, file=sys.stderr)
                watcher._restart_cluster()
                
    except KeyboardInterrupt:
        logger.info("Stopping dev watch...")
        observer.stop()
        if watcher.cluster_process:
            watcher.cluster_process.terminate()
            watcher._cleanup_ray_processes()
            
    observer.join()

if __name__ == '__main__':
    main() 