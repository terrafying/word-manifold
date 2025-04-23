#!/usr/bin/env python3
"""
Development watch script for word-manifold.

This script monitors the source and test files for changes,
and automatically runs tests and regenerates visualizations.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class CodeChangeHandler(FileSystemEventHandler):
    """Handle file system change events."""
    
    def __init__(self):
        self.last_run = 0
        self.debounce_seconds = 2.0  # Minimum seconds between runs
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        # Only process Python files
        if not event.src_path.endswith('.py'):
            return
            
        # Debounce to avoid multiple runs
        current_time = time.time()
        if current_time - self.last_run < self.debounce_seconds:
            return
            
        self.last_run = current_time
        self.run_tests_and_viz()
        
    def run_tests_and_viz(self):
        """Run tests and regenerate visualizations."""
        logging.info("Changes detected - running tests and visualizations...")
        
        # Set PYTHONPATH to include src directory
        env = os.environ.copy()
        env["PYTHONPATH"] = "src:" + env.get("PYTHONPATH", "")
        
        try:
            # Run tests first
            logging.info("Running tests...")
            result = subprocess.run(
                ["pytest", "-v", "tests/"],
                env=env,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logging.error("Tests failed:")
                logging.error(result.stdout)
                logging.error(result.stderr)
                return
                
            logging.info("Tests passed successfully")
            
            # Generate visualizations
            logging.info("Generating visualizations...")
            result = subprocess.run(
                ["python", "-m", "word_manifold"],
                env=env,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logging.error("Visualization generation failed:")
                logging.error(result.stdout)
                logging.error(result.stderr)
                return
                
            logging.info("Visualizations generated successfully")
            
        except Exception as e:
            logging.error(f"Error running tests/visualizations: {e}")

def main():
    """Run the development watch script."""
    # Create an observer and event handler
    observer = Observer()
    handler = CodeChangeHandler()
    
    # Watch both src and tests directories
    paths_to_watch = ["src", "tests"]
    for path in paths_to_watch:
        observer.schedule(handler, path, recursive=True)
    
    # Start the observer
    observer.start()
    logging.info("Watching for changes in src/ and tests/...")
    
    try:
        # Run once at startup
        handler.run_tests_and_viz()
        
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logging.info("Stopping file watch")
        
    observer.join()

if __name__ == "__main__":
    main() 