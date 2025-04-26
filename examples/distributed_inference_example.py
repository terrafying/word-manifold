"""Example of using distributed inference with human interaction."""

import asyncio
import uuid
import z3
from typing import Dict, Any
from word_manifold.manifold.distributed_inference import (
    DistributedInferenceManager,
    InferenceTask,
    InferenceMode
)
from word_manifold.manifold.pattern_selection import PatternSource
from word_manifold.manifold.z3_integration import Z3Constraint

async def main():
    """Run the distributed inference example."""
    # Create inference manager
    manager = DistributedInferenceManager(
        max_workers=4,
        human_interaction_threshold=0.8,
        pattern_source=PatternSource.HERMETIC,
        z3_timeout=30000,
        z3_max_memory=1024 * 1024 * 1024
    )
    
    # Start the manager
    await manager.start()
    
    try:
        # Create Z3 variables
        x = z3.Real('x')
        y = z3.Real('y')
        
        # Create example tasks
        tasks = [
            # Automated task with Z3 constraints
            InferenceTask(
                task_id=str(uuid.uuid4()),
                input_data={"text": "Example text 1"},
                mode=InferenceMode.AUTOMATED,
                priority=1.0,
                z3_constraints=[
                    Z3Constraint(
                        constraint=x + y == 10,
                        description="Sum constraint",
                        source="example"
                    ),
                    Z3Constraint(
                        constraint=x > y,
                        description="Ordering constraint",
                        source="example"
                    )
                ],
                optimization_objective=x * y  # Maximize product
            ),
            
            # Human-guided task
            InferenceTask(
                task_id=str(uuid.uuid4()),
                input_data={"text": "Example text 2"},
                mode=InferenceMode.HUMAN_GUIDED,
                priority=0.8,
                human_guidance={
                    "x_preference": 7.0,  # Human prefers x to be around 7
                    "y_preference": 3.0   # Human prefers y to be around 3
                },
                z3_constraints=[
                    Z3Constraint(
                        constraint=x + y == 10,
                        description="Sum constraint",
                        source="example"
                    )
                ]
            ),
            
            # Human-verified task
            InferenceTask(
                task_id=str(uuid.uuid4()),
                input_data={"text": "Example text 3"},
                mode=InferenceMode.HUMAN_VERIFIED,
                priority=0.9,
                z3_constraints=[
                    Z3Constraint(
                        constraint=x + y == 10,
                        description="Sum constraint",
                        source="example"
                    ),
                    Z3Constraint(
                        constraint=x > 0,
                        description="Positive x",
                        source="example"
                    ),
                    Z3Constraint(
                        constraint=y > 0,
                        description="Positive y",
                        source="example"
                    )
                ]
            ),
            
            # Hybrid task
            InferenceTask(
                task_id=str(uuid.uuid4()),
                input_data={"text": "Example text 4"},
                mode=InferenceMode.HYBRID,
                priority=0.7,
                z3_constraints=[
                    Z3Constraint(
                        constraint=x + y == 10,
                        description="Sum constraint",
                        source="example"
                    )
                ],
                optimization_objective=x * y
            )
        ]
        
        # Submit tasks
        print("Submitting tasks...")
        for task in tasks:
            await manager.submit_task(task)
            
        # Monitor task progress
        print("\nMonitoring task progress...")
        for task in tasks:
            while True:
                status = await manager.get_task_status(task.task_id)
                if status["status"] in ["completed", "failed"]:
                    break
                await asyncio.sleep(0.1)
                
        # Get results
        print("\nRetrieving results...")
        for task in tasks:
            result = await manager.get_result(task.task_id)
            status = await manager.get_task_status(task.task_id)
            
            print(f"\nTask {task.task_id}:")
            print(f"Mode: {task.mode.value}")
            print(f"Status: {status['status']}")
            if status["status"] == "completed":
                print(f"Result: {result}")
                if "statistics" in result:
                    print(f"Statistics: {result['statistics']}")
            elif status["status"] == "failed":
                print(f"Error: {status.get('error', 'Unknown error')}")
                
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        # Stop the manager
        await manager.stop()

if __name__ == "__main__":
    asyncio.run(main()) 