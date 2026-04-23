"""
train_all.py
-----------
Runs all 4 simple_spread implementations sequentially.
Usage: python train_all.py
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Configuration
IMPLEMENTATIONS = [
    {
        "name": "Parameter Sharing",
        "script": "train_simple_spread.py",
        "description": "Single shared policy for all 3 agents"
    },
    {
        "name": "Round-Robin Multi-Agent",
        "script": "train_simple_spread_multiagent.py",
        "description": "3 separate policies, trained round-robin"
    },
    {
        "name": "Joint Observation",
        "script": "train_joint_observation.py",
        "description": "Coordinated multi-agent with joint observations"
    },
    {
        "name": "MAPPO",
        "script": "train_mappo.py",
        "description": "Multi-Agent PPO with centralized critic"
    }
]

TOTAL_TIMESTEPS = 1_000_000  # Adjust as needed


def run_training(script_name, impl_name):
    """Run a single training script."""
    print(f"\n{'='*70}")
    print(f"Starting: {impl_name}")
    print(f"Script: {script_name}")
    print(f"{'='*70}\n")
    
    start_time = datetime.now()
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n{'='*70}")
        print(f"✓ Completed: {impl_name}")
        print(f"Duration: {duration}")
        print(f"{'='*70}\n")
        
        return True, duration
        
    except subprocess.CalledProcessError as e:
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n{'='*70}")
        print(f"✗ Failed: {impl_name}")
        print(f"Duration before failure: {duration}")
        print(f"Error: {e}")
        print(f"{'='*70}\n")
        
        return False, duration
    except KeyboardInterrupt:
        print(f"\n\n{'='*70}")
        print(f"Training interrupted by user during: {impl_name}")
        print(f"{'='*70}\n")
        raise


def main():
    """Run all implementations sequentially."""
    print("\n" + "="*70)
    print("TRAINING ALL SIMPLE_SPREAD IMPLEMENTATIONS")
    print("="*70)
    print(f"Total implementations: {len(IMPLEMENTATIONS)}")
    print(f"Timesteps per implementation: {TOTAL_TIMESTEPS:,}")
    print("="*70 + "\n")
    
    # Print summary
    for i, impl in enumerate(IMPLEMENTATIONS, 1):
        print(f"{i}. {impl['name']}")
        print(f"   Script: {impl['script']}")
        print(f"   Description: {impl['description']}\n")
    
    input("Press Enter to start training...")
    
    overall_start = datetime.now()
    results = []
    
    for i, impl in enumerate(IMPLEMENTATIONS, 1):
        print(f"\n\n{'#'*70}")
        print(f"IMPLEMENTATION {i}/{len(IMPLEMENTATIONS)}")
        print(f"{'#'*70}")
        
        success, duration = run_training(impl['script'], impl['name'])
        results.append({
            'name': impl['name'],
            'success': success,
            'duration': duration
        })
        
        # If a training fails, ask whether to continue
        if not success:
            response = input("\nTraining failed. Continue with remaining implementations? (y/n): ")
            if response.lower() != 'y':
                print("Stopping training sequence.")
                break
    
    overall_end = datetime.now()
    overall_duration = overall_end - overall_start
    
    # Print summary
    print("\n\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Total time: {overall_duration}\n")
    
    for i, result in enumerate(results, 1):
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        print(f"{i}. {result['name']}")
        print(f"   Status: {status}")
        print(f"   Duration: {result['duration']}\n")
    
    successful = sum(1 for r in results if r['success'])
    print(f"Completed: {successful}/{len(results)} implementations")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining sequence interrupted by user.")
        sys.exit(1)