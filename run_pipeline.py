"""
run_pipeline.py

Main orchestration script for the ML Model Monitoring System.
Runs the complete pipeline from training to alerting.

Usage:
    python run_pipeline.py [--skip-training] [--verbose]
"""

import argparse
import subprocess
import sys
from pathlib import Path

# =========================
# CONFIGURATION
# =========================

BASE_DIR = Path(__file__).parent
SRC_DIR = BASE_DIR / "src"

# Pipeline steps in execution order
PIPELINE_STEPS = [
    {
        "name": "Model Training",
        "script": SRC_DIR / "training" / "train_model.py",
        "description": "Train baseline model and save metrics",
        "skip_flag": "skip_training"
    },
    {
        "name": "Model Evaluation",
        "script": SRC_DIR / "training" / "evaluate_model.py",
        "description": "Evaluate model on simulated production data"
    },
    {
        "name": "Data Drift Detection",
        "script": SRC_DIR / "monitoring" / "data_drift.py",
        "description": "Detect feature distribution drift"
    },
    {
        "name": "Concept Drift Detection",
        "script": SRC_DIR / "monitoring" / "concept_drift.py",
        "description": "Detect prediction distribution drift"
    },
    {
        "name": "Performance Monitoring",
        "script": SRC_DIR / "monitoring" / "performance_monitor.py",
        "description": "Monitor model performance over batches"
    },
    {
        "name": "Alert Generation",
        "script": SRC_DIR / "monitoring" / "alert_engine.py",
        "description": "Generate alerts based on monitoring results"
    },
    {
        "name": "Retraining Decision",
        "script": SRC_DIR / "monitoring" / "retrain_controller.py",
        "description": "Evaluate whether retraining is needed"
    }
]


# =========================
# HELPER FUNCTIONS
# =========================

def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_step(step_num: int, total: int, name: str, description: str):
    """Print step information."""
    print(f"\n[{step_num}/{total}] {name}")
    print(f"    → {description}")


def run_script(script_path: Path, verbose: bool = False) -> bool:
    """
    Run a Python script and return success status.
    
    Args:
        script_path: Path to the Python script
        verbose: Whether to show script output
        
    Returns:
        True if script executed successfully, False otherwise
    """
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=not verbose,
            text=True,
            cwd=str(BASE_DIR)  # Run from project root
        )
        
        if result.returncode != 0:
            print(f"    ❌ Error running {script_path.name}")
            if not verbose and result.stderr:
                print(f"    Error: {result.stderr[:500]}")
            return False
            
        print(f"    ✅ Completed successfully")
        
        if verbose and result.stdout:
            for line in result.stdout.strip().split("\n"):
                print(f"       {line}")
                
        return True
        
    except Exception as e:
        print(f"    ❌ Exception: {str(e)}")
        return False


# =========================
# MAIN PIPELINE
# =========================

def run_pipeline(skip_training: bool = False, verbose: bool = False) -> bool:
    """
    Execute the complete monitoring pipeline.
    
    Args:
        skip_training: Skip the training step (use existing model)
        verbose: Show detailed output from each step
        
    Returns:
        True if all steps completed successfully
    """
    print_header("ML Model Monitoring Pipeline")
    print(f"\n🚀 Starting pipeline execution...")
    print(f"   Skip training: {skip_training}")
    print(f"   Verbose mode: {verbose}")
    
    total_steps = len(PIPELINE_STEPS)
    successful_steps = 0
    failed_steps = []
    
    for i, step in enumerate(PIPELINE_STEPS, 1):
        # Check if step should be skipped
        skip_flag = step.get("skip_flag")
        if skip_flag and skip_training:
            print(f"\n[{i}/{total_steps}] {step['name']} - SKIPPED")
            successful_steps += 1
            continue
            
        print_step(i, total_steps, step["name"], step["description"])
        
        if run_script(step["script"], verbose):
            successful_steps += 1
        else:
            failed_steps.append(step["name"])
            # Continue with remaining steps even if one fails
    
    # Print summary
    print_header("Pipeline Summary")
    print(f"\n📊 Results:")
    print(f"   ✅ Successful: {successful_steps}/{total_steps}")
    print(f"   ❌ Failed: {len(failed_steps)}/{total_steps}")
    
    if failed_steps:
        print(f"\n⚠️  Failed steps:")
        for step_name in failed_steps:
            print(f"   - {step_name}")
        return False
    else:
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"\n📍 Next steps:")
        print(f"   1. Run 'streamlit run streamlit_app.py' to view dashboard")
        print(f"   2. Run 'uvicorn src.api.main:app --reload' to start API")
        return True


# =========================
# CLI ENTRY POINT
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Run the ML Model Monitoring Pipeline"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training step (use existing model)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output from each step"
    )
    
    args = parser.parse_args()
    
    success = run_pipeline(
        skip_training=args.skip_training,
        verbose=args.verbose
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
