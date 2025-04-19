"""
Tennis Match Prediction - Pipeline Orchestration (v4)

This script orchestrates the entire prediction pipeline:
1. Collect historical matches (last 14 days)
2. Calculate Elo ratings
3. Collect upcoming matches
4. Update completed matches
5. Generate historical features
6. Make predictions
7. Update accuracy for past predictions

This script can be run daily via cron to maintain up-to-date predictions.
"""

import os
import sys
import logging
from pathlib import Path
import subprocess
from datetime import datetime
import time

# Add project root to path to allow imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{project_root}/predictor/v4/output/logs/pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    def __init__(self):
        """Initialize the orchestrator"""
        self.v4_dir = project_root / "predictor/v4"
        self.scripts = {
            "historical": self.v4_dir / "collect_historical_matches.py",
            "elo": self.v4_dir / "calculate_elo.py",
            "collect": self.v4_dir / "collect_future_matches.py",
            "update_completed": self.v4_dir / "update_completed_matches.py",
            "historical_features": self.v4_dir / "generate_historical_features.py",
            "predict": self.v4_dir / "predict_matches.py"
        }
        
        # Ensure all required scripts exist
        for name, path in self.scripts.items():
            if not path.exists():
                raise FileNotFoundError(f"Required script not found: {path}")
    
    def run_script(self, script_name: str) -> bool:
        """
        Run a pipeline script
        
        Args:
            script_name: Name of script to run
            
        Returns:
            bool: True if successful, False otherwise
        """
        script_path = self.scripts[script_name]
        logger.info(f"Running {script_name} script: {script_path}")
        
        try:
            # Run script and capture output
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Log script output
            if result.stdout:
                logger.info(f"{script_name} output:\n{result.stdout}")
            if result.stderr:
                logger.warning(f"{script_name} errors:\n{result.stderr}")
            
            logger.info(f"Successfully completed {script_name}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running {script_name}: {str(e)}")
            if e.stdout:
                logger.error(f"Script output:\n{e.stdout}")
            if e.stderr:
                logger.error(f"Script errors:\n{e.stderr}")
            return False
        
        except Exception as e:
            logger.error(f"Unexpected error running {script_name}: {str(e)}")
            return False
    
    def run_pipeline(self) -> bool:
        """
        Run the complete prediction pipeline
        
        Returns:
            bool: True if all steps successful, False otherwise
        """
        pipeline_start = time.time()
        logger.info("Starting prediction pipeline")
        
        try:
            # Step 1: Collect historical matches (last 14 days)
            if not self.run_script("historical"):
                logger.error("Historical match collection failed")
                return False

            # Step 2: Calculate Elo ratings
            if not self.run_script("elo"):
                logger.error("Elo rating calculation failed")
                return False
                
            # Step 3: Collect upcoming matches
            if not self.run_script("collect"):
                logger.error("Match collection failed")
                return False
                
            # Step 4: Update completed matches
            if not self.run_script("update_completed"):
                logger.error("Completed match update failed")
                logger.warning("Continuing pipeline despite completed match update failure")
                # Continue pipeline even if completed match update fails
            
            # Step 5: Update historical features
            if not self.run_script("historical_features"):
                logger.error("Historical feature generation failed")
                return False
            
            # Step 6: Make predictions
            if not self.run_script("predict"):
                logger.error("Prediction generation failed")
                return False
            
            pipeline_duration = time.time() - pipeline_start
            logger.info(f"Pipeline completed successfully in {pipeline_duration:.1f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return False

def main():
    """Main execution function"""
    try:
        orchestrator = PipelineOrchestrator()
        success = orchestrator.run_pipeline()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 