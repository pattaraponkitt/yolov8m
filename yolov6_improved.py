"""
YOLO Training System Version 6 (Improved)
Optimized for Google Colab T4 GPU with enhanced checkpoint management
and anti-disconnect features.
"""

# Standard library imports
import os
import gc
import shutil
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple

# Third party imports
import yaml
import logging
import torch
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from google.colab import drive
from IPython.display import display, Javascript

# Constants
RANDOM_SEED = 42
CHECKPOINT_INTERVAL = 900  # 15 minutes
BATCH_SIZE = 8
TARGET_SIZE = 640
DEFAULT_EPOCHS = 100

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def setup_anti_disconnect():
    """Setup anti-disconnect mechanism for Google Colab"""
    display(Javascript('''
        function ClickConnect(){
            console.log("Keeping connection alive..."); 
            document.querySelector("#top-toolbar > colab-connect-button")
                .shadowRoot.querySelector("#connect").click()
        }
        setInterval(ClickConnect, 60000);
        
        // Monitor for disconnections
        let connected = true;
        setInterval(() => {
            const toolbar = document.querySelector("#top-toolbar");
            const status = toolbar ? toolbar.getAttribute("connection-status") : "ok";
            if (status !== "ok" && connected) {
                connected = false;
                console.log("Disconnected, attempting to reconnect...");
                ClickConnect();
            } else if (status === "ok" && !connected) {
                connected = true;
                console.log("Reconnected successfully!");
            }
        }, 5000);
    '''))

class ResourceMonitor:
    """Monitor and manage system resources during training."""
    
    @staticmethod
    def get_gpu_info() -> Dict[str, float]:
        """
        Get GPU memory information.
        
        Returns:
            Dict containing GPU memory statistics
        """
        try:
            memory_info = torch.cuda.mem_get_info()
            total = memory_info[1] / 1e9
            free = memory_info[0] / 1e9
            used = total - free
            return {
                'total': total,
                'free': free,
                'used': used,
                'utilization': (used / total) * 100
            }
        except Exception as e:
            logger.error(f"GPU info error: {e}")
            return {}

    @staticmethod
    def log_system_status():
        """Log current system resource usage."""
        try:
            # GPU Info
            gpu_info = ResourceMonitor.get_gpu_info()
            if gpu_info:
                logger.info(
                    f"GPU Memory: {gpu_info['used']:.1f}GB/"
                    f"{gpu_info['total']:.1f}GB "
                    f"({gpu_info['utilization']:.1f}%)"
                )
            
            # System Memory
            import psutil
            mem = psutil.virtual_memory()
            logger.info(
                f"RAM: {mem.percent}% used, "
                f"{mem.available/1e9:.1f}GB available"
            )
        except Exception as e:
            logger.error(f"Error logging system status: {e}")

class CheckpointManager:
    """Manage training checkpoints."""
    
    def __init__(self, base_path: Path):
        """
        Initialize CheckpointManager.
        
        Args:
            base_path: Base directory for saving checkpoints
        """
        self.base_path = base_path
        self.checkpoint_dir = base_path / "models/yolo/weights"
        self.temp_dir = Path("container_detection/yolov8m_container/weights")
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_history = []
        
    def find_latest_checkpoint(self) -> Optional[Path]:
        """
        Find the most recent checkpoint.
        
        Returns:
            Path to latest checkpoint if found, None otherwise
        """
        try:
            checkpoints = [
                *list(self.checkpoint_dir.glob("*.pt")),
                *list(self.temp_dir.glob("*.pt"))
            ]
            
            if not checkpoints:
                return None
            
            latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
            logger.info(f"Found latest checkpoint: {latest}")
            return latest
            
        except Exception as e:
            logger.error(f"Error finding checkpoint: {e}")
            return None
            
    def save_checkpoint(self, source_path: Path, checkpoint_type: str = "regular") -> Optional[Path]:
        """
        Save a checkpoint with timestamp.
        
        Args:
            source_path: Source checkpoint path
            checkpoint_type: Type of checkpoint (regular/emergency)
            
        Returns:
            Path to saved checkpoint if successful, None otherwise
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest_path = self.checkpoint_dir / f"{checkpoint_type}_{timestamp}.pt"
            
            shutil.copy(source_path, dest_path)
            self.checkpoint_history.append(dest_path)
            
            # Keep only last 3 checkpoints
            if len(self.checkpoint_history) > 3:
                old_checkpoint = self.checkpoint_history.pop(0)
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    
            return dest_path
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            return None



class YOLOTrainerV6:
    """
    YOLO Training System Version 6 with enhanced features.
    
    Features:
        - Checkpoint management
        - Anti-disconnect system
        - Resource monitoring
        - Automatic cleanup
        - Enhanced error handling
    """
    
    def __init__(self, base_path: str = "/content/drive/MyDrive/btran/container_number"):
        """
        Initialize YOLO Trainer.
        
        Args:
            base_path: Base directory for all training data and models
        """
        # Disable wandb
        os.environ['WANDB_DISABLED'] = 'true'
        os.environ['WANDB_MODE'] = 'offline'
        os.environ['WANDB_SILENT'] = 'true'
        
        # Setup anti-disconnect
        setup_anti_disconnect()
        
        # Initialize paths
        self.base_path = Path(base_path)
        self.setup_paths()
        
        # Initialize components
        self.checkpoint_manager = CheckpointManager(self.base_path)
        self.resource_monitor = ResourceMonitor()
        self.dataset_info = {}
        
        # Mount Google Drive
        if not Path('/content/drive').exists():
            drive.mount('/content/drive')
        
        # Create directories
        self._create_directories()
        
        # Set memory allocation strategy
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Log initialization
        logger.info(f"\n{'-'*20} YOLOTrainer V6 Initialized {'-'*20}")
        logger.info(f"Base path: {self.base_path}")
        self._log_system_info()

    def setup_paths(self):
        """Setup system paths"""
        # Drive paths
        self.paths = {
            'base': self.base_path,
            'dataset': self.base_path / "dataset",
            'images': self.base_path / "dataset/images",
            'labels': self.base_path / "dataset/labels",
            'splits': self.base_path / "dataset/splits",
            'models': self.base_path / "models/yolo",
            'configs': self.base_path / "models/yolo/configs",
            'weights': self.base_path / "models/yolo/weights",
            'results': self.base_path / "results/detection",
            'logs': self.base_path / "logs"
        }
        
        # Working paths (local)
        self.working_dir = Path("/content/yolo_training")
        self.working_paths = {
            'root': self.working_dir,
            'dataset': self.working_dir / "dataset",
            'images': self.working_dir / "dataset/images",
            'labels': self.working_dir / "dataset/labels",
            'splits': self.working_dir / "dataset/splits",
            'weights': self.working_dir / "weights",
            'results': self.working_dir / "results"
        }

    def _create_directories(self):
        """Create necessary directories"""
        for path in [*self.paths.values(), *self.working_paths.values()]:
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Verified directory: {path}")

    def _log_system_info(self):
        """Log system information"""
        logger.info("\nSystem Information:")
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU: {device_name}")
        else:
            logger.warning("No GPU available!")
        
        self.resource_monitor.log_system_status()
