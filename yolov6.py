# yolo_trainer_v6.py

"""
YOLO Training System Version 6
- Enhanced checkpoint management
- Anti-disconnect system
- Optimized for Google Colab T4 GPU
"""

import os
import gc
import shutil
import yaml
import logging
import torch
import cv2
import numpy as np
import time
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from tqdm import tqdm
from collections import defaultdict
from google.colab import drive
from IPython.display import display, Javascript

# Global configurations
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Configure logging with timestamp
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

def setup_anti_disconnect():
    """Setup anti-disconnect mechanism"""
    display(Javascript('''
        function ClickConnect(){
            console.log("Keeping connection alive..."); 
            document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click()
        }
        setInterval(ClickConnect, 60000)
        
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

class CheckpointManager:
    """Manage training checkpoints"""
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.checkpoint_dir = self.base_path / "models/yolo/weights"
        self.temp_checkpoint_dir = Path("container_detection/yolov8m_container/weights")
        self.checkpoint_history = []
        
        # Create checkpoint directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.temp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def find_latest_checkpoint(self) -> Optional[Path]:
        """Find most recent checkpoint"""
        checkpoints = [
            *list(self.checkpoint_dir.glob("*.pt")),
            *list(self.temp_checkpoint_dir.glob("*.pt"))
        ]
        if not checkpoints:
            return None
        
        # Sort by modification time
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        logger.info(f"Found latest checkpoint: {latest}")
        return latest
    
    def save_checkpoint(self, source_path: Path, checkpoint_type: str = "regular") -> Path:
        """Save checkpoint with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_path = self.checkpoint_dir / f"{checkpoint_type}_{timestamp}.pt"
        
        shutil.copy(source_path, dest_path)
        self.checkpoint_history.append(dest_path)
        
        # Keep only last 3 checkpoints to save space
        if len(self.checkpoint_history) > 3:
            old_checkpoint = self.checkpoint_history.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
                
        return dest_path
    
    def cleanup_temp_checkpoints(self):
        """Clean temporary checkpoint directory"""
        for file in self.temp_checkpoint_dir.glob("*.pt"):
            try:
                file.unlink()
            except Exception as e:
                logger.warning(f"Could not delete {file}: {e}")

class ResourceMonitor:
    """Monitor system resources"""
    @staticmethod
    def get_gpu_memory_info() -> Dict:
        """Get GPU memory information"""
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
        except Exception:
            return {}
    
    @staticmethod
    def get_system_memory_info() -> Dict:
        """Get system memory information"""
        mem = psutil.virtual_memory()
        return {
            'total': mem.total / 1e9,
            'available': mem.available / 1e9,
            'percent': mem.percent
        }
    
    @staticmethod
    def log_resource_usage():
        """Log current resource usage"""
        gpu_info = ResourceMonitor.get_gpu_memory_info()
        sys_info = ResourceMonitor.get_system_memory_info()
        
        logger.info("\nResource Usage:")
        if gpu_info:
            logger.info(f"GPU Memory: {gpu_info['used']:.1f}GB/{gpu_info['total']:.1f}GB ({gpu_info['utilization']:.1f}%)")
        logger.info(f"System Memory: {sys_info['percent']}% used, {sys_info['available']:.1f}GB available")
        
        
        
        
class YOLOTrainerV6:
    def __init__(self, base_path: str = "/content/drive/MyDrive/btran/container_number"):
        """Initialize YOLO Trainer V6"""
        # Disable wandb
        os.environ['WANDB_DISABLED'] = 'true'
        os.environ['WANDB_MODE'] = 'offline'
        os.environ['WANDB_SILENT'] = 'true'
        
        # Setup anti-disconnect
        setup_anti_disconnect()
        
        # Initialize paths
        self.base_path = Path(base_path)
        self.setup_paths()
        
        # Initialize managers and monitors
        self.checkpoint_manager = CheckpointManager(self.base_path)
        self.resource_monitor = ResourceMonitor()
        self.dataset_info = {}
        
        # Mount Google Drive
        if not Path('/content/drive').exists():
            drive.mount('/content/drive')
        
        # Create directories
        self._create_directories()
        
        # Set environment variable for memory allocation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Log initialization
        logger.info(f"Initialized YOLOTrainer V6")
        logger.info(f"Base path: {self.base_path}")
        self._log_system_info()

    def _create_directories(self):
        """Create necessary directories"""
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Verified directory: {path}")

    def _log_system_info(self):
        """Log system information"""
        logger.info("\nSystem Information:")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU: {gpu_name}")
            
            memory_info = torch.cuda.mem_get_info()
            total_memory = memory_info[1] / 1e9
            free_memory = memory_info[0] / 1e9
            logger.info(f"GPU Memory: {free_memory:.1f}GB free / {total_memory:.1f}GB total")
        else:
            logger.warning("No GPU available!")
        
        self.resource_monitor.log_resource_usage()

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
        
def analyze_dataset(self, sample_size: int = 100):
        """Analyze dataset characteristics"""
        logger.info("\n=== Dataset Analysis ===")
        
        # Monitor resources before analysis
        self.resource_monitor.log_resource_usage()
        
        # Analyze components
        image_info = self._analyze_images(sample_size)
        label_info = self._analyze_labels(sample_size)
        split_info = self._analyze_splits()
        
        # Store information
        self.dataset_info.update({
            'images': image_info,
            'labels': label_info,
            'splits': split_info,
            'timestamp': datetime.now().isoformat()
        })
        
        # Log summary
        self._log_dataset_summary()
        
        # Monitor resources after analysis
        self.resource_monitor.log_resource_usage()

    def _analyze_images(self, sample_size: int) -> Dict:
        """Analyze image characteristics"""
        logger.info("\nAnalyzing images...")
        
        try:
            image_files = list(self.paths['images'].glob('*.jpg'))
            total_count = len(image_files)
            
            if total_count == 0:
                logger.warning("No image files found!")
                return self._get_empty_image_info()
            
            # Sample images
            sample_files = np.random.choice(image_files, 
                                          size=min(sample_size, total_count), 
                                          replace=False)
            
            # Analyze samples
            image_sizes = defaultdict(int)
            corrupt_images = []
            size_stats = self._init_size_stats()
            valid_samples = 0
            
            for img_path in tqdm(sample_files, desc="Processing images"):
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        h, w = img.shape[:2]
                        size = f"{w}x{h}"
                        image_sizes[size] += 1
                        valid_samples += 1
                        
                        # Update stats
                        self._update_size_stats(size_stats, w, h)
                    else:
                        corrupt_images.append(img_path.name)
                except Exception as e:
                    corrupt_images.append(f"{img_path.name}: {str(e)}")
            
            # Calculate final statistics
            return self._calculate_image_stats(
                total_count, valid_samples, image_sizes, size_stats, corrupt_images
            )
            
        except Exception as e:
            logger.error(f"Image analysis failed: {str(e)}")
            return self._get_empty_image_info()

    def _analyze_labels(self, sample_size: int) -> Dict:
        """Analyze label characteristics"""
        logger.info("\nAnalyzing labels...")
        
        try:
            label_files = list(self.paths['labels'].glob('*.txt'))
            total_count = len(label_files)
            
            if total_count == 0:
                logger.warning("No label files found!")
                return self._get_empty_label_info()
            
            # Sample labels
            sample_files = np.random.choice(label_files, 
                                          size=min(sample_size, total_count), 
                                          replace=False)
            
            # Analyze samples
            stats = self._process_label_samples(sample_files)
            
            # Log findings
            self._log_label_analysis(stats)
            
            return stats
            
        except Exception as e:
            logger.error(f"Label analysis failed: {str(e)}")
            return self._get_empty_label_info()

    def _analyze_splits(self) -> Dict:
        """Analyze dataset splits"""
        logger.info("\nAnalyzing splits...")
        
        split_info = {}
        total_images = 0
        
        for split_name in ['train_split.txt', 'val_split.txt']:
            split_path = self.paths['splits'] / split_name
            if split_path.exists():
                with open(split_path, 'r') as f:
                    images = f.readlines()
                split_info[split_name] = {
                    'count': len(images),
                    'samples': images[:3],
                    'valid_files': self._verify_split_files(images)
                }
                total_images += len(images)
            else:
                split_info[split_name] = {
                    'count': 0,
                    'samples': [],
                    'valid_files': 0
                }
        
        split_info['total_images'] = total_images
        return split_info

    def prepare_training(self) -> bool:
        """Prepare for training"""
        try:
            logger.info("\n=== Preparing Training ===")
            
            # Verify environment
            if not self._verify_environment():
                return False
            
            # Prepare working directory
            if not self._prepare_working_directory():
                return False
            
            # Create and verify config
            if not self._create_training_config():
                return False
            
            # Check for existing checkpoints
            self._check_existing_checkpoints()
            
            logger.info("Training preparation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Training preparation failed: {str(e)}")
            return False
        
        
def _create_training_config(self) -> bool:
        """Create optimized training configuration"""
        try:
            logger.info("\nCreating training configuration...")
            
            # Use smaller image size for T4
            target_size = 640
            
            # Create data.yaml
            data_config = {
                'path': str(self.working_paths['dataset']),
                'train': str(self.working_paths['splits'] / 'train_split.txt'),
                'val': str(self.working_paths['splits'] / 'val_split.txt'),
                'nc': 1,
                'names': ['ContainerNumber']
            }
            
            config_path = self.working_paths['root'] / 'data.yaml'
            with open(config_path, 'w') as f:
                yaml.safe_dump(data_config, f)
            
            # Create training configuration
            self.train_config = {
                'data': str(config_path),
                'epochs': 100,
                'imgsz': target_size,
                'batch': 8,  # Conservative batch size for T4
                'device': 0,
                'workers': 4,
                
                # Training parameters
                'patience': 50,
                'save_period': 5,  # Save more frequently
                'exist_ok': True,
                'pretrained': True,
                'amp': True,
                'verbose': True,
                
                # Project
                'project': 'container_detection',
                'name': 'yolov8m_container',
                'save': True,
                'plots': False,
                
                # Memory optimization
                'cache': False,
                'multi_scale': False,
                
                # Optimization
                'optimizer': 'Adam',
                'lr0': 0.01,
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                
                # Reduced augmentation for memory efficiency
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 10,
                'translate': 0.1,
                'scale': 0.5,
                'shear': 2.0,
                'flipud': 0.5,
                'fliplr': 0.5,
                'mosaic': 0.5,
                'mixup': 0.0
            }
            
            logger.info(f"Created training config with target size: {target_size}")
            return True
            
        except Exception as e:
            logger.error(f"Training configuration creation failed: {str(e)}")
            return False

    def optimize_for_t4(self) -> int:
        """Optimize settings for T4 GPU"""
        try:
            # Clear memory
            gc.collect()
            torch.cuda.empty_cache()
            
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            
            # Check memory
            memory_info = torch.cuda.mem_get_info()
            free_memory = memory_info[0] / 1e9
            
            # Log current memory state
            self.resource_monitor.log_resource_usage()
            
            # Conservative batch size for stability
            return 8
            
        except Exception as e:
            logger.error(f"Optimization error: {str(e)}")
            return 8

    def train(self, resume: bool = False) -> Optional[Path]:
        """Train the model with enhanced checkpoint management"""
        try:
            logger.info("\n=== Starting Training ===")
            
            from ultralytics import YOLO
            
            # Optimize and set batch size
            batch_size = self.optimize_for_t4()
            self.train_config['batch'] = batch_size
            
            # Initialize model
            if resume and (checkpoint := self.checkpoint_manager.find_latest_checkpoint()):
                model = YOLO(checkpoint)
                self.train_config['resume'] = True
                logger.info(f"Resuming from checkpoint: {checkpoint}")
            else:
                model = YOLO('yolov8m.pt')
                logger.info("Starting new training")
            
            # Log training configuration
            logger.info("\nTraining configuration:")
            for key, value in self.train_config.items():
                logger.info(f"{key}: {value}")
            
            # Start training with periodic checks
            last_checkpoint_time = time.time()
            checkpoint_interval = 900  # 15 minutes in seconds
            
            try:
                results = model.train(**self.train_config)
                
                # Save final models
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                best_model = self.paths['weights'] / f'best_{timestamp}.pt'
                last_model = self.paths['weights'] / f'last_{timestamp}.pt'
                
                results_dir = Path(f"{self.train_config['project']}/{self.train_config['name']}/weights")
                if results_dir.exists():
                    if (results_dir / 'best.pt').exists():
                        shutil.copy(results_dir / 'best.pt', best_model)
                    if (results_dir / 'last.pt').exists():
                        shutil.copy(results_dir / 'last.pt', last_model)
                    
                logger.info("\nTraining completed successfully")
                logger.info(f"Best model saved: {best_model}")
                logger.info(f"Last model saved: {last_model}")
                
                # Save final checkpoint
                self.checkpoint_manager.save_checkpoint(best_model, "final")
                
                return best_model
                
            except Exception as e:
                # Try to save checkpoint if training fails
                if results_dir.exists() and (results_dir / 'last.pt').exists():
                    emergency_checkpoint = self.checkpoint_manager.save_checkpoint(
                        results_dir / 'last.pt',
                        "emergency"
                    )
                    logger.warning(f"Saved emergency checkpoint: {emergency_checkpoint}")
                raise e
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return None
        finally:
            # Cleanup
            self.cleanup()

    def cleanup(self):
        """Cleanup temporary files and resources"""
        try:
            logger.info("\nCleaning up...")
            
            # Remove working directory
            if self.working_dir.exists():
                shutil.rmtree(self.working_dir)
            
            # Remove temporary files
            temp_files = [
                'train_batch0.jpg',
                'val_batch0_labels.jpg',
                'val_batch0_pred.jpg',
                'confusion_matrix.png'
            ]
            
            for file in temp_files:
                if Path(file).exists():
                    os.remove(file)
            
            # Clear GPU memory
            gc.collect()
            torch.cuda.empty_cache()
            
            # Log final resource state
            self.resource_monitor.log_resource_usage()
            
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            
            
            
            
def setup_training_environment():
    """Setup and verify training environment"""
    try:
        # Setup anti-disconnect
        setup_anti_disconnect()
        
        # Mount Google Drive
        if not Path('/content/drive').exists():
            drive.mount('/content/drive')
        
        # Check GPU
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU available")
            
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {device_name}")
        
        # Install dependencies
        os.system('pip install -q ultralytics')
        
        return True
        
    except Exception as e:
        logger.error(f"Environment setup failed: {str(e)}")
        return False

def check_resume_training() -> bool:
    """Check if training can be resumed"""
    checkpoint_paths = [
        Path('container_detection/yolov8m_container/weights/last.pt'),
        Path('container_detection/yolov8m_container/weights/best.pt')
    ]
    
    for path in checkpoint_paths:
        if path.exists():
            return True
    return False

def main():
    """Main execution function with enhanced error handling and resume capability"""
    trainer = None
    try:
        logger.info("=== Container Number Detection Training System V6 ===")
        logger.info("Setting up environment...")
        
        if not setup_training_environment():
            raise RuntimeError("Environment setup failed")
        
        # Initialize trainer
        trainer = YOLOTrainerV6()
        
        # Check for existing training
        can_resume = check_resume_training()
        if can_resume:
            user_input = input("\nFound existing checkpoint. Resume training? (y/n): ")
            if user_input.lower() == 'y':
                logger.info("Resuming training...")
                best_model = trainer.train(resume=True)
                if best_model:
                    logger.info("\n=== Training Resumed and Completed Successfully ===")
                    logger.info(f"Best model saved at: {best_model}")
                return
        
        # Start new training process
        logger.info("\nStarting new training process...")
        
        # Analyze dataset
        logger.info("\nStep 1: Analyzing dataset...")
        trainer.analyze_dataset()
        
        # Ask for confirmation
        user_input = input("\nProceed with training preparation? (y/n): ")
        if user_input.lower() != 'y':
            logger.info("Training cancelled by user")
            return
        
        # Prepare training
        logger.info("\nStep 2: Preparing training...")
        if not trainer.prepare_training():
            raise RuntimeError("Training preparation failed")
        
        # Final confirmation
        user_input = input("\nStart training? (y/n): ")
        if user_input.lower() != 'y':
            logger.info("Training cancelled by user")
            if trainer:
                trainer.cleanup()
            return
        
        # Start training
        logger.info("\nStep 3: Training model...")
        best_model = trainer.train()
        if not best_model:
            raise RuntimeError("Training failed")
        
        logger.info("\n=== Training Pipeline Completed Successfully ===")
        logger.info(f"Best model saved at: {best_model}")
        
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        if trainer:
            # Try to save emergency checkpoint
            logger.info("Attempting to save emergency checkpoint...")
            trainer.checkpoint_manager.save_checkpoint(
                Path('container_detection/yolov8m_container/weights/last.pt'),
                "emergency_interrupt"
            )
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Cleanup in any case
        if trainer:
            try:
                trainer.cleanup()
            except:
                pass
        
        # Final resource check
        ResourceMonitor.log_resource_usage()

if __name__ == "__main__":
    main()