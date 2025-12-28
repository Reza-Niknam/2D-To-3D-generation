# -----------------------------------------------------------------------------
# Background Removal with AI
# Author: Mohammad Reza Niknam
# Institution: Shahrood University of Technology
# Program: Master's in Telecommunications Systems
# Description: This code implements a personalized background removal tool
#              using artificial intelligence. All rights reserved.
# -----------------------------------------------------------------------------

# Import required libraries
import os, sys
import argparse
import shutil
import subprocess
from omegaconf import OmegaConf # Library for declarative configuration management (YAML)

# PyTorch Lightning imports
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.strategies import DDPStrategy # Strategy for Distributed Data Parallel training
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn

# Import utility for creating objects from configuration dictionaries
from src.utils.train_util import instantiate_from_config


@rank_zero_only
def rank_zero_print(*args):
    """
    Wrapper function for print that ensures output is only generated 
    by the global rank 0 process in a distributed setup.
    """
    print(*args)


def get_parser(**parser_kwargs):
    """
    Initializes and returns the argument parser for command-line options.
    """
    def str2bool(v):
        """Helper function to parse string arguments into boolean values."""
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        default=None,
        help="Path to the checkpoint file to resume training from.",
    )
    parser.add_argument(
        "--resume_weights_only",
        action="store_true",
        help="If set, only the model weights are loaded, ignoring trainer state.",
    )
    parser.add_argument(
        "-b",
        "--base",
        type=str,
        default="base_config.yaml",
        help="Path to the base configuration YAML file (using OmegaConf).",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="",
        help="A custom name for the experiment, used in the log directory path.",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Number of computational nodes to use for distributed training.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,",
        help="Comma-separated string of GPU IDs to use (e.g., '0,1,2,3').",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed for `seed_everything` for reproducibility.",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="Base directory where experiment logs and checkpoints will be saved.",
    )
    return parser


class SetupCallback(Callback):
    """
    A PyTorch Lightning Callback to handle experiment setup on the rank-zero process.
    This includes creating necessary directories and saving the configuration file.
    """
    def __init__(self, resume, logdir, ckptdir, cfgdir, config):
        super().__init__()
        self.resume = resume
        self.logdir = logdir # Root logging directory for the experiment
        self.ckptdir = ckptdir # Checkpoint directory
        self.cfgdir = cfgdir # Configuration files directory
        self.config = config # The OmegaConf configuration object

    def on_fit_start(self, trainer, pl_module):
        """
        Called when `trainer.fit()` starts. Only executes on the global rank 0 process.
        """
        if trainer.global_rank == 0:
            # Create the main logging and sub-directories
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            # Print and save the full configuration for documentation
            rank_zero_print("Project config")
            rank_zero_print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "project.yaml"))


class CodeSnapshot(Callback):
    """
    A Callback to save a snapshot of the current code base (tracked by git) 
    to the experiment log directory for reproducibility.
    Modified from https://github.com/threestudio-project/threestudio/blob/main/threestudio/utils/callbacks.py#L60
    """
    def __init__(self, savedir):
        self.savedir = savedir

    def get_file_list(self):
        """
        Uses git commands to list tracked and untracked (but not excluded) files.
        This ensures all relevant source code is captured.
        """
        return [
            b.decode()
            for b in set(
                # Get tracked files, excluding the configs directory
                subprocess.check_output(
                    'git ls-files -- ":!:configs/*"', shell=True
                ).splitlines()
            )
            | set( # Union with untracked files (hardcoded exclusion logic)
                subprocess.check_output(
                    "git ls-files --others --exclude-standard", shell=True
                ).splitlines()
            )
        ]

    @rank_zero_only
    def save_code_snapshot(self):
        """Copies all relevant source files to the savedir."""
        os.makedirs(self.savedir, exist_ok=True)
        for f in self.get_file_list():
            if not os.path.exists(f) or os.path.isdir(f):
                continue
            # Create subdirectories if necessary and copy the file
            os.makedirs(os.path.join(self.savedir, os.path.dirname(f)), exist_ok=True)
            shutil.copyfile(f, os.path.join(self.savedir, f))

    def on_fit_start(self, trainer, pl_module):
        """Saves the code snapshot when training begins."""
        try:
            self.save_code_snapshot()
        except:
            # Warn the user if git commands failed (e.g., not a git repo or git not installed)
            rank_zero_warn(
                "Code snapshot is not saved. Please make sure you have git installed and are in a git repository."
            )


if __name__ == "__main__":
    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    sys.path.append(os.getcwd())

    # --- Argument Parsing and Directory Setup ---
    parser = get_parser()
    opt, unknown = parser.parse_known_args() # Parse known args, ignoring unknown ones

    # Construct the experiment name and log directory path
    cfg_fname = os.path.split(opt.base)[-1]
    cfg_name = os.path.splitext(cfg_fname)[0]
    exp_name = "-" + opt.name if opt.name != "" else ""
    logdir = os.path.join(opt.logdir, cfg_name+exp_name)

    # Define paths for checkpoints, configs, and code snapshots within the log directory
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    codedir = os.path.join(logdir, "code")
    seed_everything(opt.seed) # Set the global seed

    # --- Configuration Loading and Trainer Setup ---
    # Load the base configuration file
    config = OmegaConf.load(opt.base)
    lightning_config = config.lightning # Access PyTorch Lightning specific configuration
    trainer_config = lightning_config.trainer
    
    # Configure hardware for GPU/DDP training
    trainer_config["accelerator"] = "gpu"
    rank_zero_print(f"Running on GPUs {opt.gpus}")
    ngpu = len(opt.gpus.strip(",").split(',')) # Count number of GPUs specified
    trainer_config['devices'] = ngpu # Set the number of devices for the trainer

    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # --- Model Initialization ---
    model = instantiate_from_config(config.model) # Instantiate the model from config
    # Resume model weights only if specified
    if opt.resume and opt.resume_weights_only:
        # Load checkpoint directly into a new instance of the model class
        model = model.__class__.load_from_checkpoint(opt.resume, **config.model.params)
    
    model.logdir = logdir # Attach the log directory to the model (useful for logging inside the module)

    # --- Trainer Arguments and Callbacks Setup ---
    trainer_kwargs = dict()

    # Logger Configuration (Default: TensorBoardLogger)
    default_logger_cfg = {
        "target": "pytorch_lightning.loggers.TensorBoardLogger",
        "params": {
            "name": "tensorboard",
            "save_dir": logdir, # Log directory is the root of the experiment
            "version": "0", # Use version 0, as the logdir already includes experiment name
        }
    }
    logger_cfg = OmegaConf.merge(default_logger_cfg)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    # Model Checkpoint Configuration
    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": ckptdir,
            "filename": "{step:08}", # Checkpoint file name includes the global step
            "verbose": True,
            "save_last": True, # Always save the latest checkpoint
            "every_n_train_steps": 5000, # Save a checkpoint every 5000 steps
            "save_top_k": -1, # Save all checkpoints (no pruning)
        }
    }

    if "modelcheckpoint" in lightning_config:
        modelckpt_cfg = lightning_config.modelcheckpoint
    else:
        modelckpt_cfg = OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)

    # Callbacks Configuration (SetupCallback, LR Monitor, Code Snapshot)
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "train.SetupCallback", # Custom callback to create dirs and save config
            "params": {
                "resume": opt.resume,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
            }
        },
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {
                "logging_interval": "step", # Log LR on every training step
            }
        },
        "code_snapshot": {
            "target": "train.CodeSnapshot", # Custom callback to save a git snapshot of the code
            "params": {
                "savedir": codedir,
            }
        },
    }
    default_callbacks_cfg["checkpoint_callback"] = modelckpt_cfg

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)

    # Instantiate all configured callbacks
    trainer_kwargs["callbacks"] = [
        instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
    
    # Set training precision and DDP strategy
    trainer_kwargs['precision'] = '32-true' # Use full 32-bit precision
    # Use Distributed Data Parallel, enabling finding unused parameters (common when model has different training/validation paths)
    trainer_kwargs["strategy"] = DDPStrategy(find_unused_parameters=True)

    # Instantiate the PyTorch Lightning Trainer
    trainer = Trainer(**trainer_config, **trainer_kwargs, num_nodes=opt.num_nodes)
    trainer.logdir = logdir # Store the final log directory path on the trainer object

    # --- Data Module Initialization ---
    data = instantiate_from_config(config.data) # Instantiate the DataModule
    data.prepare_data() # Execute data preparation steps (e.g., download datasets)
    data.setup("fit") # Setup data loaders for the 'fit' stage (training/validation)

    # --- Learning Rate Configuration ---
    base_lr = config.model.base_learning_rate
    # Get gradient accumulation steps from trainer config
    if 'accumulate_grad_batches' in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    rank_zero_print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    
    # NOTE: No automatic LR scaling is applied based on batch size/GPU count.
    model.learning_rate = base_lr
    rank_zero_print("++++ NOT USING LR SCALING ++++")
    rank_zero_print(f"Setting learning rate to {model.learning_rate:.2e}")

    # --- Training Execution ---
    if opt.resume and not opt.resume_weights_only:
        # Resume full training, including optimizer state and epoch count
        trainer.fit(model, data, ckpt_path=opt.resume)
    else:
        # Start new training or resumed only model weights
        trainer.fit(model, data)