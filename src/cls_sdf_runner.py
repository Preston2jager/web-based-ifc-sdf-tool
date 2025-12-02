
import os
import time
import shutil
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from . import cls_sdf_model as sdf_model
from . import cls_sdf_dataset as dataset


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()


class SDF_Runner:
    """Coordinator class for training DeepSDF models.
    
    Handles model initialization, data loading, training loop, validation,
    and checkpoint management. Supports both fresh training and fine-tuning
    from pretrained models.
    """

    def __init__(self, cfg, target_folder: Optional[str] = None):
        """Initialize SDF runner with configuration.
        
        Args:
            cfg: Configuration object containing training parameters and paths.
            target_folder: Optional path to pretrained model for fine-tuning.
                          If None, starts fresh training.
        """
        self._init_paths(cfg, target_folder)
        self._init_hyperparameters(cfg)
        self._init_model_and_optimizers()

    # ========== Internal API ==========

    def _get_project_root(self) -> str:
        """Get project root directory (parent of src/).
        
        Returns:
            Absolute path to project root directory.
        """
        current_file = Path(__file__).resolve()
        src_dir = current_file.parent
        project_root = src_dir.parent
        return str(project_root)

    def _init_paths(self, cfg, target_folder: Optional[str]) -> None:
        """Initialize all file paths for training.
        
        Args:
            cfg: Configuration object.
            target_folder: Optional pretrained model folder path.
        """
        if target_folder is None:
            self.root_dir = self._get_project_root()
            self.timestamp_run = datetime.now().strftime('%d_%m_%H%M%S')
            self.runs_base_dir = cfg.Pathes.Trained_SDF_folder_path
            self.Converted_SDF_folder_path = cfg.Pathes.Converted_SDF_folder_path
            self.run_dir = os.path.join(self.runs_base_dir, self.timestamp_run)
            self.src_cfg_path = os.path.join(
                self.root_dir, 
                "Data/Training/Configs/configs.yaml"
            )
            self.target_cfg_path = os.path.join(self.run_dir, 'configs.yaml')
            self._training_folder_creation()
            shutil.copy2(self.src_cfg_path, self.target_cfg_path)
            self.samples_dict_path = os.path.join(
                self.Converted_SDF_folder_path, 
                'samples_dict.npy'
            )
        else:
            self.run_dir = target_folder
            self.Converted_SDF_folder_path = target_folder
            self.samples_dict_path = os.path.join(target_folder, 'samples_dict.npy')

    def _init_hyperparameters(self, cfg) -> None:
        """Initialize training hyperparameters from configuration.
        
        Args:
            cfg: Configuration object.
        """
        self.num_layer = cfg.Train.Num_layers
        self.skip_connections = cfg.Train.Skip_connections
        self.inner_dim = cfg.Train.Inner_dim
        self.latent_size = cfg.Train.Latent_size
        self.lr_model = cfg.Train.Lr_model
        self.lr_latent = cfg.Train.Lr_latent
        self.lr_multiplier = cfg.Train.Lr_multiplier
        self.lr_scheduler_status = cfg.Train.Lr_scheduler
        self.patience = cfg.Train.Patience
        self.sigma_regulariser = cfg.Train.Sigma_regulariser
        self.epochs = cfg.Train.Epochs
        self.batch_size = cfg.Train.Batch_size
        self.seed = cfg.Train.Seed

    def _init_model_and_optimizers(self) -> None:
        """Initialize model, latent codes, optimizers, and schedulers."""
        self.writer = SummaryWriter(log_dir=self.run_dir)
        self.model = sdf_model.SDFModel(
            self.num_layer,
            self.skip_connections,
            self.latent_size,
            self.inner_dim
        ).float().to(device)

        self.samples_dict = np.load(self.samples_dict_path, allow_pickle=True).item()
        self.latent_codes = self._generate_latent_codes(
            self.latent_size, 
            self.samples_dict
        )

        self.optimizer_model = optim.Adam(
            self.model.parameters(), 
            lr=self.lr_model, 
            weight_decay=0
        )
        self.optimizer_latent = optim.Adam(
            [self.latent_codes], 
            lr=self.lr_latent, 
            weight_decay=0
        )

        if self.lr_scheduler_status:
            self.scheduler_model = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_model,
                mode='min',
                factor=self.lr_multiplier,
                patience=self.patience,
                threshold=0.0001,
                threshold_mode='rel'
            )
            self.scheduler_latent = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_latent,
                mode='min',
                factor=self.lr_multiplier,
                patience=self.patience,
                threshold=0.0001,
                threshold_mode='rel'
            )

    def _get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders.
        
        Returns:
            Tuple of (train_loader, val_loader).
        """
        data = dataset.SDF_dataset(self.Converted_SDF_folder_path)
        g = torch.Generator().manual_seed(self.seed)
        train_size = int(0.7 * len(data))
        val_size = len(data) - train_size
        train_data, val_data = random_split(
            data, 
            [train_size, val_size], 
            generator=g
        )

        train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=12,
            drop_last=True,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=12,
            drop_last=False,
            pin_memory=True
        )
        return train_loader, val_loader

    def _train(self, train_loader: DataLoader) -> float:
        """Execute one training epoch.
        
        Args:
            train_loader: DataLoader for training data.
            
        Returns:
            Average training loss for the epoch.
        """
        total_loss = 0.0
        self.model.train()

        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            self.optimizer_model.zero_grad()
            self.optimizer_latent.zero_grad()

            x, y, latent_codes_indices_batch, latent_codes_batch = self._generate_xy(batch)
            x = x.to(device)
            y = y.to(device)
            latent_codes_indices_batch = latent_codes_indices_batch.to(device)
            latent_codes_batch = latent_codes_batch.to(device)

            predictions = self.model(x)
            loss_value, loss_rec, loss_latent = self._sdf_loss(
                y, 
                predictions, 
                x[:, :self.latent_size], 
                sigma=self.sigma_regulariser
            )

            if batch_idx % 50 == 0:
                with torch.no_grad():
                    per_point_loss = (y - predictions).pow(2).view(-1)
                    k = int(0.05 * per_point_loss.numel())
                    if k > 0:
                        topk_vals, _ = torch.topk(per_point_loss, k)

            loss_value.backward()
            self.optimizer_latent.step()
            self.optimizer_model.step()
            total_loss += loss_value.item()

        avg_train_loss = total_loss / (batch_idx + 1)
        print(f'Training: loss {avg_train_loss}, L1: {loss_rec}, L2: {loss_latent}')
        self.writer.add_scalar('Training loss', avg_train_loss, self.epoch)
        return avg_train_loss

    def _validate(self, val_loader: DataLoader) -> float:
        """Execute validation on validation set.
        
        Args:
            val_loader: DataLoader for validation data.
            
        Returns:
            Average validation loss.
        """
        total_loss = 0.0
        total_loss_rec = 0.0
        total_loss_latent = 0.0
        iterations = 0.0
        self.model.eval()

        with torch.no_grad():
            for batch in val_loader:
                iterations += 1.0
                x, y, _, latent_codes_batch = self._generate_xy(batch)
                x = x.to(device)
                y = y.to(device)
                latent_codes_batch = latent_codes_batch.to(device)

                predictions = self.model(x)
                loss_value, loss_rec, loss_latent = self._sdf_loss(
                    y, 
                    predictions, 
                    latent_codes_batch, 
                    self.sigma_regulariser
                )

                total_loss += loss_value.item()
                total_loss_rec += loss_rec.item()
                total_loss_latent += loss_latent.item()

            avg_val_loss = total_loss / iterations
            avg_loss_rec = total_loss_rec / iterations
            avg_loss_latent = total_loss_latent / iterations

            print(f'Validation: loss {avg_val_loss}')
            self.writer.add_scalar('Validation loss', avg_val_loss, self.epoch)
            self.writer.add_scalar('Reconstruction loss', avg_loss_rec, self.epoch)
            self.writer.add_scalar('Latent code loss', avg_loss_latent, self.epoch)
            return avg_val_loss

    def _sdf_loss(
        self, 
        sdf: torch.Tensor, 
        prediction: torch.Tensor, 
        x_latent: torch.Tensor, 
        sigma: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute DeepSDF loss function for multiple shapes.
        
        Args:
            sdf: Ground truth SDF values.
            prediction: Predicted SDF values.
            x_latent: Latent code vectors.
            sigma: Regularization weight for latent codes.
            
        Returns:
            Tuple of (total_loss, reconstruction_loss, latent_regularization_loss).
        """
        l1 = torch.mean(torch.abs(prediction - sdf))
        l2 = sigma**2 * torch.mean(torch.linalg.norm(x_latent, dim=1, ord=2))
        loss = l1 + l2
        return loss, l1, l2

    def _generate_latent_codes(
        self, 
        latent_size: int, 
        samples_dict: Dict[int, Any]
    ) -> torch.Tensor:
        """Generate initial latent codes for all shapes.
        
        Handles sparse shape indices by allocating tensor size based on maximum index.
        Only initializes latent codes for shapes present in samples_dict.
        
        Args:
            latent_size: Dimension of latent code vectors.
            samples_dict: Dictionary mapping shape indices to sample data.
            
        Returns:
            Tensor of latent codes with shape (num_latent, latent_size).
            
        Raises:
            RuntimeError: If samples_dict is empty.
        """
        if not samples_dict:
            raise RuntimeError("samples_dict is empty; no shapes to train on.")

        max_idx = max(samples_dict.keys())
        num_latent = max_idx + 1

        latent_codes = torch.zeros(
            (num_latent, latent_size),
            dtype=torch.float32,
            device=device,
        )

        for obj_idx in samples_dict.keys():
            latent_codes[obj_idx] = torch.normal(
                0.0,
                0.01,
                size=(latent_size,),
                dtype=torch.float32,
                device=device,
            )

        latent_codes.requires_grad_(True)
        print(
            f"[Runner] latent_codes created: num_latent={num_latent}, "
            f"max_idx={max_idx}, num_samples_dict_keys={len(samples_dict)}"
        )
        return latent_codes

    def _generate_xy(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate model inputs from batch data with safety checks.
        
        Filters out samples with invalid latent indices to prevent CUDA errors.
        
        Args:
            batch: Tuple of (input_tensor, target_tensor) from DataLoader.
            
        Returns:
            Tuple of (x, y, latent_indices, latent_codes) where:
                - x: Model input (latent_code + coordinates)
                - y: Target SDF values
                - latent_indices: Valid latent code indices
                - latent_codes: Corresponding latent code vectors
                
        Raises:
            RuntimeError: If all samples in batch have invalid indices.
        """
        batch[0] = batch[0].to(device, non_blocking=True)
        batch[1] = batch[1].to(device, non_blocking=True)

        latent_classes_batch = batch[0][:, 0].view(-1).to(torch.long)
        coords = batch[0][:, 1:]
        y = batch[1].view(-1, 1)

        num_latent = self.latent_codes.shape[0]
        valid_mask = (latent_classes_batch >= 0) & (latent_classes_batch < num_latent)

        if not torch.all(valid_mask):
            num_invalid = (~valid_mask).sum().item()
            print(
                f"[Warning] _generate_xy: {num_invalid} samples have invalid "
                f"latent indices (min={latent_classes_batch.min().item()}, "
                f"max={latent_classes_batch.max().item()}, "
                f"num_latent={num_latent}). These samples will be skipped."
            )

            latent_classes_batch = latent_classes_batch[valid_mask]
            coords = coords[valid_mask]
            y = y[valid_mask]

        if latent_classes_batch.numel() == 0:
            raise RuntimeError(
                "All samples in the current batch have invalid latent indices; "
                "check samples_dict and idx_int2str_dict consistency."
            )

        latent_codes_batch = self.latent_codes[latent_classes_batch]
        x = torch.hstack((latent_codes_batch, coords))

        return x, y, latent_classes_batch, latent_codes_batch

    def _copy_training_idx_files(
        self, 
        source_path: str, 
        target_path: str
    ) -> None:
        """Copy index mapping files to training directory.
        
        Args:
            source_path: Source file path.
            target_path: Destination file path.
        """
        if os.path.exists(source_path):
            idx_dict = np.load(source_path, allow_pickle=True).item()
            np.save(target_path, idx_dict)
            print(f'Saved index dict to {target_path}')
        else:
            print(f'Warning: {source_path} not found! Index dict not saved.')

    def _training_folder_creation(self) -> None:
        """Create training directory and copy necessary index files."""
        os.makedirs(self.run_dir, exist_ok=True)

        source_sample_dict_path = os.path.join(
            self.Converted_SDF_folder_path, 
            'samples_dict.npy'
        )
        target_sample_dict_path = os.path.join(self.run_dir, 'samples_dict.npy')
        source_idx_int2str_path = os.path.join(
            self.Converted_SDF_folder_path, 
            'idx_int2str_dict.npy'
        )
        target_idx_int2str_path = os.path.join(self.run_dir, 'idx_int2str_dict.npy')
        source_idx_str2int_path = os.path.join(
            self.Converted_SDF_folder_path, 
            'idx_str2int_dict.npy'
        )
        target_idx_str2int_path = os.path.join(self.run_dir, 'idx_str2int_dict.npy')

        self._copy_training_idx_files(source_sample_dict_path, target_sample_dict_path)
        self._copy_training_idx_files(source_idx_int2str_path, target_idx_int2str_path)
        self._copy_training_idx_files(source_idx_str2int_path, target_idx_str2int_path)

    # ========== Public API ==========

    def execute(self) -> None:
        """Execute the complete training loop.
        
        Trains the model for the specified number of epochs, validates after each epoch,
        and saves the best model checkpoint based on validation loss.
        """
        train_loader, val_loader = self._get_loaders()
        self.results = {'best_latent_codes': []}
        best_loss = float('inf')
        self.target_status = False
        start = time.time()

        for epoch in range(self.epochs):
            print(f'============================ Epoch {epoch} ============================')
            self.epoch = epoch
            avg_train_loss = self._train(train_loader)

            with torch.no_grad():
                avg_val_loss = self._validate(val_loader)
                if avg_val_loss < best_loss:
                    best_loss = np.copy(avg_val_loss)
                    best_weights = self.model.state_dict()
                    best_latent_codes = self.latent_codes.detach().cpu().numpy()
                    optimizer_model_state = self.optimizer_model.state_dict()
                    optimizer_latent_state = self.optimizer_latent.state_dict()

                    self.results['best_latent_codes'] = best_latent_codes
                    np.save(os.path.join(self.run_dir, 'results.npy'), self.results)
                    torch.save(best_weights, os.path.join(self.run_dir, 'weights.pt'))
                    torch.save(
                        optimizer_model_state, 
                        os.path.join(self.run_dir, 'optimizer_model_state.pt')
                    )
                    torch.save(
                        optimizer_latent_state, 
                        os.path.join(self.run_dir, 'optimizer_latent_state.pt')
                    )

                if self.lr_scheduler_status:
                    self.scheduler_model.step(avg_val_loss)
                    self.scheduler_latent.step(avg_val_loss)
                    self.writer.add_scalar(
                        'Learning rate (model)', 
                        self.scheduler_model._last_lr[0], 
                        epoch
                    )
                    self.writer.add_scalar(
                        'Learning rate (latent)', 
                        self.scheduler_latent._last_lr[0], 
                        epoch
                    )

                if best_loss < 0.008 and not self.target_status:
                    self.target_status = True

        end = time.time()
        print(f'Time elapsed: {end - start} s')

    def load_pretrained(self, folder_path: str) -> None:
        """Load pretrained model weights and optimizer states for fine-tuning.
        
        Args:
            folder_path: Path to directory containing pretrained checkpoint files.
        """
        pretrain_weights_path = os.path.join(folder_path, "weights.pt")
        pretrain_optim_model = os.path.join(folder_path, "optimizer_model_state.pt")
        pretrain_optim_latent = os.path.join(folder_path, "optimizer_latent_state.pt")
        results_path = os.path.join(folder_path, "results.npy")

        results_latent_codes = np.load(results_path, allow_pickle=True).item()
        self.model.load_state_dict(
            torch.load(pretrain_weights_path, map_location=device)
        )
        self.latent_codes = torch.tensor(
            results_latent_codes['best_latent_codes']
        ).float().to(device)
        self.latent_codes.requires_grad_()

        self.optimizer_model.load_state_dict(
            torch.load(pretrain_optim_model, map_location=device)
        )
        self.optimizer_latent = optim.Adam(
            [self.latent_codes], 
            lr=self.lr_latent, 
            weight_decay=0
        )
        self.optimizer_latent.load_state_dict(
            torch.load(pretrain_optim_latent, map_location=device)
        )

        if self.lr_scheduler_status:
            self.scheduler_model = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_model,
                mode='min',
                factor=self.lr_multiplier,
                patience=self.patience,
                threshold=0.0001,
                threshold_mode='rel'
            )
            self.scheduler_latent = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_latent,
                mode='min',
                factor=self.lr_multiplier,
                patience=self.patience,
                threshold=0.0001,
                threshold_mode='rel'
            )