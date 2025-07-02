import os

import hydra
import mlflow
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

from anomaly_detection.models.baseline import BaselineModule
from anomaly_detection.pipelines.utils import get_git_info

mlflow.pytorch.autolog()


@hydra.main(
    version_base=None, config_path="../configs", config_name="baseline"
)
def main(cfg: DictConfig):
    # Set random seed for reproducibility
    pl.seed_everything(42)
    torch.manual_seed(42)

    # Get git information
    git_info = get_git_info()

    # Initialize MLFlow
    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.experiment.name,
        tracking_uri="file:./ml-runs",
        tags={
            "dataset_directory": cfg.dataset.directory,
            "dataset_version": cfg.dataset.version,
            "commit_id": git_info["commit_id"],
            "branch": git_info["branch"],
        },
    )

    # Log git diff as an artifact
    with open("git_diff.txt", "w") as f:
        f.write(git_info["diff"])

    # Initialize model
    model = BaselineModule(
        num_epochs=cfg.training.num_epochs,
        num_classes=cfg.model_parameters.num_classes,
        learning_rate=cfg.model_parameters.learning_rate,
        weight_decay=cfg.model_parameters.weight_decay,
        pretrained=cfg.model_parameters.pretrained,
        freeze_backbone=cfg.model_parameters.freeze_backbone,
    )

    # Define transformations
    transform = transforms.Compose(
        [
            transforms.Resize(cfg.transform.resize_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    # Initialize datasets and dataloaders
    full_train_dataset = ImageFolder(
        cfg.dataset.directory + "/train/images", transform=transform
    )

    # Method 1: Simple random split (may not preserve class balance perfectly)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    test_dataset = ImageFolder(
        cfg.dataset.directory + "/validation/images", transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    # Model checkpointing
    model_checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.output.output_dir, "checkpoints"),
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        save_last=True,
    )

    print("Intended batch size:", cfg.training.intended_batch_size)
    print("Batch size:", cfg.training.batch_size)
    accumulated_grad_batches = int(
        np.ceil(cfg.training.intended_batch_size / cfg.training.batch_size)
    )
    print("Accumulated grad batches:", accumulated_grad_batches)

    # Enable mixed precision training
    precision = (
        "bf16-mixed"
        if (torch.cuda.is_bf16_supported() or not torch.cuda.is_available())
        else "16-mixed"
    )

    # Initialize trainer
    trainer = pl.Trainer(
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        max_epochs=cfg.training.num_epochs,
        logger=mlflow_logger,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        callbacks=[model_checkpoint_callback],
        accumulate_grad_batches=accumulated_grad_batches,
        precision=precision,
        log_every_n_steps=5,
    )

    # Log the model for deployment
    # 1. Save the model code
    with mlflow.start_run(run_id=mlflow_logger.run_id):

        # Train model
        trainer.fit(model, train_loader, val_loader)

        # Test model
        trainer.test(model, test_loader)

        # Log git diff as artifact
        mlflow.log_artifact("git_diff.txt")

        os.remove("git_diff.txt")


if __name__ == "__main__":
    main()
