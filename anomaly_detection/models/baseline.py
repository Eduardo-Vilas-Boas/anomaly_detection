import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy, F1Score, Recall
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


class BaselineModule(pl.LightningModule):
    def __init__(
        self,
        num_epochs,
        num_classes,
        learning_rate,
        weight_decay,
        pretrained,
        freeze_backbone,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load model with or without pretrained weights
        self.model = mobilenet_v3_large(
            weights=MobileNet_V3_Large_Weights.DEFAULT
        )

        # Replace the classifier for transfer learning
        in_features = self.model.classifier[0].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 1280),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, num_classes),
        )

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.model.features.parameters():
                param.requires_grad = False

        # Set the device for the model
        self.accuracy_metric_func = Accuracy(
            task="multiclass", num_classes=num_classes, average="macro"
        ).to(self.device)
        self.f1_score_metric_func = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        ).to(self.device)
        self.recall_metric_func = Recall(
            task="multiclass", num_classes=num_classes, average="macro"
        ).to(self.device)

    def forward(self, x):
        return self.model(x)

    def common_step(self, prefix, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)

        self.log(f"{prefix}_loss", loss, sync_dist=True, on_epoch=True)
        self.log(
            f"{prefix}_acc",
            self.accuracy_metric_func(preds, y),
            sync_dist=True,
            on_epoch=True,
        )
        self.log(
            f"{prefix}_f1",
            self.f1_score_metric_func(preds, y),
            sync_dist=True,
            on_epoch=True,
        )
        self.log(
            f"{prefix}_recall",
            self.recall_metric_func(preds, y),
            sync_dist=True,
            on_epoch=True,
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self.common_step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.common_step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.common_step("test", batch, batch_idx)

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.num_epochs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }
