"""Training and evaluation loop for the NER model."""

from __future__ import annotations

import logging
import os

import swanlab
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from utils import NERMetrics

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model,
        config,
        device,
        train_loader,
        dev_loader,
        test_loader,
        id2label,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.metrics = NERMetrics(id2label)

        parameter_groups = [
            {"params": model.bert.parameters(), "lr": config.bert_lr},
            {"params": model.bilstm.parameters(), "lr": config.lstm_lr},
            {"params": model.classifier.parameters(), "lr": config.classifier_lr},
        ]
        self.optimizer = AdamW(parameter_groups, weight_decay=config.weight_decay)
        total_steps = len(train_loader) * config.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * config.warmup_ratio),
            num_training_steps=total_steps,
        )

        self.best_dev_f1 = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.best_model_state = None
        os.makedirs(config.output_dir, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        progress = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")

        for batch_index, batch in enumerate(progress):
            batch = {name: tensor.to(self.device) for name, tensor in batch.items()}
            self.optimizer.zero_grad()
            outputs = self.model(**batch)
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")
            if batch_index % 10 == 0:
                swanlab.log(
                    {"train/loss": loss.item(), "train/lr": self.scheduler.get_last_lr()[0]}
                )

        return total_loss / len(self.train_loader)

    def evaluate(self, dataloader, desc="Evaluating"):
        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc):
                batch = {name: tensor.to(self.device) for name, tensor in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs["loss"].item()
                self.metrics.update(outputs["predictions"], batch["labels"])

        results = self.metrics.compute()
        results["loss"] = total_loss / len(dataloader) if len(dataloader) else 0.0
        return results

    def train(self):
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(epoch)
            dev_results = self.evaluate(self.dev_loader, desc="Validating")
            dev_f1 = float(dev_results["f1"])
            swanlab.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "dev/f1": dev_f1,
                    "dev/loss": dev_results["loss"],
                }
            )
            logger.info(
                "Epoch %d/%d: train_loss=%.4f, dev_f1=%.4f",
                epoch + 1,
                self.config.epochs,
                train_loss,
                dev_f1,
            )

            if dev_f1 > self.best_dev_f1:
                self.best_dev_f1 = dev_f1
                self.best_epoch = epoch + 1
                self.patience_counter = 0
                self.best_model_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.model.state_dict().items()
                }
                self._save_model(f"best_model_epoch_{epoch + 1}")
            else:
                self.patience_counter += 1

            if (epoch + 1) % 5 == 0:
                self._save_model(f"checkpoint_epoch_{epoch + 1}")
            if self.patience_counter >= self.config.patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        self.model.to(self.device)
        test_results = self.evaluate(self.test_loader, desc="Testing")
        self._save_model("final_model")
        return float(test_results["f1"]), str(test_results["report"])

    def _save_model(self, name):
        save_path = os.path.join(self.config.output_dir, f"{name}.pt")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config.__dict__,
                "best_dev_f1": self.best_dev_f1,
                "best_epoch": self.best_epoch,
            },
            save_path,
        )
        logger.info("Saved model to %s", save_path)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        return checkpoint