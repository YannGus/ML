from utils.metrics import ExperimentMetrics, generate_synthetic_training_data
from models.lr_predictor import LossPredictor, LearningRateOptimizer
from models.base_model import ChallengingMNISTClassifier
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from config import Config
from typing import Tuple
import torch.optim as optim
import torch.nn as nn
import torchvision
import torch
import time
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PredictorTrainer:
    def __init__(self, config: Config = Config()) -> None:
        self.config: Config = config
        self.device: torch.device = config.DEVICE
        self.metrics: ExperimentMetrics = ExperimentMetrics()

        self.train_loader, self.val_loader = self._prepare_data()

        self.model: nn.Module = ChallengingMNISTClassifier(num_classes=config.NUM_CLASSES).to(self.device)
        self.criterion: nn.Module = nn.CrossEntropyLoss()
        self.optimizer: optim.Optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.INITIAL_LR,
            momentum=0.9,
            weight_decay=1e-4
        )

        self.predictor: LossPredictor = LossPredictor().to(self.device)
        self.lr_optimizer: LearningRateOptimizer = LearningRateOptimizer(self.predictor)

        self.loss_history: list[float] = []
        self.lr_history: list[float] = []

    def _prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])

        trainset = torchvision.datasets.FashionMNIST(
            root='./data', train=True, download=True, transform=transform
        )
        valset = torchvision.datasets.FashionMNIST(
            root='./data', train=False, download=True, transform=transform
        )

        if self.config.USE_SUBSET:
            train_indices = torch.randperm(len(trainset))[:self.config.TRAIN_SUBSET_SIZE]
            val_indices = torch.randperm(len(valset))[:self.config.VAL_SUBSET_SIZE]
            trainset = Subset(trainset, train_indices)
            valset = Subset(valset, val_indices)

        train_loader = DataLoader(trainset, batch_size=self.config.BATCH_SIZE, shuffle=True, num_workers=self.config.NUM_WORKERS)
        val_loader = DataLoader(valset, batch_size=self.config.BATCH_SIZE, shuffle=False, num_workers=self.config.NUM_WORKERS)

        return train_loader, val_loader

    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss: float = 0.0
        total_correct: int = 0
        total_samples: int = 0

        for data, targets in self.train_loader:
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == targets).sum().item()
            total_samples += targets.size(0)

        avg_loss: float = total_loss / len(self.train_loader)
        accuracy: float = 100.0 * total_correct / total_samples

        return avg_loss, accuracy

    def validate(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss: float = 0.0
        total_correct: int = 0
        total_samples: int = 0

        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                total_correct += (outputs.argmax(1) == targets).sum().item()
                total_samples += targets.size(0)

        avg_loss: float = total_loss / len(self.val_loader)
        accuracy: float = 100.0 * total_correct / total_samples

        return avg_loss, accuracy

    def train_predictor(self) -> None:
        predictor_start: float = time.time()
        X, y = generate_synthetic_training_data(num_samples=500)
        self.lr_optimizer.train_predictor(
            (X, y),
            epochs=self.config.PREDICTOR_EPOCHS,
            lr=self.config.PREDICTOR_LR
        )
        predictor_time: float = time.time() - predictor_start
        self.metrics.predictor_overhead = predictor_time

    def calculate_jump_lr(self) -> float:
        if len(self.loss_history) >= 3:
            recent_loss_trend = self.loss_history[-3:]
            loss_reduction_rate: float = (recent_loss_trend[0] - recent_loss_trend[-1]) / 3.0
            current_lr: float = self.lr_history[-1]

            if loss_reduction_rate > 0:
                multiplier: float = min(self.config.JUMP_LR_MULTIPLIER, 1 + loss_reduction_rate * 10)
                jump_lr: float = current_lr * multiplier
            else:
                jump_lr = current_lr * 0.8

            jump_lr = max(jump_lr, current_lr * 0.5)
            jump_lr = min(jump_lr, current_lr * self.config.JUMP_LR_MULTIPLIER)

            return jump_lr

        return self.config.INITIAL_LR

    def train(self) -> ExperimentMetrics:
        start_time: float = time.time()

        for epoch in range(self.config.PRETRAINING_EPOCHS):
            epoch_start: float = time.time()
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            epoch_time: float = time.time() - epoch_start
            current_lr: float = self.optimizer.param_groups[0]['lr']

            self.loss_history.append(train_loss)
            self.lr_history.append(current_lr)
            self.metrics.update(train_loss, train_acc, val_loss, val_acc, current_lr, epoch_time)

        self.train_predictor()

        jump_lr: float = self.calculate_jump_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = jump_lr

        for epoch in range(self.config.JUMP_EPOCHS):
            epoch_start: float = time.time()
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            epoch_time: float = time.time() - epoch_start
            current_lr: float = self.optimizer.param_groups[0]['lr']

            self.loss_history.append(train_loss)
            self.lr_history.append(current_lr)
            self.metrics.update(train_loss, train_acc, val_loss, val_acc, current_lr, epoch_time)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config.FINE_TUNE_LR

        for epoch in range(self.config.FINE_TUNING_EPOCHS):
            epoch_start: float = time.time()
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            epoch_time: float = time.time() - epoch_start
            current_lr: float = self.optimizer.param_groups[0]['lr']

            self.metrics.update(train_loss, train_acc, val_loss, val_acc, current_lr, epoch_time)
            total_epoch: int = len(self.metrics.train_losses)
            self.metrics.set_convergence(total_epoch)

        total_time: float = time.time() - start_time
        self.metrics.finalize(total_time)

        return self.metrics


def run_predictor_experiment() -> ExperimentMetrics:
    torch.manual_seed(Config.RANDOM_SEED)
    trainer: PredictorTrainer = PredictorTrainer()
    metrics: ExperimentMetrics = trainer.train()
    return metrics


if __name__ == "__main__":
    metrics: ExperimentMetrics = run_predictor_experiment()
    print(metrics.get_summary())
