from typing import Dict, List, Tuple, Optional
import numpy as np
import torch

class ExperimentMetrics:
    """
    Collects and analyzes training and validation metrics during experiments.
    """
    
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.train_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.val_losses: List[float] = []
        self.val_accuracies: List[float] = []
        self.learning_rates: List[float] = []
        self.epoch_times: List[float] = []
        self.total_time: float = 0.0
        self.convergence_epoch: Optional[int] = None
        self.final_accuracy: float = 0.0
        self.best_accuracy: float = 0.0
        self.predictor_overhead: float = 0.0

    def update(
        self,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float,
        epoch_time: float
    ) -> None:
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)

        if val_acc > self.best_accuracy:
            self.best_accuracy = val_acc

    def set_convergence(self, epoch: int, threshold: float = 0.001) -> None:
        """
        Marks convergence if recent training loss is stable within a threshold.
        """
        if len(self.train_losses) >= 5:
            recent_losses = self.train_losses[-5:]
            if max(recent_losses) - min(recent_losses) < threshold:
                if self.convergence_epoch is None:
                    self.convergence_epoch = epoch

    def finalize(self, total_time: float) -> None:
        self.total_time = total_time
        self.final_accuracy = self.val_accuracies[-1] if self.val_accuracies else 0.0

    def get_summary(self) -> Dict[str, float]:
        return {
            "total_time": self.total_time,
            "total_epochs": len(self.train_losses),
            "final_train_loss": self.train_losses[-1] if self.train_losses else 0.0,
            "final_val_loss": self.val_losses[-1] if self.val_losses else 0.0,
            "final_accuracy": self.final_accuracy,
            "best_accuracy": self.best_accuracy,
            "convergence_epoch": self.convergence_epoch or len(self.train_losses),
            "avg_epoch_time": np.mean(self.epoch_times) if self.epoch_times else 0.0,
            "predictor_overhead": self.predictor_overhead,
            "final_lr": self.learning_rates[-1] if self.learning_rates else 0.0
        }


def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Computes top-1 accuracy for classification tasks.
    """
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == targets).sum().item()
    return 100.0 * correct / targets.size(0)


def generate_synthetic_training_data(num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates synthetic training data to train a learning rate predictor model.
    Each sample consists of handcrafted features and a corresponding target representing
    expected future loss reduction.
    """
    np.random.seed(42)
    X: List[List[float]] = []
    y: List[float] = []

    for _ in range(num_samples):
        initial_loss = np.random.uniform(2.0, 3.5)
        epochs = np.random.randint(5, 15)

        loss_history = []
        current_loss = initial_loss

        for _ in range(epochs):
            decay = np.random.uniform(0.85, 0.95)
            noise = np.random.normal(0, 0.05)
            current_loss = current_loss * decay + noise
            loss_history.append(max(current_loss, 0.1))

        loss_trend = loss_history[-1] - loss_history[-2] if len(loss_history) >= 2 else 0.0
        loss_variance = np.var(loss_history[-5:]) if len(loss_history) >= 5 else np.var(loss_history)
        loss_slope = np.polyfit(np.arange(len(loss_history)), loss_history, 1)[0]
        current_lr = np.random.uniform(0.001, 0.1)
        epoch_norm = len(loss_history) / 100.0

        features = [loss_trend, loss_variance, loss_slope, current_lr, epoch_norm]
        future_loss_reduction = loss_history[-1] - loss_history[0]

        X.append(features)
        y.append(future_loss_reduction)

    return np.array(X), np.array(y)
