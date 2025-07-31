from torch.utils.data import DataLoader
from torch.nn import Module
from typing import Tuple
from torch import nn
import torch


def evaluate_model(model: Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """
    Evaluate a PyTorch model on a given DataLoader
    :model: The model to evaluate
    :loader: DataLoader containing the evaluation dataset
    :device: The device to run the evaluation on (CPU or GPU)
    :return: A tuple containing the accuracy and average loss
    """
    model.eval()
    correct: int = 0
    total: int = 0
    total_loss: float = 0.0
    loss_fn: nn.Module = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            total_loss += loss_fn(preds, y).item()
            correct += (preds.argmax(dim=1) == y).sum().item()
            total += y.size(0)

    accuracy: float = correct / total if total > 0 else 0.0
    average_loss: float = total_loss / total if total > 0 else 0.0

    return accuracy, average_loss
