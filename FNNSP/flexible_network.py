import torch.nn.functional as F
from typing import Tuple
from torch import Tensor
import torch.nn as nn
import torch


class WorkerModule(nn.Module):
    """Single worker module with two linear layers.

    Args:
        in_features (int): Input dimension.
        out_features (int): Output dimension for task-specific classification.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 32)
        self.fc2 = nn.Linear(32, out_features)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class Worker(nn.Module):
    """A collection of worker modules combined with a weighted sum according to a configuration mask.

    Args:
        in_features (int): Input dimension.
        out_features (int): Output dimension for each worker module.
        num_modules (int): Number of worker modules.
    """

    def __init__(self, in_features: int, out_features: int, num_modules: int = 3):
        super().__init__()
        self.modules_list = nn.ModuleList(
            [WorkerModule(in_features, out_features) for _ in range(num_modules)]
        )
        self.num_modules = num_modules

    def forward(self, x: Tensor, config_mask: Tensor) -> Tensor:
        out = torch.zeros(x.size(0), self.modules_list[0].fc2.out_features, device=x.device)
        for i, module in enumerate(self.modules_list):
            mask = config_mask[:, i].unsqueeze(1)
            out += module(x) * mask
        return out


class Controller(nn.Module):
    """Controller network generating a task-specific mask to activate/deactivate worker modules.

    Args:
        in_features (int): Input dimension.
        num_modules (int): Number of worker modules.
    """

    def __init__(self, in_features: int, num_modules: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 64)
        self.fc2 = nn.Linear(64, num_modules)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        mask = torch.sigmoid(self.fc2(x))
        return mask


class FlexibleNetwork(nn.Module):
    """Flexible neural network with structural plasticity combining a controller and worker modules.

    Args:
        in_features (int): Input dimension.
        out_features (int): Output dimension for classification.
        num_modules (int): Number of worker modules.
    """

    def __init__(self, in_features: int, out_features: int, num_modules: int = 3):
        super().__init__()
        self.worker = Worker(in_features, out_features, num_modules)
        self.controller = Controller(in_features, num_modules)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        config_mask = self.controller(x)
        out = self.worker(x, config_mask)
        return out, config_mask
