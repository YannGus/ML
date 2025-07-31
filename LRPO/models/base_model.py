import torch.nn.functional as F
import torch.nn as nn
import torch


class ChallengingMNISTClassifier(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super(ChallengingMNISTClassifier, self).__init__()

        self.fc1: nn.Linear = nn.Linear(28 * 28, 256)
        self.fc2: nn.Linear = nn.Linear(256, 128)
        self.fc3: nn.Linear = nn.Linear(128, 64)
        self.fc4: nn.Linear = nn.Linear(64, 32)
        self.fc5: nn.Linear = nn.Linear(32, num_classes)

        self.dropout1: nn.Dropout = nn.Dropout(0.3)
        self.dropout2: nn.Dropout = nn.Dropout(0.4)
        self.dropout3: nn.Dropout = nn.Dropout(0.3)

        self.bn1: nn.BatchNorm1d = nn.BatchNorm1d(256)
        self.bn2: nn.BatchNorm1d = nn.BatchNorm1d(128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


SimpleClassifier = ChallengingMNISTClassifier
SimpleMNISTClassifier = ChallengingMNISTClassifier
