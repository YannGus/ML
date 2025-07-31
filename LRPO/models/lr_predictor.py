from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple, Optional
import torch.nn as nn
import numpy as np
import torch


class LossPredictor(nn.Module):
    def __init__(self, input_size: int = 5, hidden_size: int = 64, output_size: int = 1) -> None:
        super(LossPredictor, self).__init__()
        self.fc1: nn.Linear = nn.Linear(input_size, hidden_size)
        self.fc2: nn.Linear = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3: nn.Linear = nn.Linear(hidden_size // 2, output_size)
        self.dropout: nn.Dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


class LearningRateOptimizer:
    def __init__(self, predictor_model: nn.Module, scaler: Optional[MinMaxScaler] = None) -> None:
        self.predictor: nn.Module = predictor_model
        self.scaler: MinMaxScaler = scaler or MinMaxScaler()

    def prepare_features(self, loss_history: List[float], lr_history: List[float]) -> np.ndarray:
        if len(loss_history) >= 2:
            loss_trend: float = loss_history[-1] - loss_history[-2]
            loss_variance: float = np.var(loss_history[-5:]) if len(loss_history) >= 5 else 0.0
            loss_slope: float = np.polyfit(np.arange(len(loss_history)), loss_history, 1)[0]
            current_lr: float = lr_history[-1] if lr_history else 0.01
            epoch_normalized: float = len(loss_history) / 100.0
            features: List[float] = [loss_trend, loss_variance, loss_slope, current_lr, epoch_normalized]
        else:
            features = [0.0, 0.0, 0.0, 0.01, 0.0]
        return np.array(features).reshape(1, -1)

    def predict_optimal_lr(
        self,
        loss_history: List[float],
        lr_history: List[float],
        target_epochs: int = 100
    ) -> float:
        features: np.ndarray = self.prepare_features(loss_history, lr_history)
        features_scaled: np.ndarray = self.scaler.transform(features)

        with torch.no_grad():
            features_tensor: torch.Tensor = torch.FloatTensor(features_scaled)
            predicted_loss_reduction: float = self.predictor(features_tensor).item()

        current_lr: float = lr_history[-1] if lr_history else 0.01
        remaining_epochs: int = target_epochs - len(loss_history)

        if remaining_epochs > 0 and predicted_loss_reduction < 0:
            lr_multiplier: float = min(abs(predicted_loss_reduction) * 10, 5.0)
            optimal_lr: float = current_lr * (1 + lr_multiplier)
        else:
            optimal_lr = current_lr * 0.9

        return max(optimal_lr, 1e-6)

    def train_predictor(
        self,
        training_data: Tuple[np.ndarray, np.ndarray],
        epochs: int = 50,
        lr: float = 0.001
    ) -> None:
        optimizer: torch.optim.Optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        criterion: nn.Module = nn.MSELoss()

        X, y = training_data
        X_scaled: np.ndarray = self.scaler.fit_transform(X)

        X_tensor: torch.Tensor = torch.FloatTensor(X_scaled)
        y_tensor: torch.Tensor = torch.FloatTensor(y).unsqueeze(1)

        self.predictor.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions: torch.Tensor = self.predictor(X_tensor)
            loss: torch.Tensor = criterion(predictions, y_tensor)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Predictor training epoch {epoch}, Loss: {loss.item():.6f}")
