from torch.utils.data import DataLoader, TensorDataset
from baseline_classifier.model import CNNClassifier
from token_classifier.model import TokenClassifier
from torchvision import datasets, transforms
from evaluation import evaluate_model
from typing import Any, Dict
from tqdm import tqdm
from torch import nn
import numpy as np
import json
import time
import torch
import os


CONFIG_PATH: str = "config.json"


def count_params(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a PyTorch model
    :model (nn.Module): The model to count parameters for
    :return: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_file_size(path: str) -> float:
    """
    Get the file size in kilobytes
    :path (str): Path to the file
    :return: Size of the file in kilobytes (KB)
    """
    return round(os.path.getsize(path) / 1024, 2)


def train_model(model: nn.Module, loader: DataLoader, device: torch.device, config: Dict[str, Any]) -> None:
    """
    Train a PyTorch model on the MNIST dataset.
    :model (nn.Module): The model to train
    :loader (DataLoader): DataLoader for the training dataset
    :device (torch.device): Device to run the model on (CPU or GPU)
    :config (Dict[str, Any]): Configuration dictionary containing training parameters
    :return: Nothing...
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    model.train()

    for _ in tqdm(range(config["epochs"]), desc="Training"):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def main() -> None:
    """
    Main training and evaluation routine
    Trains a baseline CNN and a token-based classifier on the MNIST dataset,
    evaluates both models, and saves results to disk.
    """
    with open(CONFIG_PATH, "r") as f:
        config: Dict[str, Any] = json.load(f)

    device: torch.device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)

    results: Dict[str, Any] = {}

    # Baseline CNN
    print("\nTraining baseline CNN model...")
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("data", train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    cnn = CNNClassifier()
    start = time.time()
    train_model(cnn, train_loader, device, config)
    end = time.time()
    torch.save(cnn.state_dict(), config["baseline_model_path"])

    acc, loss = evaluate_model(cnn, test_loader, device)
    results["baseline"] = {
        "accuracy": round(acc, 4),
        "loss": round(loss, 4),
        "train_time_sec": round(end - start, 2),
        "num_params": count_params(cnn),
        "model_size_kb": get_file_size(config["baseline_model_path"])
    }

    # Token Classifier
    print("\nTraining token classifier model...")
    token_data = np.load(config["token_path"])
    labels = torch.tensor(train_dataset.targets.numpy())
    X = torch.tensor(token_data[:len(labels)], dtype=torch.long)
    Y = labels
    token_loader = DataLoader(TensorDataset(X, Y), batch_size=config["batch_size"], shuffle=True)

    token_model = TokenClassifier()
    start = time.time()
    train_model(token_model, token_loader, device, config)
    end = time.time()
    torch.save(token_model.state_dict(), config["token_model_path"])

    acc, loss = evaluate_model(token_model, token_loader, device)
    results["token"] = {
        "accuracy": round(acc, 4),
        "loss": round(loss, 4),
        "train_time_sec": round(end - start, 2),
        "num_params": count_params(token_model),
        "model_size_kb": get_file_size(config["token_model_path"])
    }

    with open(config["results_path"], "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {config['results_path']}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
