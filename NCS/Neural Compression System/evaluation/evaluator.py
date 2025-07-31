from evaluation.metrics import mse, ssim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import List, Tuple, Any
import matplotlib.pyplot as plt
from models.vqvae import VQVAE
from tqdm import tqdm
import numpy as np
import torch
import json
import os


def save_image(tensor: torch.Tensor, path: str) -> None:
    """
    Save a tensor as an image file
    :param tensor: The tensor to save, expected to be a single-channel image
    :param path: The file path where the image will be saved
    returns: Nothing...
    """
    arr: np.ndarray = tensor.squeeze().cpu().numpy()
    plt.imsave(path, arr, cmap="gray")


def evaluate_codebook_sizes() -> List[Tuple[int, float, float, float]]:
    """
    Evaluate the VQVAE model with different codebook sizes on the MNIST dataset
    :return: A list of tuples containing codebook size, average MSE, average SSIM, and token size in KB
    """
    config: dict[str, Any] = json.load(open("config.json"))
    device: torch.device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    transform: transforms.Compose = transforms.ToTensor()
    dataset: datasets.MNIST = datasets.MNIST(
        config["data_dir"], train=True, download=True, transform=transform
    )
    train_loader: DataLoader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True
    )
    test_loader: DataLoader = DataLoader(dataset, batch_size=1, shuffle=False)

    os.makedirs(config["model_dir"], exist_ok=True)
    os.makedirs(config["recon_dir"], exist_ok=True)

    results: List[Tuple[int, float, float, float]] = []

    for codebook_size in config["codebook_sizes"]:
        model: VQVAE = VQVAE(num_embeddings=codebook_size).to(device)
        optimizer: torch.optim.Optimizer = torch.optim.Adam(
            model.parameters(), lr=config["learning_rate"]
        )

        for _ in range(config["epochs"]):
            for x, _ in tqdm(train_loader, desc=f"Training {codebook_size}"):
                x = x.to(device)
                x_hat: torch.Tensor
                vq_loss: torch.Tensor
                x_hat, vq_loss, _ = model(x)
                loss: torch.Tensor = torch.nn.functional.mse_loss(x_hat, x) + vq_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        torch.save(model.state_dict(), f"{config['model_dir']}/vqvae_{codebook_size}.pt")

        model.eval()
        mse_total: float = 0.0
        ssim_total: float = 0.0
        recon_path: str = os.path.join(config["recon_dir"], str(codebook_size))
        os.makedirs(recon_path, exist_ok=True)

        with torch.no_grad():
            for i, (img, _) in enumerate(test_loader):
                if i >= config["num_eval_images"]:
                    break
                img = img.to(device)
                x_hat, _, _ = model(img)

                save_image(img, f"{recon_path}/original_{i}.png")
                save_image(x_hat, f"{recon_path}/img_{i}.png")

                mse_total += mse(x_hat, img)
                ssim_total += ssim(x_hat, img)

        avg_mse: float = mse_total / config["num_eval_images"]
        avg_ssim: float = ssim_total / config["num_eval_images"]
        token_size_kb: float = config["num_eval_images"] * (np.ceil(np.log2(codebook_size)) / 8)

        results.append((codebook_size, avg_mse, avg_ssim, token_size_kb))

    return results
