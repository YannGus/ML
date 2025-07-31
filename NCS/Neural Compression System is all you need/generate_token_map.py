from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model.vqvae import VQVAE
from typing import Any
from tqdm import tqdm
import numpy as np
import torch
import json
import os


def generate_token_map() -> None:
    """
    Generates a token map by encoding MNIST images using a trained VQ-VAE model,
    and saves the resulting discrete latent codes (tokens) to a .npy file.

    The function reads the model and output paths from a config file (`config.json`),
    loads the VQ-VAE model, processes the MNIST training dataset, encodes each image,
    and stores the resulting tokens as a NumPy array.
    """
    with open("config.json", "r") as f:
        config: dict[str, Any] = json.load(f)

    codebook_size: int = 128
    device: torch.device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    model: VQVAE = VQVAE(num_embeddings=codebook_size).to(device)
    model.load_state_dict(torch.load(config["vqvae_model_path"], map_location=device))
    model.eval()

    dataset = datasets.MNIST(root="data", train=True, transform=transforms.ToTensor())
    loader: DataLoader = DataLoader(dataset, batch_size=1, shuffle=False)

    os.makedirs(os.path.dirname(config["token_path"]), exist_ok=True)
    all_tokens: list[np.ndarray] = []

    with torch.no_grad():
        for img, _ in tqdm(loader, desc="Tokenizing"):
            img = img.to(device)
            z_e = model.encoder(img)
            _, _, indices = model.vq(z_e)
            tokens: np.ndarray = indices.view(-1).cpu().numpy().astype(np.uint8)
            all_tokens.append(tokens)

    tokens_np: np.ndarray = np.stack(all_tokens)
    np.save(config["token_path"], tokens_np)

    print(f"Tokens saved to {config['token_path']}.npy with shape {tokens_np.shape}")


if __name__ == "__main__":
    generate_token_map()
