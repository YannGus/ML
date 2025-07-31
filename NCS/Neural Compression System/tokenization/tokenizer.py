
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.vqvae import VQVAE
from tqdm import tqdm
import numpy as np
import torch
import json
import os

def generate_token_map(codebook_size: int) -> None:
    """
    Generate a token map for the MNIST dataset using a VQVAE model
    :param codebook_size: Number of embeddings in the VQVAE codebook
    :return: Nothing...
    """
    config = json.load(open("config.json"))
    
    os.makedirs(os.path.dirname(config["token_path"]), exist_ok=True)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    model = VQVAE(num_embeddings=codebook_size).to(device)
    model.load_state_dict(torch.load(f"{config['model_dir']}/vqvae_{codebook_size}.pt", map_location=device))
    model.eval()

    dataset = datasets.MNIST(config["data_dir"], train=True, download=False, transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_tokens = []

    with torch.no_grad():
        for img, _ in tqdm(loader, desc="Tokenizing images"):
            img = img.to(device)
            z_e = model.encoder(img)
            _, _, indices = model.vq(z_e) 
            tokens = indices.view(-1).cpu().numpy().astype(np.uint8)
            all_tokens.append(tokens)

    token_array = np.stack(all_tokens, axis=0)

    np.save(config["token_path"], token_array)

    print(f"Token map saved to: {config['token_path']}.npy with shape {token_array.shape}")
    
