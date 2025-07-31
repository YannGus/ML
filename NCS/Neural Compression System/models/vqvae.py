import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class Encoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=128, latent_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, latent_dim, 4, 2, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

class Decoder(nn.Module):
    def __init__(self, out_channels=1, latent_dim=64):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.deconv(x)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = commitment_cost

    def forward(self, x: Tensor):
        b, c, h, w = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, c)

        distances = (x_flat.pow(2).sum(1, keepdim=True)
                     - 2 * x_flat @ self.embedding.weight.t()
                     + self.embedding.weight.pow(2).sum(1))

        indices = torch.argmin(distances, 1)
        quantized = self.embedding(indices).view(b, h, w, c).permute(0, 3, 1, 2)

        commitment_loss = F.mse_loss(quantized.detach(), x)
        embedding_loss = F.mse_loss(quantized, x.detach())
        loss = commitment_loss + self.beta * embedding_loss

        return quantized, loss, indices.view(b, h, w)

class VQVAE(nn.Module):
    def __init__(self, num_embeddings=8192, embedding_dim=64):
        super().__init__()
        self.encoder = Encoder(latent_dim=embedding_dim)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = Decoder(latent_dim=embedding_dim)

    def forward(self, x: Tensor):
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.vq(z_e)
        x_hat = self.decoder(z_q)
        return x_hat, vq_loss, indices
