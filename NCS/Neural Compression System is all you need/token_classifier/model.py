import torch.nn as nn

class TokenClassifier(nn.Module):
    def __init__(self, vocab_size=128, embed_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(49 * embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.embedding(x)
        return self.classifier(x)
