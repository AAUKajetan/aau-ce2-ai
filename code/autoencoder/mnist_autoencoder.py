"""
Fully connected autoencoder for MNIST
  Input:      28x28 -> 784
  Latent dim: 32
  Loss:       MSE
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

LATENT_DIM = 32

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
logging.info(f"Using device: {device}")

# ── 1. Data ───────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),                          # scales to [0, 1]
    transforms.Lambda(lambda x: x.view(-1)),        # 1x28x28 -> 784
])
logging.info("Loading MNIST dataset...")
train_dataset = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
logging.info(f"Number of training samples: {len(train_dataset)}")
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
logging.info(f"Number of test samples: {len(test_dataset)}")
logging.info("MNIST dataset loaded successfully.")

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True,  num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_dataset,  batch_size=1024, shuffle=False, num_workers=0, pin_memory=False)

logging.info("Data loaders created successfully.")

# ── 2. Model ──────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, input_dim: int = 784, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
            nn.Linear(128, latent_dim),nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM, output_dim: int = 784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256),        nn.ReLU(),
            nn.Linear(256, output_dim), nn.Sigmoid(),   # outputs in [0, 1]
        )

    def forward(self, z):
        return self.net(z)


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int = 784, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)



model = Autoencoder(input_dim=784, latent_dim=LATENT_DIM).to(device)
logging.info(f"Autoencoder model initialized with latent dimension {LATENT_DIM}.")
logging.debug(f"Model architecture:\n{model}")
logging.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ── 3. Training ───────────────────────────────────────────────────────────────
NUM_EPOCHS    = 40
LEARNING_RATE = 1e-3

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses = []

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for images, _ in train_loader:
        images = images.to(device)
        loss   = criterion(model(images), images)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_dataset)
    train_losses.append(epoch_loss)
    logging.info(f"Epoch [{epoch:2d}/{NUM_EPOCHS}]  MSE Loss: {epoch_loss:.6f}")

# Plot training loss
plt.figure(figsize=(8, 4))
plt.plot(range(1, NUM_EPOCHS + 1), train_losses, marker="o", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss.png", dpi=150)
plt.show()

# ── 4. Visualize Reconstructions ──────────────────────────────────────────────
model.eval()
NUM_DISPLAY = 10

test_images, _ = next(iter(test_loader))
test_images = test_images.to(device)

with torch.no_grad():
    reconstructed = model(test_images).cpu()

test_images_cpu = test_images.cpu()

fig, axes = plt.subplots(2, NUM_DISPLAY, figsize=(NUM_DISPLAY * 1.4, 3))
for i in range(NUM_DISPLAY):
    axes[0, i].imshow(test_images_cpu[i].view(28, 28), cmap="gray")
    axes[0, i].axis("off")
    if i == 0:
        axes[0, i].set_title("Original", fontsize=9)

    axes[1, i].imshow(reconstructed[i].view(28, 28), cmap="gray")
    axes[1, i].axis("off")
    if i == 0:
        axes[1, i].set_title("Reconstructed", fontsize=9)

plt.suptitle("Original vs Reconstructed MNIST Digits", y=1.02)
plt.tight_layout()
plt.savefig("reconstructions.png", dpi=150)
plt.show()

# ── 5. Latent Space (t-SNE) ───────────────────────────────────────────────────
all_latents, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        z = model.encode(images.to(device)).cpu().numpy()
        all_latents.append(z)
        all_labels.append(labels.numpy())

all_latents = np.concatenate(all_latents)
all_labels  = np.concatenate(all_labels)

#logging.info("Running t-SNE on latent vectors...")
#tsne = TSNE(n_components=2, random_state=42, perplexity=30)
#latents_2d = tsne.fit_transform(all_latents)

#plt.figure(figsize=(9, 7))
#scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1],
 #                     c=all_labels, cmap="tab10", s=5, alpha=0.7)
#plt.colorbar(scatter, ticks=range(10), label="Digit class")
#plt.title("t-SNE of 32-dim Latent Space")
#plt.xlabel("t-SNE 1")
#plt.ylabel("t-SNE 2")
#plt.tight_layout()
#plt.savefig("latent_space_tsne.png", dpi=150)
#   plt.show()
