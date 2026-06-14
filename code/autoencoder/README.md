# MNIST Autoencoder

Module 9 experiment. Trains a small autoencoder on MNIST and visualizes
reconstructions and the latent space.

- `mnist_autoencoder.py` / `mnist_autoencoder.ipynb` — the same experiment as a script and a notebook.
- `data/` — MNIST dataset (downloaded by the script).
- `outputs/` — generated plots: `training_loss.png`, `reconstructions.png`, `latent_space_tsne.png`.
- `requirements.txt` — `pip install -r requirements.txt` (torch, torchvision, numpy, matplotlib, scikit-learn).

Note: the script writes the PNGs to the current working directory, so run it
from inside this folder (`cd code/autoencoder && python mnist_autoencoder.py`)
to keep outputs in `outputs/`, or adjust the `savefig` paths.
