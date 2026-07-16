from .base import *
from .vanilla_vae import *
# from .twostage_vae import *
from .vq_vae import *


# Aliases
VAE = VanillaVAE
GaussianVAE = VanillaVAE

vae_models = {'VQVAE': VQVAE,
              'VanillaVAE': VanillaVAE,
              }
