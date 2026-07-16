from SR_model.edsr import EDSR
from .vae_models.vanilla_vae import VanillaVAE
import torch
import torch.nn as nn


def tensor_mean_shift(img: torch.Tensor):
    img_max, img_min = img.max(), img.min()
    img_output = (img - img_min) / (img_max - img_min)
    return img_output


class DIVAESR(nn.Module):
    def __init__(self, sr_args, vae_args):
        super(DIVAESR, self).__init__()
        self.sr_args = sr_args
        self.vae_in_channels = vae_args['in_channels']
        self.vae_latent_dim = vae_args['latent_dim']
        self.vae = VanillaVAE(self.vae_in_channels, self.vae_latent_dim)
        # input: (Tensor) Img to encoder [N x C x H x W] output: (Tensor) List of latent codes
        self.vae_encoder = self.vae.encoder
        # input: param z (Tensor) [B x D] output: (Tensor) [B x C x H x W]
        self.vae_decoder = self.vae.decoder
        self.mean_shift = tensor_mean_shift
        self.sr_decoder = EDSR(self.sr_args)

    def forward(self, img: torch.Tensor, label: torch.Tensor, hr_label: torch.Tensor):
        # step1: vae model
        vae_output_noise = self.vae(img, label)
        vae_output = img - vae_output_noise[0]
        # adjust vae model output to [0, 1]
        sr_input = self.mean_shift(vae_output)
        # step2: sr_decoder
        sr_output = self.sr_decoder(sr_input)

        return [vae_output_noise, vae_output, sr_output, hr_label]


