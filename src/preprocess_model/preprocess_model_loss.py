import torch
import torch.nn as nn
from torch.nn import functional as F


def loss_function_total(*args, **kwargs) -> dict:
    vae_output_noise, sr_output, hr_label, hr_cat_labels = args[0], args[2], args[3], args[-1]
    vae_loss_input = vae_output_noise
    vae_loss = vae_loss_function(*vae_loss_input, M_N=kwargs['M_N'])
    attention_map = create_attention_map(hr_cat_labels)
    print("attn min/max :", attention_map.min().item(), attention_map.max().item())
    sr_loss = weighted_loss(sr_output, hr_label, attention_map)
    total_loss = vae_loss['loss'] + sr_loss * 0.1
    return {'loss': total_loss, 'VAE_Loss': vae_loss['loss'], 'sr_loss': sr_loss.detach()}


def vae_loss_function(*args, **kwargs) -> dict:
    """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
    """
    recons = args[0]
    input = args[1]
    label = args[2]
    mu = args[3]
    log_var = args[4]
    noise = input - label

    kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
    # recons_loss = F.mse_loss(recons, input)
    recons_loss = F.mse_loss(recons, noise)

    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

    loss = recons_loss + kld_weight * kld_loss
    return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}


def weighted_loss(sr_output, hr_label, attention_map):
    # Calculate per-pixel loss for main framework
    per_pixel_loss_framework = F.l1_loss(sr_output, hr_label, reduction='none')
    
    # Apply the attention map (weighting) for small molecule area
    weighted_per_pixel_loss = per_pixel_loss_framework * attention_map

    # Sum and average for final loss
    total_loss = torch.mean(weighted_per_pixel_loss)
    
    return total_loss


def create_attention_map(binary_attention_map):
    # create weight attention map
    return binary_attention_map * 7 + 1
