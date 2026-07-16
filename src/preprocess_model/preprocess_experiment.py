import os
import math
import torch
from torch import optim
import pytorch_lightning as pl
import torchvision.utils as vutils
from preprocess_model.preprocess_model_loss import loss_function_total


class Prexpriment(pl.LightningModule):

    def __init__(self,
                 ensemble_model,
                 params: dict) -> None:
        super(Prexpriment, self).__init__()

        self.model = ensemble_model
        self.loss_function = loss_function_total
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: torch.Tensor, label: torch.Tensor, lr_labels: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(input, label, lr_labels, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels, hr_labels, hr_cat_labels = batch
        self.curr_device = real_img.device
        results = self.forward(real_img, labels, hr_labels)
        results.append(hr_cat_labels)

        train_loss = self.loss_function(*results, M_N=self.params['kld_weight'],  # al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels, hr_labels, hr_cat_labels = batch
        self.curr_device = real_img.device
        results = self.forward(real_img, labels, hr_labels)
        results.append(hr_cat_labels)

        val_loss = self.loss_function(*results, M_N=self.params['kld_weight'],  # real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def on_validation_end(self) -> None:
        self.sample_images()

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label, hr_label, hr_cat_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        hr_label = hr_label.to(self.curr_device)
        hr_cat_label = hr_cat_label.to(self.curr_device)

        results = self.forward(test_input, test_label, hr_label)
        # [vae_output_noise, vae_output, sr_output, hr_label]
        vae_output_noise, vae_output, sr_output, output_label = results[0], results[1], results[2], results[3]

        vutils.save_image(vae_output.cpu().data,
                          os.path.join(self.logger.log_dir,
                                       "VAE_Reconstructions",
                                       f"VAE_recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        vutils.save_image(sr_output.cpu().data,
                          os.path.join(self.logger.log_dir,
                                       "SR_Reconstructions",
                                       f"SR_recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        vutils.save_image(output_label.cpu().data,
                          os.path.join(self.logger.log_dir,
                                       "Samples",
                                       f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims
