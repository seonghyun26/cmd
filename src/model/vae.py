import os
import torch
import lightning

import numpy as np
import pandas as pd

from typing import Any, Optional, Tuple

from mlcolvar.cvs import BaseCV,VariationalAutoEncoderCV
from mlcolvar.core.loss.mse import mse_loss
from mlcolvar.core import FeedForward, Normalization
from mlcolvar.core.transform.utils import Inverse
from mlcolvar.utils.io import create_dataset_from_files
from mlcolvar.data import DictModule


class ELBOGaussiansLossBeta(torch.nn.Module):
    def forward(
        self,
        target: torch.Tensor,
        output: torch.Tensor,
        mean: torch.Tensor,
        log_variance: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        beta: float = 4.0,
    ) -> torch.Tensor:
        return elbo_gaussians_loss_beta(target, output, mean, log_variance, weights, beta)


def elbo_gaussians_loss_beta(
    target: torch.Tensor,
    output: torch.Tensor,
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    beta: float = 4.0,
) -> torch.Tensor:
    kl = -0.5 * (log_variance - log_variance.exp() - mean**2 + 1).sum(dim=1)

    # Weighted mean over batches.
    if weights is None:
        kl = kl.mean()
    else:
        weights = weights.squeeze()
        if weights.shape != kl.shape:
            raise ValueError(
                f"weights should be a tensor of shape (n_batches,) or (n_batches,1), not {weights.shape}."
            )
        kl = (kl * weights).sum()

    # Reconstruction loss.
    reconstruction = mse_loss(output, target, weights=weights)

    return reconstruction, beta * kl


class VariationalAutoEncoderCVBeta(BaseCV, lightning.LightningModule):
	BLOCKS = ["norm_in", "encoder", "decoder"]
	def __init__(
		self,
		n_cvs: int,
		encoder_layers: list,
		decoder_layers: Optional[list] = None,
		options: Optional[dict] = None,
		**kwargs,
	):
		super().__init__(in_features=encoder_layers[0], out_features=n_cvs, **kwargs)

		# =======   LOSS  =======
		# ELBO loss function when latent space and reconstruction distributions are Gaussians.
		self.loss_fn = ELBOGaussiansLossBeta()

		# ======= OPTIONS =======
		# parse and sanitize
		options = self.parse_options(options)

		# if decoder is not given reverse the encoder
		if decoder_layers is None:
			decoder_layers = encoder_layers[::-1]

		# ======= BLOCKS =======

		# initialize norm_in
		o = "norm_in"
		if (options[o] is not False) and (options[o] is not None):
			self.norm_in = Normalization(self.in_features, **options[o])

		# initialize encoder
		o = "encoder"
		if "last_layer_activation" not in options[o]:
			options[o]["last_layer_activation"] = True

		self.encoder = FeedForward(encoder_layers, **options[o])
		self.mean_nn = torch.nn.Linear(
			in_features=encoder_layers[-1], out_features=n_cvs
		)
		self.log_var_nn = torch.nn.Linear(
			in_features=encoder_layers[-1], out_features=n_cvs
		)

		# initialize encoder
		o = "decoder"
		self.decoder = FeedForward([n_cvs] + decoder_layers, **options[o])
		
	def training_step(self, train_batch, batch_idx):
		x = train_batch["data"]
		loss_kwargs = {}
		if "weights" in train_batch:
			loss_kwargs["weights"] = train_batch["weights"]

		mean, log_variance, x_hat = self.encode_decode(x)

		if "target" in train_batch:
			x_ref = train_batch["target"]
		else:
			x_ref = x

		recon, regular = self.loss_fn(x_ref, x_hat, mean, log_variance, **loss_kwargs)
		loss = recon + regular
		name = "train" if self.training else "valid"
		self.log(f"{name}_loss", loss, prog_bar=True, on_epoch=True)
		self.log(f"{name}_recon", recon, prog_bar=True, on_epoch=True)
		self.log(f"{name}_regul", regular, prog_bar=True, on_epoch=True)

		return loss


	def forward_cv(self, x: torch.Tensor) -> torch.Tensor:
		if self.norm_in is not None:
			x = self.norm_in(x)
		x = self.encoder(x)

		# Take only the means and ignore the log variances.
		return self.mean_nn(x)

	def encode_decode(
		self, x: torch.Tensor
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		# Normalize inputs.
		if self.norm_in is not None:
			x = self.norm_in(x)

		# Encode input into a Gaussian distribution.
		x = self.encoder(x)
		mean, log_variance = self.mean_nn(x), self.log_var_nn(x)

		# Sample from the Gaussian distribution in latent space.
		std = torch.exp(log_variance / 2)
		z = torch.distributions.Normal(mean, std).rsample()

		# Decode sample.
		x_hat = self.decoder(z)
		if self.norm_in is not None:
			x_hat = self.norm_in.inverse(x_hat)

		return mean, log_variance, x_hat

	def get_decoder(self, return_normalization=False):
		"""Return a torch model with the decoder and optionally the normalization inverse"""
		if return_normalization:
			if self.norm_in is not None:
				inv_norm = Inverse(module=self.norm_in)
				decoder_model = torch.nn.Sequential(*[self.decoder, inv_norm])
			else:
				raise ValueError(
					"return_normalization is set to True but self.norm_in is None"
				)
		else:
			decoder_model = self.decoder
		return decoder_model

