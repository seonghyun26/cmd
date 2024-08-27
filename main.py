import os
import wandb
import hydra
import torch
import logging
import datetime
import argparse

from src import *
from tqdm.auto import tqdm, trange
from omegaconf import DictConfig, OmegaConf

import numpy as np
import matplotlib.pyplot as plt

@hydra.main(
    version_base=None,
    config_path="config",
    config_name="debug"
)
def main(cfg):
    # Load configs
    print(f"Loading configs...")
    print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")
    print(OmegaConf.to_yaml(cfg))
    print("Done!!!\n")
    device = torch.device(cfg.logging.device if "device" in cfg.logging else "cpu")
    
    # Load logger, model, data
    if cfg.logging.wandb:
        wandb.init(
            project=cfg.logging.project,
            entity=cfg.logging.entity,
            tags=cfg.logging.tags,
            config=OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            )
        )
    logger = logging.getLogger("CMD")
    model = load_model(cfg, device)
    loss_func = load_loss(cfg)
    train_loader = load_data(cfg)
    data_num = len(train_loader.dataset)
    logger.info(f"Model: {cfg.model.name}")
    logger.info(f"MD Dataset size: {data_num}")
    
    # Train model
    pbar = trange(cfg.training.epoch)
    for epoch in pbar:
        loss = 0
        for i, data in enumerate(train_loader):
            # Load data and predict next state
            current_state, next_state = data
            current_state, next_state = current_state.to(device), next_state.to(device)
            
            step = 1
            temperature = 300
            predicition_next_state = model(
                current_state,
                goal_state,
                step,
                temperature
            )
            
            # Compute loss
            loss += loss_func(predicition_next_state, next_state)
        loss /= data_num
        loss.backward()
        
        wandb.log({"loss": loss.item()})
        if epoch % 100 == 0:
            # logger.info(f"Epoch: {epoch}, Loss: {batch_loss.item():4f}")
            pbar.set_description(f"Loss: {loss.item():4f}")
            pbar.refresh()
    
    # Finish training, save model, evaluate
    if cfg.logging.wandb:
        wandb.finish()
        
    
if __name__ == "__main__":
    main()