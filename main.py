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
    data_process_func = load_process_data(cfg)
    pred_process_func = load_process_prediction(cfg)
    
    data_num = len(train_loader.dataset)
    batch_size = cfg.training.batch_size
    loss_lambda = 1000
    
    logger.info(f"Model: {cfg.model.name}")
    logger.info(f"MD Dataset size: {data_num}")
    
    
    # Train model
    logger.info("Training...")
    pbar = trange(cfg.training.epoch)
    for epoch in pbar:
        total_loss = 0
        step = 1
        temperature = 300
        
        for i, data in enumerate(train_loader):
            # Load data
            current_state, next_state = data
            current_state, next_state = current_state.to(device), next_state.to(device)
            
            # Predict next state
            goal_state = current_state
            x_processed = data_process_func(current_state, goal_state, step, temperature)
            predicted_state = model(x_processed)
            predicted_state = pred_process_func(predicted_state, current_state.shape)
            
            # Compute loss
            loss = loss_func(next_state, predicted_state) * loss_lambda
            total_loss += loss.item()
            loss.backward()
        
        # Logging
        total_loss /= data_num / loss_lambda
        if cfg.logging.wandb:
            wandb.log({"loss": total_loss})
        if epoch % 100 == 0:
            pbar.set_description(f"Loss: {total_loss:4f}")
            pbar.refresh()
    
    # Finish training, save model
    logger.info("Traiing complete!!!")
    if cfg.logging.wandb:
        wandb.finish()
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        model_save_path = os.path.join(output_dir, "model_weights.pt")
        torch.save(model.state_dict(), model_save_path)
        logger.info(f"Model weights saved at: {model_save_path}")

    # Test model (generation)
    logger.info("Evaluating...")
    evaluate(cfg, model, device, logger)
    logger.info("Evaluation complete!!!")
        
    
if __name__ == "__main__":
    main()