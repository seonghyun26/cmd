import os
import wandb
import hydra
import torch
import logging
import datetime

from src import *
from tqdm.auto import tqdm, trange
from omegaconf import DictConfig, OmegaConf, open_dict

import numpy as np
import matplotlib.pyplot as plt

@hydra.main(
    version_base=None,
    config_path="config",
    config_name="basic"
)
def main(cfg):
    # Load configs
    print(f">> Loading configs...")
    with open_dict(cfg):
        cfg.logging.dir = hydra.core.hydra_config.HydraConfig.get().run.dir
    print(OmegaConf.to_yaml(cfg))
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
    model_wrapper, optimizer, scheduler = load_model_wrapper(cfg, device)
    loss_func = load_loss(cfg)
    train_loader = load_data(cfg)
    logger.info(f"Model: {cfg.model.name}")
    
    if cfg.training.train:
        # Load dataset
        temperature = train_loader.dataset.temperature
        data_num = len(train_loader.dataset)
        batch_size = cfg.training.batch_size
        loss_lambda = 1
        logger.info(f"MD Dataset size: {data_num}")
        
        # Train model
        logger.info("Training...")
        pbar = trange(cfg.training.epoch, desc="Training")
        for epoch in pbar:
            total_loss = 0
            
            for i, data in tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch}",
                leave=False
            ):
                # Load data
                data = [d.to(device) for d in data]
                current_state, next_state, goal_state, step = data
                optimizer.zero_grad()
                
                # Predict next state
                encoded, decoded, mu, logvar = model_wrapper(next_state, current_state, goal_state, step, temperature)
                
                # Compute loss
                recon_loss, kl_div = loss_func(next_state, decoded, mu, logvar)
                loss = recon_loss + kl_div
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            
            # Save result and logg
            scheduler.step()
            total_loss /= data_num / loss_lambda
            result = {
                "lr": optimizer.param_groups[0]["lr"],
                "loss/recon": recon_loss,
                "loss/kl": kl_div,
                "loss/total": total_loss
            }
            
            if cfg.logging.wandb:
                wandb.log(
                    result,
                    step=epoch
                )
            if epoch % cfg.logging.update_freq == 0:
                pbar.set_description(f"Training (loss: {total_loss:4f})")
                pbar.refresh()  
            if cfg.logging.ckpt_freq != 0 and epoch % cfg.logging.ckpt_freq == 0:
                output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
                model_wrapper.save_model(output_dir)
                logger.info(f"Model weights saved at: {output_dir}")
    else:
        # Load trainined model from checkpoint
        ckpt_path = f"model/{cfg.data.molecule}/{cfg.training.ckpt_name}"
        logger.info(f"Loading checkpoint from {ckpt_path}")
        mode_wrapper.load_from_checkpoint(ckpt_path)
    
    # Save model
    logger.info("Training complete!!!")
    if cfg.logging.checkpoint:
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        model_wrapper.save_model(output_dir)
        logger.info(f"Model weights saved at: {output_dir}")


    # Test model on downstream task (generation)
    logger.info("Evaluating...")
    trajectory_list = generate(cfg, model_wrapper, device, logger)
    evaluate(cfg=cfg, trajectory_list=trajectory_list, logger=logger)
    logger.info("Evaluation complete!!!")
        
        
    # Finish and exit
    if cfg.logging.wandb:
        wandb.finish()
    
if __name__ == "__main__":
    main()