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
    logger = logging.getLogger("CMD")
    model_wrapper, optimizer, scheduler = load_model_wrapper(cfg, device)
    logger.info(f"Model: {cfg.model.name}")
    if cfg.logging.wandb:
        wandb.init(
            project=cfg.logging.project,
            entity=cfg.logging.entity,
            tags=cfg.logging.tags,
            config=OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            )
        )
    
    # Train or load model from checkpoint
    if cfg.training.train:
        # Load dataset
        train_loader, test_loader = load_data(cfg)
        criteria = load_loss(cfg)
        if cfg.training.test:
            temperature = train_loader.dataset.dataset.temperature
        else:
            temperature = train_loader.dataset.temperature
        data_num = len(train_loader.dataset)
        batch_size = cfg.training.batch_size
        logger.info(f"MD Dataset size: {data_num}")
        
        # Train model
        logger.info("Training...")
        pbar = trange(
            cfg.training.epoch,
            desc="Training"
        )
        for epoch in pbar:
            model_wrapper.train()
            total_loss = 0
            total_mse = 0
            total_reg = 0
            
            for i, data in tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch}",
                leave=False
            ):
                # Load data
                data = [d.to(device) for d in data]
                current_state, next_state, goal_state, step = data
                
                # current_state *= 1000.0
                # next_state *= 1000.0
                # goal_state *= 1000.0
                optimizer.zero_grad()
                
                # Predict next state
                state_offset, mu, log_var = model_wrapper(current_state, goal_state, step, temperature)
                
                # Compute loss
                mse_loss, reg_loss = criteria(next_state, current_state + state_offset, mu, log_var)
                loss = mse_loss + reg_loss
                total_mse += mse_loss.item()
                total_reg += reg_loss.item()
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            
            # results 
            scheduler.step()
            if cfg.training.test:
                model_wrapper.eval()
                test_loss = 0
                for i, test_data in enumerate(test_loader):
                    data = [d.to(device) for d in data]
                    current_state, next_state, goal_state, step = data
                    # current_state *= 1000.0
                    # next_state *= 1000.0
                    # goal_state *= 1000.0
                    state_offset, var = model_wrapper(current_state, goal_state, step, temperature)
                    loss = criteria(next_state, current_state + state_offset)
                    test_loss += loss.mean()
            result = {
                "lr": optimizer.param_groups[0]["lr"],
                "loss/total": total_loss / len(train_loader),
                "loss/mse": total_mse / len(train_loader),
                "loss/reg": total_reg / len(train_loader),
                # "test/loss": test_loss / len(test_loader)
            }
            pbar.set_description(f"Training (loss: {total_loss:4f})")
            pbar.refresh()  
            
            # Wandb loggging
            if cfg.logging.wandb:
                wandb.log(
                    result,
                    step=epoch
                )
                
            # Jobs to do at epoch frequency
            if epoch * cfg.logging.ckpt_freq != 0 and epoch % cfg.logging.ckpt_freq == 0:
                output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
                model_wrapper.save_model(output_dir, epoch)
                logger.info(f"Epcoh {epoch}, model weights saved at: {output_dir}")
                model_wrapper.eval()
                trajectory_list = generate(cfg, model_wrapper, epoch, device, logger)
                evaluate(cfg=cfg, trajectory_list=trajectory_list, logger=logger, epoch=epoch)
                model_wrapper.train()
        
        # Save model
        logger.info("Training complete!!!")
        if cfg.logging.checkpoint:
            output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            model_wrapper.save_model(output_dir, "final")
            logger.info(f"Final model weights saved at: {output_dir}")
        
    else:
        # Load trainined model from checkpoint
        ckpt_path = f"model/{cfg.data.molecule}/{cfg.training.ckpt_name}"
        logger.info(f"Loading checkpoint from {ckpt_path}")
        model_wrapper.load_from_checkpoint(ckpt_path)
        epoch = cfg.training.epoch

    # Test model on downstream task (generation)
    logger.info("Evaluating...")
    trajectory_list = generate(cfg, model_wrapper, epoch, device, logger)
    evaluate(cfg=cfg, trajectory_list=trajectory_list, logger=logger, epoch=cfg.training.epoch)
    logger.info("Evaluation complete!!!")
        
        
    # Finish and exit
    if cfg.logging.wandb:
        wandb.finish()
    
if __name__ == "__main__":
    main()