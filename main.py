import os
import wandb
import hydra
import torch
import logging
import datetime

from src import *
from accelerate import Accelerator
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
    
    # Load logger, model, data]
    logger = logging.getLogger("CMD")
    model_wrapper, optimizer, scheduler = load_model_wrapper(cfg, device)
    accelerator = Accelerator()
    logger.info(f"Model: {cfg.model.name}, param num: TBA")
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
        train_loader = load_data(cfg)
        loss_func = load_loss(cfg)
        temperature = train_loader.dataset.temperature
        data_num = len(train_loader.dataset)
        batch_size = cfg.training.batch_size
        logger.info(f"MD Dataset size: {data_num}")
        
        # Accelerate        
        model_wrapper, optimizer, train_loader = accelerator.prepare(model_wrapper, optimizer, train_loader)

        
        # Train model
        logger.info("Training...")
        pbar = trange(
            cfg.training.epoch,
            desc="Training"
        )
        for epoch in pbar:
            total_loss = 0
            total_var = 0
            
            for i, data in tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch}",
                leave=False
            ):
                # Load data
                # data = [d.to(device) for d in data]
                current_state, next_state, goal_state, step = data
                
                current_state *= 1000.0
                next_state *= 1000.0
                goal_state *= 1000.0
                optimizer.zero_grad()
                
                # Predict next state
                state_offset, var = model_wrapper(current_state, goal_state, step, temperature)
                
                # Compute loss
                loss = loss_func(next_state, current_state + state_offset)
                if cfg.training.loss_scale == "step":
                    loss = loss.reshape(step.shape[0], -1) / step
                    loss = loss.mean()
                total_var += var.sum()
                if cfg.training.loss_variance:
                    loss -= var.mean()
                # loss.backward()
                accelerator.backward(loss)
                if cfg.training.max_grad_norm is not None:
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model_wrapper.parameters(), cfg.training.max_grad_norm)
                    optimizer.step()
                
                total_loss += loss.item()
                # optimizer.step()
            
            # results 
            scheduler.step()
            result = {
                "lr": optimizer.param_groups[0]["lr"],
                "loss/total": total_loss / len(train_loader),
                "train/var": total_var  / len(train_loader),
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