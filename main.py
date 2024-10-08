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
    with open_dict(cfg):
        cfg.logging.dir = hydra.core.hydra_config.HydraConfig.get().run.dir
    device = torch.device(cfg.logging.device if "device" in cfg.logging else "cpu")

    # Load logger, model, data
    logger = logging.getLogger("CMD")
    logger.info(">> Configs")
    logger.info(OmegaConf.to_yaml(cfg))
    model_wrapper, optimizer, scheduler = load_model_wrapper(cfg, device)
    if cfg.logging.wandb:
        wandb.init(
            project=cfg.logging.project,
            entity=cfg.logging.entity,
            tags=cfg.logging.tags,
            config=OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            )
        )
        wandb.config.update({"Model Parameters": sum([p.numel() for p in model_wrapper.parameters() if p.requires_grad])})
    
    # Train or load model from checkpoint
    if cfg.training.train:
        # Load dataset
        logger.info(">> Loading dataset...")
        train_loader, test_loader = load_data(cfg)
        criteria, loss_names = load_loss(cfg)
        scale = cfg.training.scale
        batch_size = cfg.training.batch_size
        logger.info(f">> Dataset size: {len(train_loader.dataset)}")
        
        # Train model
        logger.info(">> Training...")
        pbar = trange(
            cfg.training.epoch,
            desc="Training (Loss: ----)"
        )
        for epoch in pbar:
            model_wrapper.train()
            loss_list = { f"loss/{name}": 0 for name in loss_names }
            loss_list.update({"loss/total": 0})
            
            for i, data in tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch}",
                leave=False
            ):
                # Load data
                if cfg.data.molecule == "double-well":
                    current_state, next_state, goal_state, step, temperature = (d.to(device) for d in data)
                    current_state *= scale
                    next_state *= scale
                    goal_state *= scale
                elif cfg.data.molecule == "alanine":
                    current_state, next_state, goal_state, step = (d.to(device) for d in data)
                    current_state *= scale
                    next_state *= scale
                    goal_state *= scale
                    temperature = torch.tensor([300] * current_state.shape[0]).unsqueeze(1).to(device)
                else:
                    raise ValueError(f"Molecule {cfg.molecule} not found")
                optimizer.zero_grad()
                
                # Predict next state
                if cfg.training.state_representation == "difference":
                    goal_representation = goal_state - current_state
                elif cfg.training.state_representation == "original":
                    goal_representation = goal_state
                else:
                    raise ValueError(f"State representation {cfg.training.state_representation} not found")
                if cfg.training.repeat:
                    step = step.repeat(1, current_state.shape[1])
                    temperature = temperature.repeat(1, current_state.shape[1])
                state_offset, mu, log_var = model_wrapper(
                    current_state=current_state,
                    goal_state=goal_representation,
                    step=step,
                    temperature=temperature,
                )
                
                # Compute loss
                loss_list_batch = criteria(next_state, current_state + state_offset, mu, log_var, step)
                for name in loss_list_batch.keys():
                    loss_list[f"loss/{name}"] += loss_list_batch[name]
                
                loss = sum(loss_list_batch.values())
                loss_list["loss/total"] += loss
                loss.backward()
                optimizer.step()
            
            # Update results
            if scheduler is not None:
                scheduler.step()
            loss_list = {k: v / (len(train_loader) * scale) for k, v in loss_list.items()}
            loss_list.update({"lr": optimizer.param_groups[0]["lr"]})
            pbar.set_description(f"Training (loss: {loss_list['loss/total']:8f})")
            pbar.refresh() 

            # Wandb loggging
            if cfg.logging.wandb:
                wandb.log(
                    loss_list,
                    step=epoch
                )
                
            # Save checkpoint 
            if epoch * cfg.logging.ckpt_freq != 0 and epoch % cfg.logging.ckpt_freq == 0:
                output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
                model_wrapper.save_model(output_dir, epoch)
                logger.info(f"Epoch {epoch}, model weights saved at: {output_dir}")
                model_wrapper.eval()
                logger.info(">> Evaluating...")
                trajectory_list = generate(cfg, model_wrapper, epoch, device, logger)
                evaluate(cfg=cfg, trajectory_list=trajectory_list, logger=logger, epoch=epoch)
                model_wrapper.train()
    
        # Save model
        logger.info("Training complete")
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
    if cfg.job.evaluate:
        logger.info("Evaluating...")
        trajectory_list = generate(cfg, model_wrapper, epoch, device, logger)
        evaluate(
            cfg=cfg, 
            trajectory_list=trajectory_list,
            logger=logger,
            epoch=cfg.training.epoch
        )
        logger.info("Evaluation complete!!!")
        
    # Finish and exit
    if cfg.logging.wandb:
        wandb.finish()
    
if __name__ == "__main__":
    main()