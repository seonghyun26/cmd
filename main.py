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
    
    
    if cfg.training.train:
        # Load dataset
        train_loader, test_loader, total_data_mean, total_data_std = load_data(cfg)
        model_wrapper.set_normalization(total_data_mean, total_data_std)
        criteria, loss_type = load_loss(cfg)
        loss_dict = { f"loss/{name}": 0 for name in loss_type }
        loss_dict["loss/total"] = 0
        logger.info(f">> Loading dataset...")
        logger.info(f">> Dataset size: {len(train_loader.dataset)}")
        
        # Train model
        logger.info(">> Training...")
        pbar = trange(
            cfg.training.epoch,
            desc=f"Training (loss: {loss_dict['loss/total']})"
        )
        for epoch in pbar:
            model_wrapper.train()
            for key in loss_dict.keys():
                loss_dict[key] = 0
            
            for i, data in tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch}",
                leave=False
            ):
                # Load data
                if cfg.data.molecule == "alanine":
                    current_state, positive_sample, negative_sample, temperature = (d.to(device) for d in data)
                else:
                    raise ValueError(f"Molecule {cfg.molecule} not found")

                # Self-supervised learning
                result_dict = model_wrapper(
                    current_state=current_state,
                    positive_sample=positive_sample,
                    negative_sample=negative_sample,
                    temperature=temperature
                )
                
                # Compute loss
                loss_dict_batch = criteria(result_dict)
                for name in loss_dict_batch.keys():
                    loss_dict[f"loss/{name}"] += loss_dict_batch[name]
                loss = loss_dict_batch["total"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Update results
            if scheduler is not None:
                scheduler.step()
            loss_dict = {k: v / (len(train_loader) ) for k, v in loss_dict.items()}
            loss_dict.update({"lr": optimizer.param_groups[0]["lr"]})
            pbar.set_description(f"Training (loss: {loss_dict['loss/total']})")
            pbar.refresh() 

            # Wandb loggging
            if cfg.logging.wandb:
                wandb.log(
                    loss_dict,
                    step=epoch
                )
                
            # Save checkpoint
            if epoch * cfg.logging.ckpt_freq != 0 and epoch % cfg.logging.ckpt_freq == 0:
                output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
                model_wrapper.save_model(output_dir, epoch)
                logger.info(f"Epoch {epoch}, model weights saved at: {output_dir}")
                model_wrapper.eval()
                logger.info(">> Evaluating...")
                trajectory_list = generate(
                    cfg=cfg,
                    model_wrapper=model_wrapper,
                    epoch=epoch,
                    device=device,
                    logger=logger
                )
                evaluate(
                    cfg=cfg, 
                    model_wrapper=model_wrapper,
                    trajectory_list=trajectory_list,
                    logger=logger,
                    epoch=epoch,
                    device=device
                )
                model_wrapper.train()
    
        # Save model
        logger.info("Training complete")
        if cfg.logging.checkpoint:
            output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            model_wrapper.save_model(output_dir, "final")
            logger.info(f"Final model weights saved at: {output_dir}")
    else:
        # Load trainined model from checkpoint
        epoch = 0
        if cfg.job.generate:
            model_wrapper.load_from_checkpoint(f"./model/{cfg.job.molecule}/{cfg.model.name}/{cfg.training.ckpt_file}.pt")
            model_wrapper.eval()
        else:
            logger.info(f"Using generated trajectories from {cfg.job.traj_dir}")

    # Test model on downstream task (generation)
    if cfg.job.evaluate:
        logger.info("Evaluating...")
        if cfg.job.generate:
            trajectory_list = generate(
                cfg=cfg,
                model_wrapper=model_wrapper,
                epoch=epoch,
                device=device,
                logger=logger
            )
        else:
            trajectory_dir = cfg.job.traj_dir
            trajectory_list = []
            for i in range(cfg.job.sample_num):
                trajectory = np.load(f"{trajectory_dir}/{i}.npy")
                trajectory_list.append(trajectory)
            trajectory_list = np.array(trajectory_list)
        evaluate(
            cfg=cfg,
            model_wrapper=model_wrapper,
            trajectory_list=trajectory_list,
            logger=logger,
            epoch=epoch,
            device=device
        )
        logger.info("Evaluation complete!!!")
        
    # Finish and exit
    if cfg.logging.wandb:
        wandb.finish()
    
if __name__ == "__main__":
    main()