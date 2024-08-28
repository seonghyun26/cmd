import torch

def evaluate(cfg, model, device, logger):
    task = cfg.evaluation.task
    
    if task == "simulation":
        simulation(cfg, model, device, logger)
    elif task == "tps":
        tps(cfg, model, device, logger)
    else:
        raise ValueError(f"Task {task} not found")
    
    return
    
def simulation(cfg, model, device, logger):
    return

def tps(cfg, model, device, logger):
    return