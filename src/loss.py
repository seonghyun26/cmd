import torch
import torch.nn as nn


def mse_loss(y_true, y_pred, *args):
    mse_loss = nn.MSELoss(reduction=cfg.training.loss.reduction)(y_pred, y_true)

    return mse_loss, 0

def mse_reg_loss(y_true, y_pred, mu, log_var, *args):
    mse_loss = nn.MSELoss(reduction=cfg.training.loss.reduction)(y_pred, y_true)
    reg_loss = -0.5 * torch.sum(1 + log_var - log_var.exp())

    return mse_loss, reg_loss.mean()

def mse_reg2_loss(y_true, y_pred, mu, log_var, *args):
    mse_loss = nn.MSELoss(reduction=cfg.training.loss.reduction)(y_pred, y_true)
    reg_loss = torch.square(log_var.exp())

    return mse_loss, reg_loss.mean()

def mse_reg3_loss(y_true, y_pred, mu, log_var, *args):
    mse_loss = nn.MSELoss(reduction=cfg.training.loss.reduction)(y_pred, y_true)
    reg_loss = torch.square(log_var)

    return mse_loss, reg_loss.mean()

def mse_reg4_loss(y_true, y_pred, mu, log_var, step, *args):
    mse_loss = nn.MSELoss(reduction="none")(y_pred, y_true).mean(dim=(1,2))
    reg_loss = torch.square(log_var).mean(dim=(1))
    step_div = torch.sqrt(step).squeeze()
    mse_loss /= step_div
    reg_loss /= step_div
    
    if cfg.training.loss.reduction == "mean":
        mse_loss = mse_loss.mean()
        reg_loss = reg_loss.mean()
    elif cfg.training.loss.reduction == "sum":
        mse_loss = mse_loss.sum()
        reg_loss = reg_loss.sum()
    else:
        raise ValueError(f"Reduction {cfg.training.loss.reduction} not found")

    return mse_loss, reg_loss