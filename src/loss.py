import torch
import torch.nn as nn
import torch.nn.functional as F

import mdtraj as md

from abc import ABC, abstractmethod

from .utils.distance import coordinate2distance


def load_similarity(similarity_type):
    if similarity_type == "cosine":
        similarity = nn.CosineSimilarity(dim=1)
    
    else:
        raise ValueError(f"Similarity {similarity_type} not found")

    return similarity


class BaseLoss(ABC):
    def __init__(self, cfg, normalization = None):
        self.cfg = cfg
        self.reduction = cfg.training.loss.reduction
        self.normalization = normalization

    @abstractmethod
    def __call__(self, result_dict):
        pass

    @property
    def loss_types(self):
        pass
    

class TripletLoss(BaseLoss):
    def __init__(self, cfg, normalization = None):
        super().__init__(cfg, normalization)
        self.margin = cfg.training.loss.margin
        self.temperature = cfg.training.loss.temperature
        self.reduction = torch.mean if self.cfg.training.loss.reduction == "mean" else torch.sum
        self.normalization = normalization
    def __call__(
        self,
        result_dict
    ):
        anchor = result_dict["current_state_rep"] / self.temperature    
        positive = result_dict["positive_sample_rep"] / self.temperature
        negative = result_dict["negative_sample_rep"] / self.temperature
        
        distance_positive = F.pairwise_distance(anchor, positive, p=2) / self.temperature
        distance_negative = F.pairwise_distance(anchor, negative, p=2) / self.temperature
        triplet_loss = F.relu(distance_positive - distance_negative + self.margin)
        
        return {
            "positive": self.reduction(distance_positive),
            "negative": self.reduction(distance_negative),
            "total": self.reduction(triplet_loss),
        }

    @property
    def loss_types(self):
        return ["positive", "negative", "total"]
    
    
class NTXent(BaseLoss):
    def __init__(self, cfg, normalization = None):
        super().__init__(cfg, normalization)
        self.margin = cfg.training.loss.margin
        self.temperature = cfg.training.loss.temperature
        self.similarity = load_similarity(cfg.training.loss.similarity)
        self.reduction = torch.mean if self.cfg.training.loss.reduction == "mean" else torch.sum
        
    def __call__(
        self,
        result_dict
    ):
        anchor = result_dict["current_state_rep"] / self.temperature    
        positive = result_dict["positive_sample_rep"] / self.temperature
        negative = result_dict["negative_sample_rep"] / self.temperature
        
        batch_size = anchor.shape[0]
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        pos_sim = torch.exp(self.similarity(anchor, positive).sum(dim=1) / self.temperature)
        
        negative = F.normalize(negative, dim=1)
        all_samples = torch.cat([positive, negative], dim=0)
        all_sim = torch.exp(self.similarity(anchor, all_samples).sum(dim=1) / self.temperature)
        
        mask = torch.eye(batch_size, device=anchor.device)
        mask = torch.cat([mask, torch.zeros_like(mask)], dim=1)
        all_sim = all_sim * (1 - mask)
        
        denominator = all_sim.sum(dim=1)
        losses = -torch.log(pos_sim / denominator)
        
        return {
            "numerator": self.reduction(pos_sim),
            "denominator": self.reduction(denominator),
            "total": self.reduction(losses),
        }
        
    @property
    def loss_types(self):
        return ["numerator", "denominator", "total"]
    

class TripletLossTest(BaseLoss):
    def __init__(self, cfg, normalization = None):
        super().__init__(cfg, normalization)
        self.margin = cfg.training.loss.margin
        self.temperature = cfg.training.loss.temperature
        self.reduction = torch.mean if self.cfg.training.loss.reduction == "mean" else torch.sum
        
    def __call__(
        self,
        result_dict
    ):
        anchor = result_dict["current_state_rep"] / self.temperature    
        positive = result_dict["positive_sample_rep"] / self.temperature
        negative = result_dict["negative_sample_rep"] / self.temperature
        
        distance_positive = F.pairwise_distance(anchor, positive, p=2) / self.temperature
        distance_negative = F.pairwise_distance(anchor, negative, p=2) / self.temperature
        triplet_loss = F.relu(distance_positive - distance_negative + self.margin)
        
        return {
            "positive": self.reduction(distance_positive),
            "negative": self.reduction(distance_negative),
            "total": self.reduction(triplet_loss),
        }

    @property
    def loss_types(self):
        return ["positive", "negative", "total"]

class TripletLossNegative(BaseLoss):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.margin = cfg.training.loss.margin
        self.temperature = cfg.training.loss.temperature
        self.reduction = torch.mean if self.cfg.training.loss.reduction == "mean" else torch.sum

    def __call__(
        self,
        result_dict
    ):
        anchor = result_dict["current_state_rep"] / self.temperature    
        positive = result_dict["positive_sample_rep"] / self.temperature
        negative = result_dict["negative_sample_rep"] / self.temperature
        
        distance_positive = F.pairwise_distance(anchor, positive, p=2) / self.temperature
        distance_negative = F.pairwise_distance(anchor, negative, p=2) / self.temperature
        triplet_loss = F.relu(-distance_negative + self.margin)
        
        return {
            "positive": self.reduction(distance_positive),
            "negative": self.reduction(distance_negative),
            "total": self.reduction(triplet_loss),
        }

    @property
    def loss_types(self):
        return ["positive", "negative", "total"]


class TripletTorchLoss(BaseLoss):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.temperature = cfg.training.loss.temperature
        self.loss = nn.TripletMarginLoss(
            margin=self.cfg.training.loss.margin,
            p=self.cfg.training.loss.p,
            eps=self.cfg.training.loss.eps,
            reduction=self.cfg.training.loss.reduction
        )

    def __call__(self, result_dict):
        anchor = result_dict["current_state_rep"] / self.temperature    
        positive = result_dict["positive_sample_rep"] / self.temperature
        negative = result_dict["negative_sample_rep"] / self.temperature
        
        total = self.loss(anchor, positive, negative)
        
        return {
            "total": total
        }

    @property
    def loss_types(self):
        return ["positive", "negative", "total"]
    
    
class InfoNCELoss(BaseLoss):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.temperature = cfg.training.loss.temperature
        self.similarity = load_similarity(cfg.training.loss.similarity)

    def __call__(self, result_dict):
        anchor = result_dict["current_state_rep"] / self.temperature    
        positive = result_dict["positive_sample_rep"] / self.temperature
        negative = result_dict["negative_sample_rep"] / self.temperature
        batch_size = anchor.shape[0]
        
        logits_pos = anchor @ positive.T
        logits_neg = anchor @ negative.T
        logits = torch.cat((logits_pos, logits_neg), dim=1)
        labels = torch.arange(batch_size).to(logits.device)
        
        positive = F.cross_entropy(logits_pos / self.temperature, torch.arange(logits_pos.shape[0]).to(logits_pos.device), reduction=self.cfg.training.loss.reduction )
        negative = F.cross_entropy(logits_neg / self.temperature, torch.arange(logits_neg.shape[0]).to(logits_neg.device), reduction=self.cfg.training.loss.reduction )
        total = F.cross_entropy(logits / self.temperature, labels, reduction=self.cfg.training.loss.reduction )
        
        return {
            "positive": positive,
            "negative": negative,
            "total": total
        }

    @property
    def loss_types(self):
        return ["positive", "negative", "total"]
    
