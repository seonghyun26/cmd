import torch

from torch.utils.data import Dataset

class CL_dataset(Dataset):
    def __init__(
        self,
        data_list,
        data_augmented_list,
        data_augmented_hard_list,
        temperature_list,
    ):
        super(CL_dataset, self).__init__()
        self.device = "cpu"
        
        self.x = data_list.to(self.device)
        self.x_augmented = data_augmented_list.to(self.device)
        self.x_augmented_hard = data_augmented_hard_list.to(self.device)
        self.temperature = temperature_list.to(self.device)
        
    def __getitem__(self, index):
	    return self.x[index], self.x_augmented[index], self.x_augmented_hard[index], self.temperature[index]
 
    def __len__(self):
	    return self.x.shape[0]
 