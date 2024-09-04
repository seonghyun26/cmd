import torch
from torch.utils.data import Dataset

class MD_Dataset(Dataset):
    def __init__(self, loaded_traj, config):
        self.molecule = config['molecule']
        self.state = config['state']
        self.temperature = config['temperature']
        self.time = config['time']
        self.force_field = config['force_field']
        self.solvent = config['solvent']
        self.platform = config['platform']
        self.precision = config['precision']
        self.device = "cuda"
        
        data_x_list = []
        data_y_list = []
        traj = loaded_traj.xyz.squeeze()
        for t in tqdm(
            range((self.time -1) // 10),
            desc="Loading data"
        ):
            current_state = torch.tensor(loaded_traj[t].xyz.squeeze()).to(self.device)
            next_state = torch.tensor(loaded_traj[t+1].xyz.squeeze()).to(self.device)
            data_x_list.append(current_state)
            data_y_list.append(next_state)
        self.x = torch.stack(data_x_list).to(self.device)
        self.y = torch.stack(data_y_list).to(self.device)
        
        # self.sanity_check(loaded_traj)
    
    def sanity_check(self, loaded_traj):
        # print("Running sanity check...")
        # print(f">> x size: {self.x.shape}")
        # print(f">> y size: {self.y.shape}")
        assert torch.equal(x.shape, y.shape), f"Shape of x and y not equal"
        
        for t in tqdm(
            range(self.time -1),
            desc="Sanity check"
        ):
            x = self.x[t]
            y = self.y[t]
            x_frame = torch.tensor(loaded_traj[t].xyz.squeeze()).to(self.device)
            y_frame = torch.tensor(loaded_traj[t+1].xyz.squeeze()).to(self.device)
            
            assert torch.equal(x, x_frame), f"Frame {t}, x not equal"
            assert torch.equal(y, y_frame), f"Frame {t+1}, y not equal"        
            

    def __getitem__(self, index):
	    return self.x[index], self.y[index]
 
    def __len__(self):
	    return self.x.shape[0]
