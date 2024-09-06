import torch

from torch.utils.data import Dataset


class MD_Dataset(Dataset):
    def __init__(
        self,
        loaded_traj,
        config,
        args,
        sanity_check=False
    ):
        super(MD_Dataset, self).__init__()
        
        self.molecule = config['molecule']
        self.state = config['state']
        self.temperature = config['temperature']
        self.time = config['time']
        self.force_field = config['force_field']
        self.solvent = config['solvent']
        self.platform = config['platform']
        self.precision = config['precision']
        self.device = "cpu"
        
        data_x_list = []
        data_y_list = []
        data_interval_list = []
        data_goal_list = []
        
        if args.index == "random":
            random_indices = random.sample(range(0, self.time - 1), self.time // args.percent)
            for t in tqdm(
                random_indices,
                desc="Loading data by random idx"
            ):
                current_state = torch.tensor(loaded_traj[t].xyz.squeeze()).to(self.device)
                next_state = torch.tensor(loaded_traj[t+1].xyz.squeeze()).to(self.device)
                random_interval = random.sample(range(1, self.time - t), 1)[0]
                goal_state = torch.tensor(loaded_traj[t+random_interval].xyz.squeeze()).to(self.device)
                
                data_x_list.append(current_state)
                data_y_list.append(next_state)
                data_goal_list.append(goal_state)
                data_interval_list.append(torch.tensor(random_interval).to(self.device).unsqueeze(0))
        else:
            for t in tqdm(
                range((self.time -1) // args.percent),
                desc=f"Loading {args.precent} precent of dataset from initial frame"
            ):
                current_state = torch.tensor(loaded_traj[t].xyz.squeeze()).to(self.device)
                next_state = torch.tensor(loaded_traj[t+1].xyz.squeeze()).to(self.device)
                data_x_list.append(current_state)
                data_y_list.append(next_state)
                data_interval_list.append(1)
                
        self.x = torch.stack(data_x_list).to(self.device)
        self.y = torch.stack(data_y_list).to(self.device)
        self.goal = torch.stack(data_goal_list).to(self.device)
        self.delta_time = torch.stack(data_interval_list).to(self.device)
        
        # if sanity_check:
        #     self.sanity_check(loaded_traj)
        
    def sanity_check(self, loaded_traj):
        assert torch.equal(self.x.shape, self.y.shape), f"Shape of x and y not equal"
        
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
	    return self.x[index], self.y[index], self.goal[index], self.delta_time[index]
 
    def __len__(self):
	    return self.x.shape[0]
