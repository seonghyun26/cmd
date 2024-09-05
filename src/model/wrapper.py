class ModelWrapper1(nn.Module):
    def __init__(self, cfg, device):
        super(ModelWrapper, self).__init__()
        
        self.atom_num = cfg.data.atom
        self.batch_size = cfg.training.batch_size
        self.latent_dim = cfg.model.encoder.output_dim
        
        self.encoder = self.load_model(cfg.model.encoder).to(device)
        self.decoder = self.load_model(cfg.model.decoder).to(device)
        self.mu = nn.Linear(self.latent_dim, self.latent_dim).to(device)
        self.logvar = nn.Linear(self.latent_dim, self.latent_dim).to(device)

    def __parameters__(self):
        return self.encoder.parameters(), self.decoder.parameters()
    
    def load_model(self, cfg_model):
        model_dict = {
            "MLP": MLP,
        }
        
        if cfg_model.name in model_dict.keys():
            model = model_dict[cfg_model.name](cfg=cfg_model)
        else:
            raise ValueError(f"Model {cfg_model.name} not found")
        
        if "inii" in cfg_model:
            model.apply(init(cfg_model.init))
        
        return model
    
    def save_model(self, path, epoch):
        torch.save(self.encoder.state_dict(), f"{path}/encoder-{epoch}.pt")
        torch.save(self.decoder.state_dict(), f"{path}/decoder-{epoch}.pt")
        
    def load_from_checkpoint(self, path):
        self.encoder.load_state_dict(torch.load(f"{path}/encoder.pt"))
        self.decoder.load_state_dict(torch.load(f"{path}/decoder.pt"))
    
    def forward(self, next_state, current_state, goal_state, step, temperature):
        # Encode
        x = self.process_data(next_state, current_state, goal_state, step, temperature)
        shape = x.shape
        encoded = self.encoder(x)
        
        # Reparameterize
        mu = self.mu(encoded)
        logvar = self.logvar(encoded)
        # Check if mu and logvar are not nan and inf
        # assert torch.isnan(mu).sum() == 0, f"mu has nan"
        # assert torch.isnan(logvar).sum() == 0, f"logvar has nan"
        # assert torch.isinf(mu).sum() == 0, f"mu has inf"
        # assert torch.isinf(logvar).sum() == 0, f"logvar has inf"
        logvar = torch.clamp(logvar, max=10)
        z = self.reparameterize(mu, logvar)
        
        # Decode
        decoded = self.decoder(self.process_latent(z, current_state, goal_state, step, temperature))
        decoded = self.process_prediction(decoded, shape)
        
        return encoded, decoded, mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        
        return mu + eps * std
    
    def generate(self, condition):
        # Condition: current_state, goal_state, step, temperature
        sample_num = condition.shape[0]
        gaussian_noise = torch.randn(sample_num, self.latent_dim).to(condition.device)
        generated_values = self.decoder(torch.cat((gaussian_noise, condition), dim=1))
        
        return generated_values
    
    def process_data(self, latent_var, current_state, goal_state, step, temperature):   
        batch_size = latent_var.shape[0]     
        temperature = torch.tensor(temperature).to(current_state.device).repeat(batch_size, 1)
        
        processed_state = torch.cat([
            latent_var.reshape(batch_size, -1), 
            current_state.reshape(batch_size, -1),
            goal_state.reshape(batch_size, -1),
            step.reshape(batch_size, -1),
            temperature
        ], dim=1)
        
        return processed_state
    
    def process_latent(self, latent, current_state, goal_state, step, temperature):
        batch_size = latent.shape[0]
        temperature = torch.tensor(temperature).to(latent.device).repeat(batch_size, 1)
        
        processed_latent = torch.cat((
            latent,
            current_state.reshape(batch_size, -1),
            goal_state.reshape(batch_size, -1),
            step.reshape(batch_size, -1),
            temperature
        ), dim=1)
        
        return processed_latent
    
    def process_prediction(self, prediction, shape):
        processed_prediction = prediction.reshape(
            shape[0],
            self.atom_num,
            3
        )
        return processed_prediction
    
    def train(self):
        self.encoder.train()
        self.decoder.train()
        
    def eval(self):
        self.encoder.eval()
        self.decoder.eval()
        