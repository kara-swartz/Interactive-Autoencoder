import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F

epochs = 700
encoded = []

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        # encoder part
        self.fc1 = nn.Linear(11,6)
        self.fc21 = nn.Linear(6,2)
        self.fc22 = nn.Linear(6,2)
        # decoder part
        self.fc3 = nn.Linear(2,6)
        self.fc4 = nn.Linear(6,11)
        
    def encoder(self, x):
        # encode data points, and return posterior parameters for each point.
        h = F.relu(self.fc1(x))
        mu = F.relu(self.fc21(h))
        variance = F.relu(self.fc22(h))
        return mu, variance
    
    def decoder(self, z):
        # decode latent variables
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h)) 
                
    def reparameterise(self, mu, var):
         # reparameterisation trick to sample z values
        std = torch.exp(0.5*var)
        eps = torch.randn_like(std) 
        return mu + eps*std
     
    def forward(self, x):
        mu,var=self.encoder(x)
        encoded=self.reparameterise(mu, var)
        decoded=self.decoder(encoded)        
        return encoded, decoded, mu, var
    
    def loss_fn(self, x, recon_x, mu, var):
        # return reconstruction error + KL divergence losses
        reconstruction_loss=nn.MSELoss()
        ER = reconstruction_loss(recon_x, x)
        KL = -0.5*torch.sum(1 + var - mu.pow(2) - var.exp())  
        return ER + KL

    def get_data(self, x_data):
        self.x_data = torch.Tensor(x_data).float()
        return self.x_data
    
    def train(self):
        self.model=VAE()    
        self.model = self.model.float() 
        optimizer=torch.optim.Adam(self.model.parameters(), lr=1e-3)
        train_loss=0.0
        for epoch in range(1,epochs+1):
            ############ forward ##############
            encoded, decoded, mu, var=self.model(self.x_data)
            loss=self.loss_fn(self.x_data, decoded, mu, var)
            train_loss+=loss.data
            ############ backward ##############
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'loss {epoch} = {loss}')
        return encoded.detach().numpy(), decoded.detach().numpy()
    
    def projection(self, pr_data):
        pr_data = torch.Tensor(pr_data).float()
        encoded, decoded, mu, var=self.model(pr_data)
        return encoded.detach().numpy(), decoded.detach().numpy()