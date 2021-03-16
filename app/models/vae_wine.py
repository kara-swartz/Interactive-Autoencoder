import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms

import pandas as pd
import numpy as np
import math

from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances

#loading data
data_white = pd.read_csv("../app/static/data/winequality-white.csv", sep =";")
data_red = pd.read_csv("../app/static/data/winequality-red.csv", sep =";")
data = pd.concat([data_red, data_white])
df = data.drop(columns=['quality'])

#data noramlization
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(df.astype(float))
df = pd.DataFrame(x_scaled)

x_data = df.to_numpy()
x_data = torch.Tensor(x_data).float()

class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        
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
    
    def reparameterize(self, mu, var):
        # reparameterisation trick to sample z values
        std = torch.exp(0.5*var)
        eps = torch.randn_like(std) 
        return mu + eps*std
                
    def decoder(self, z):
        # decode latent variables
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h)) 
     
    def forward(self, x):
        # encodes samples and then decodes them
        mu, var = self.encoder(x.view(-1, 11))
        z = self.reparameterize(mu, var)
        return self.decoder(z), mu, var

class VAE:
    
    def loss_function(self, recon_x, x, mu, var):
        # return reconstruction error + KL divergence losses
        reconstruction_loss=nn.MSELoss()
        ER = reconstruction_loss(recon_x, x)
        KL = -0.5*torch.sum(1 + var - mu.pow(2) - var.exp())  
        return ER + KL
    
    def fn_train(self, data, num_epochs):
        self.model = EncoderDecoder()    
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        for epoch in range(1, num_epochs+1):
            train_loss = 0.0
            for epoch in range(1,epoch+1):
                recon_x, mu, var = self.model(data)
                loss=self.loss_function(recon_x, data, mu, var)
                train_loss+=loss.data
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'loss {epoch} = {loss}')
        mu, log_var = self.model.encoder(data)
        encoded = self.model.reparameterize(mu, log_var)
        decoded = self.model.decoder(encoded)
        return encoded.detach().numpy(), decoded.detach().numpy()
    
    def data_projection(self, data):
        mu, log_var = self.model.encoder(data)
        encoded = self.model.reparameterize(mu, log_var)
        decoded = self.model.decoder(encoded)
        return encoded.detach().numpy(), decoded.detach().numpy()

class LatentSpace:

    def qm_mse_dist(self, original, reduced):
        # mean squared error of distances
        hd_dists = euclidean_distances(original)
        ld_dists = euclidean_distances(reduced)
        total_squared_error = np.sum((hd_dists - ld_dists) ** 2)
        return total_squared_error / original.shape[0]   
        
    def get_latent(self):
        # number of epochs for retraining
        n_epochs = 10

        # train model
        self.vae = VAE()

        # get a latent space z
        z, decoded  = self.vae.fn_train(x_data, n_epochs)
        df_z = pd.DataFrame({'x':z[:,0], 'y':z[:, 1], 'label':data['quality'], 'is_trained':'no'})
        
        # quality measures. All points. Projection 1:
        self.decoded = decoded
        qm_mse_all_pr1 = round(self.qm_mse_dist(x_data, decoded), 4)
        return df_z, qm_mse_all_pr1

    def projection(self, points):
        # number of epochs for retraining
        n_epochs = 10

        # filter the selected points
        selected_points = x_data[points]
        
        # get a latent space z of the selected points
        z_selected, decoded_selected = self.vae.fn_train(selected_points, n_epochs)
        df_yes = pd.DataFrame({'x':z_selected[:,0], 'y':z_selected[:, 1], 'label':data['quality'].iloc[points], 'is_trained':'yes'})
        
        # get a latent space z of the all points        
        z_all_points, decoded_all_points = self.vae.data_projection(x_data)
        
        # filter non-trained data
        arr_no = np.delete(z_all_points, [points], axis=0)
        labels = data['quality'].to_numpy()
        arr_no_labels = np.delete(labels, [points])
        df_no = pd.DataFrame({'x':arr_no[:,0], 'y':arr_no[:, 1], 'label':arr_no_labels, 'is_trained':'no'})
        projection = pd.concat([df_yes, df_no])
        
        ## quality measures##

        # selected points. Projection 1:
        qm_mse_selected_pr1 = round(self.qm_mse_dist(selected_points, self.decoded[points]), 4)
        # all points. Projection 2:
        qm_mse_all_points_pr2 = round(self.qm_mse_dist(x_data, decoded_all_points), 4)
        # selected points. Projection 2:
        qm_mse_selected_pr2 = round(self.qm_mse_dist(selected_points, decoded_all_points[points]), 4)
        return projection, qm_mse_selected_pr1, qm_mse_all_points_pr2, qm_mse_selected_pr2