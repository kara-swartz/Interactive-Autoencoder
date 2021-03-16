import torch
from torch import nn, optim
from torch.nn import functional as F

import pandas as pd
import numpy as np
import math

from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances

# loading data
data_white = pd.read_csv("../app/static/data/winequality-white.csv", sep =";")
data_red = pd.read_csv("../app/static/data/winequality-red.csv", sep =";")
data = pd.concat([data_red, data_white])
df = data.drop(columns=['quality'])

# data noramlization
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(df.astype(float))
df = pd.DataFrame(x_scaled)

x_data = df.to_numpy()
x_data = torch.Tensor(x_data).float()

class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(11, 6),
            nn.Tanh(),
            nn.Linear(6, 2),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 6),
            nn.Tanh(),
            nn.Linear(6, 11),
            nn.LeakyReLU(),
        )
        
    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return z, x

class AutoEncoder:
    
    def fn_train(self, x_data, num_epochs):
              
        self.model = EncoderDecoder()
        loss_fun = nn.MSELoss()
        opt= torch.optim.SGD(self.model.parameters(),lr=0.01)
        
        for epoch in range(1, num_epochs+1):
            encoded, decoded = self.model(x_data)
            # calculate loss
            loss = loss_fun(decoded,x_data)                        
            # clear previous gradients
            opt.zero_grad()      
            # compute gradients of all variables wrt loss                                       
            loss.backward()  
            # perform updates using calculated gradients                                           
            opt.step()
            print(f'loss {epoch} = {loss}')
        encoded, decoded = self.model(x_data)
        return encoded, decoded
    
    def data_projection(self, x_data):
        encoded, decoded = self.model(x_data)
        return encoded, decoded

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
        self.ae = AutoEncoder()

        # get a latent space z
        encoded, decoded  = self.ae.fn_train(x_data, n_epochs)
        # convert the latent variable z into dataframe
        z = encoded.data.numpy()
        df_z = pd.DataFrame({'x':z[:,0], 'y':z[:, 1], 'label':data['quality'], 'is_trained':'no'})
        # quality measures. All points. Projection 1:
        self.decoded = decoded.detach().cpu().numpy()
        qm_mse_all_pr1 = round(self.qm_mse_dist(x_data, self.decoded), 4)
        return df_z, qm_mse_all_pr1
    
    def projection(self, points):
        # number of epochs for retraining
        n_epochs = 10

        # filter the selected points
        selected_points = x_data[points]
        
        # train model and get a latent space z of the selected points
        encoded_selected, decoded_selected = self.ae.fn_train(selected_points, n_epochs)
        z_selected = encoded_selected.data.numpy()
        df_yes = pd.DataFrame({'x':z_selected[:,0], 'y':z_selected[:, 1], 'label':data['quality'].iloc[points], 'is_trained':'yes'})
        
        # get a latent space z of the all points
        encoded_all_points, decoded_all_points = self.ae.data_projection(x_data)
        z_all_points = encoded_all_points.data.numpy()
        
        # all points except selected 
        arr_no = np.delete(z_all_points, [points], axis=0)
        labels = data['quality'].to_numpy()
        arr_no_labels = np.delete(labels, [points])
        df_no = pd.DataFrame({'x':arr_no[:,0], 'y':arr_no[:, 1], 'label':arr_no_labels, 'is_trained':'no'})
        projection = pd.concat([df_yes, df_no])
        
        ## quality measures##
        # selected points. Projection 1:
        qm_mse_selected_pr1 = round(self.qm_mse_dist(selected_points, self.decoded[points]), 4)
        # all points. Projection 2:
        decoded_all_points = decoded_all_points.detach().cpu().numpy()
        qm_mse_all_points_pr2 = round(self.qm_mse_dist(x_data, decoded_all_points), 4)
        # selected points. Projection 2:
        decoded_selected = decoded_selected.detach().cpu().numpy()
        qm_mse_selected_pr2 = round(self.qm_mse_dist(selected_points, decoded_selected), 4)        
        return projection, qm_mse_selected_pr1, qm_mse_all_points_pr2, qm_mse_selected_pr2