import torch
from torch import nn, cuda, optim, tensor
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import numpy as np
import pandas as pd
import math
import datetime
import timeit
from sklearn.metrics.pairwise import euclidean_distances

class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        
        # encoder part
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 20)
        self.fc31 = nn.Linear(20, 2)
        self.fc32 = nn.Linear(20, 2)
        
        # decoder part
        self.fc4 = nn.Linear(2, 20)
        self.fc5 = nn.Linear(20, 400)
        self.fc6 = nn.Linear(400, 784)
        
    def encoder(self, x):
        # encode data points, and return posterior parameters for each point.
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)
    
    def reparameterize(self, mu, log_var):
        # reparameterisation trick to sample z values
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std 
        
    def decoder(self, z):
        # decode latent variables
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h)) 
    
    def forward(self, x):
        # encodes samples and then decodes them
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

class VAE:
    
    def loss_function(self, recon_x, x, mu, log_var):
        # return reconstruction error + KL divergence losses
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD
    
    def fn_train(self, data, num_epochs, train_type):
        
        self.model = EncoderDecoder()
        if torch.cuda.is_available():
            self.model.cuda()

        batch_size = 100
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        date_time = datetime.datetime.now().strftime("%d%m%y%H%M%S")

        # train the model
        for epoch in range(num_epochs):
            for i in range(int(data.shape[0] / batch_size)):
                batch = data[i * batch_size:(i + 1) * batch_size]
                batch = tensor(batch, dtype=torch.float)
                if cuda.is_available():
                    batch = batch.cuda()
                optimizer.zero_grad()                
                recon_batch, mu, log_var = self.model(batch)
                loss = self.loss_function(recon_batch, batch, mu, log_var)
                loss.backward()
                optimizer.step()
        data = tensor(data, dtype=torch.float)
        if cuda.is_available():
            data = data.cuda()
        mu, log_var = self.model.encoder(data.view(-1, 784))
        encoded = self.model.reparameterize(mu, log_var)
        with torch.no_grad():
            z = torch.randn(64, 2).cuda()
            sample = self.model.decoder(z).cpu()    
            save_image(sample.view(64, 1, 28, 28), './results/vae_sample_' + date_time + '_' + train_type + '.png')
        return encoded
    
    def data_projection(self, data):
        mu, log_var = self.model.encoder(data.view(-1, 784))
        encoded = self.model.reparameterize(mu, log_var)
        decoded = self.model.decoder(encoded)
        return encoded, decoded

class LatentSpace:

    def qm_mse_dist(self, original, reduced):
        # mean squared error of distances
        hd_dists = euclidean_distances(original)
        ld_dists = euclidean_distances(reduced)
        total_squared_error = np.sum((hd_dists - ld_dists) ** 2)
        return total_squared_error / original.shape[0]

    def get_latent(self):

        train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=60000)         
        
        # number of epochs
        n_epochs = 10
        train_type = 'learn'
        
        # train model
        self.vae = VAE()

        for data in train_loader:
            data = data[0].numpy()
            print('Start projection training')
            start_time = timeit.default_timer()
            encoded = self.vae.fn_train(data,  n_epochs, train_type)
            print('Training done', timeit.default_timer() - start_time)

        # load one big batch for visualization
        self.train_loader_batch = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                    batch_size=10000)
        one_batch = next(iter(self.train_loader_batch))
        self.img, self.labels = one_batch
        #images_flatten = self.img.view(self.img.size(0), -1)
        self.img_batch = self.img.cuda()

        # get a latent space z
        encoded, decoded = self.vae.data_projection(self.img_batch)
        self.z = encoded.detach().cpu().numpy()
        df_z = pd.DataFrame({'x':self.z[:,0], 'y':self.z[:, 1], 'label':self.labels, 'is_trained':'no'})

        # quality measures. All points. Projection 1:
        self.decoded = decoded.detach().cpu().numpy()
        self.x_data = self.img.view(-1, 784).detach().numpy()
        qm_mse_all_pr1 = round(self.qm_mse_dist(self.x_data, self.decoded), 4)
        return df_z, qm_mse_all_pr1
    
    def projection(self, points):
        # number of epochs for retraining
        n_epochs = 10
        train_type = 'proj'

        # convert a batch of images to an array to filter the selected points
        img_array = self.img.numpy()
        # filter the selected points
        data = img_array[points]
        
        # get a latent space z of the selected points
        encoded_selected = self.vae.fn_train(data, n_epochs, train_type)
        z_selected = encoded_selected.detach().cpu().numpy()
        df_yes = pd.DataFrame({'x':z_selected[:,0], 'y':z_selected[:, 1], 'label':self.labels[points], 'is_trained':'yes'})
        
        # get a latent space z of the all points
        encoded_all_points, decoded_all_points = self.vae.data_projection(self.img_batch)
        z_all_points = encoded_all_points.detach().cpu().numpy()
        
        # filter non-trained data
        arr_no = np.delete(z_all_points, [points], axis=0)
        arr_no_labels = np.delete(self.labels, [points])
        df_no = pd.DataFrame({'x':arr_no[:,0], 'y':arr_no[:, 1], 'label':arr_no_labels, 'is_trained':'no'})
        projection = pd.concat([df_yes, df_no])
        
        ## quality measures##

        # selected points. Projection 1:
        qm_mse_selected_pr1 = round(self.qm_mse_dist(self.x_data[points], self.decoded[points]), 4)
        # all points. Projection 2:
        decoded_all_points = decoded_all_points.detach().cpu().numpy()
        qm_mse_all_points_pr2 = round(self.qm_mse_dist(self.x_data, decoded_all_points), 4)
        # selected points. Projection 2:
        qm_mse_selected_pr2 = round(self.qm_mse_dist(self.x_data[points], decoded_all_points[points]), 4)
        return projection, qm_mse_selected_pr1, qm_mse_all_points_pr2, qm_mse_selected_pr2