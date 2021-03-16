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
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), 
            nn.Linear(64, 12), 
            nn.ReLU(True), 
            nn.Linear(12, 2)
        )
        # decoder part
        self.decoder = nn.Sequential(
            nn.Linear(2, 12),
            nn.LeakyReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), 
            nn.Linear(128, 28 * 28), 
            nn.Tanh()
        )
        
    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return z, x

class AutoEncoder:
    
    def fn_train(self, data, num_epochs, train_type):

        self.model = EncoderDecoder()
        if torch.cuda.is_available():
            self.model.cuda()
        
        batch_size = 100
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        date_time = datetime.datetime.now().strftime("%d%m%y%H%M%S")
        
        # train the model        
        for epoch in range(num_epochs):
            for i in range(int(data.shape[0] / batch_size)):
                batch = data[i * batch_size:(i + 1) * batch_size]
                batch = tensor(batch, dtype=torch.float)
                if cuda.is_available():
                    batch = batch.cuda()
                # ===================forward=====================
                # forward pass: compute predicted outputs by passing inputs to the model
                encoded, decoded = self.model(batch)
                # calculate the loss
                loss = criterion(decoded, batch)
                # ===================backward====================
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
        data = tensor(data, dtype=torch.float)
        if cuda.is_available():
            data = data.cuda()
        encoded, decoded = self.model(data)
        try:
            with torch.no_grad():
                    z = encoded[0:64].cuda()
                    sample = self.model.decoder(z).cpu()    
                    save_image(sample.view(64, 1, 28, 28), './results/ae_sample_' + date_time + '_' + train_type + '.png')
        except RuntimeError:
            print("Number of points less than 64. Cannot save image") 
        return encoded, decoded

    def data_projection(self, data):
        encoded, decoded = self.model(data)
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
        self.ae = AutoEncoder()
        
        for data in train_loader:
            data = data[0].numpy()
            data = data.reshape((data.shape[0], -1))
            data = data.astype(np.float32)
            print('Start projection training')
            start_time = timeit.default_timer()
            encoded, decoded = self.ae.fn_train(data,  n_epochs, train_type)
            print('Training done', timeit.default_timer() - start_time)

        # load one big batch for visualization
        self.train_loader_batch = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                    batch_size=10000)
        one_batch = next(iter(self.train_loader_batch))
        self.img, self.labels = one_batch
        images_flatten = self.img.view(self.img.size(0), -1)
        self.all_data = images_flatten.cuda()

        # get a latent space z
        encoded, decoded = self.ae.data_projection(self.all_data)
        z = encoded.detach().cpu().numpy()
        df_z = pd.DataFrame({'x':z[:,0], 'y':z[:, 1], 'label':self.labels, 'is_trained':'no'})

        # quality measures. All points. Projection 1:
        self.x_data = self.img.view(-1, 784).detach().numpy()
        self.decoded = decoded.detach().cpu().numpy()
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
        data = data.reshape((data.shape[0], -1))
        data = data.astype(np.float32)

        # get a latent space z of the selected points
        encoded_selected, decoded_selected = self.ae.fn_train(data, n_epochs, train_type)
        z_selected = encoded_selected.detach().cpu().numpy()
        
        # convert array into dataframe
        df_yes = pd.DataFrame({'x':z_selected[:,0], 'y':z_selected[:, 1], 'label':self.labels[points], 'is_trained':'yes'})
        
        # get a latent space z of the all points
        encoded_all_points, decoded_all_points = self.ae.data_projection(self.all_data)
        z_all_points = encoded_all_points.detach().cpu().numpy()
        
        # all points except selected 
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