import torch
import torch.nn as nn

learning_rate = 0.01
epochs = 700

class AutoEncoder(nn.Module):

    x_data = []

    def __init__(self):
        super(AutoEncoder, self).__init__()

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
    
    def get_data(self, x_data):
        self.x_data = torch.Tensor(x_data).float()
        return self.x_data
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def train(self):
        self.model = AutoEncoder()
        loss_fun = nn.MSELoss()
        opt= torch.optim.SGD(self.model.parameters(),lr=learning_rate)
        for epoch in range(1,epochs+1):
            encoded, decoded = self.model(self.x_data)
            # calculate loss
            loss = loss_fun(decoded,self.x_data)                        
            # clear previous gradients
            opt.zero_grad()      
            # compute gradients of all variables wrt loss                                       
            loss.backward()  
            # perform updates using calculated gradients                                           
            opt.step()
            print(f'loss {epoch} = {loss}')
            """ print(torch.sum(self.model.encoder[0].weight))
            print(torch.sum(self.model.decoder[0].weight)) """                                                  
        #print(self.model)        
        return encoded.detach().numpy(), decoded.detach().numpy() 
    
    def projection(self, pr_data):
        pr_data = torch.Tensor(pr_data).float()
        encoded, decoded = self.model(pr_data)
        """ print(torch.sum(self.model.encoder[0].weight))
        print(torch.sum(self.model.decoder[0].weight)) """
        return encoded.detach().numpy(), decoded.detach().numpy()
        