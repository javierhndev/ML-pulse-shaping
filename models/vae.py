import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

from .types_ import *

import time

#############################################
	
def norm_data(X_data,X_norm):
	#X_data is the array we want to normalize
	#X_norm is what we are using to normalize
	X_norm_out=X_data.copy()
	
	X_norm_out['order2']= (X_norm_out['order2']-X_norm['order2'].mean())/X_norm['order2'].std()
	
	X_norm_out['order3']= (X_norm_out['order3']-X_norm['order3'].mean())/X_norm['order3'].std()
	

	X_norm_out['order4']= (X_norm_out['order4']-X_norm['order4'].mean())/X_norm['order4'].std()
	
	
	#print(X_norm_out['order4'])
	
	return X_norm_out

###############################################

def renorm_data(X_data,X_norm):
	X_renorm=X_data.copy()
	
	X_renorm[:,0]=(X_renorm[:,0]*X_norm['order2'].std())+X_norm['order2'].mean()
	X_renorm[:,1]=(X_renorm[:,1]*X_norm['order3'].std())+X_norm['order3'].mean()
	X_renorm[:,2]=(X_renorm[:,2]*X_norm['order4'].std())+X_norm['order4'].mean()
	
	return X_renorm
	
######################################3

#Y: Laser pulse
#X: Dazzler param
def train_vae(y_train,y_test,X_train,X_test,
			model,device,n_epochs,batch_size,print_freq,learning_rate):
	
    startTime = time.time()

    #norm the X (Dazz parameters)
    X_train_norm=norm_data(X_train,X_train)
    X_test_norm=norm_data(X_test,X_train)

    #convert data to tensors (no need to normalize values for Y)
    y_train_tensor=torch.tensor(y_train.values,dtype=torch.float32)
    y_test_tensor=torch.tensor(y_test.values,dtype=torch.float32)

    X_train_tensor=torch.tensor(X_train_norm.values,dtype=torch.float32)
    X_test_tensor=torch.tensor(X_test_norm.values,dtype=torch.float32)

    #Create the model and define the loss and optimizer
    nn_model=model.to(device)

    print('Training on:',device)

    #loss_func=nn_model.loss_func()
    optimizer = optim.Adam(nn_model.parameters(), lr=learning_rate)

    #main training loop
    train_error=[]
    train_error_rec=[]
    train_error_kl=[]
    test_error=[]
    test_error_rec=[]
    test_error_kl=[]
    epoch_list=[]
    for epoch in range(n_epochs):
        # set the model in training mode
        nn_model.train()
        
        train_loss=0
        train_loss_rec=0
        train_loss_kl=0
        for i in range(0,len(y_train_tensor),batch_size):
            y_batch=y_train_tensor[i:i+batch_size]
            X_batch=X_train_tensor[i:i+batch_size]
            y_batch =y_batch.to(device)
            X_batch =X_batch.to(device)
            result=nn_model(X_batch,y_batch) #results is a list of [prediction,input,mu,log_var] to be used on the loss function
            loss=nn_model.loss_function(*result,batch_size/len(y_train_tensor))
            optimizer.zero_grad()
            loss['loss'].backward()
            optimizer.step()
            train_loss += loss['loss']
            train_loss_rec += loss['Reconstruction_Loss']
            train_loss_kl += -loss['KLD']

        #get training error
        train_loss=train_loss/(len(y_train_tensor)/batch_size)
        train_loss_rec=train_loss_rec/(len(y_train_tensor)/batch_size)
        train_loss_kl=train_loss_kl/(len(y_train_tensor)/batch_size)
        train_error.append(train_loss.detach().cpu().numpy())
        train_error_rec.append(train_loss_rec.detach().cpu().numpy())
        train_error_kl.append(train_loss_kl.detach().cpu().numpy()*batch_size/len(y_train_tensor))
        #evaluate test error
        with torch.no_grad():
            nn_model.eval()
            y_test_tensor = y_test_tensor.to(device)
            X_test_tensor = X_test_tensor.to(device)
            result=nn_model(X_test_tensor,y_test_tensor)
            test_loss=nn_model.loss_function(*result,batch_size/len(y_train_tensor))
            test_error.append(test_loss['loss'].detach().cpu().numpy())
            test_error_rec.append(test_loss['Reconstruction_Loss'].detach().cpu().numpy())
            test_error_kl.append(-test_loss['KLD'].detach().cpu().numpy()*batch_size/len(y_train_tensor))

        epoch_list.append(epoch+1)
    
        if(epoch%print_freq==0 or epoch+1==n_epochs):
            print(f'Finished epoch {epoch},latest loss {train_loss}')
    #print(train_error)
    #print(test_error)
    endTime = time.time()
    print("Total time taken to train the model: {:.2f}s".format(endTime - startTime))
	
    return nn_model,[train_error,train_error_rec,train_error_kl],[test_error,test_error_rec,test_error_kl]


#############################

class ConditionalVAE(nn.Module):


    def __init__(self,
                 in_pulse_features: int, #number of points in pulse shape
                 num_param: int, #number of Dazzler coefficients
                 latent_dim: int,
                 hidden_dims_enc: List,
                 hidden_dims_dec: List) -> None:
        super(ConditionalVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []

        in_features= in_pulse_features+num_param  #include pulse and dazz input
        # Build Encoder
        for h_dim in hidden_dims_enc:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features,h_dim),
                    nn.LeakyReLU(0.2))
            )
            in_features = h_dim

        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(hidden_dims_enc[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims_enc[-1], latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim+in_pulse_features, hidden_dims_dec[0])

        in_features=hidden_dims_dec[0]
        for h_dim in hidden_dims_dec[1:]:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features,h_dim),
                    nn.LeakyReLU(0.2))
            )
            in_features = h_dim



        self.decoder = nn.Sequential(*modules)



    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [B x Input]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x O]
        """
        result = self.decoder_input(z)
        result = self.decoder(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self,
                 input_dazz: Tensor,
                   input_pulse: Tensor) -> List[Tensor]:
        input=torch.cat([input_pulse,input_dazz],dim=1) #condition and input concatenated
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        z=torch.cat([z,input_pulse],dim=1) #concatenate latent z and condition
        return  [self.decode(z), input_dazz, mu, log_var]

    def loss_function(self,
                      *args) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = args[4] #batch_size/num_of_images # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    
    def sample(self,
               num_samples:int,
               in_pulse: Tensor,
               current_device: int) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        in_pulse=in_pulse.repeat(num_samples,1)
        z = torch.randn(num_samples,
                        self.latent_dim)

        z=torch.cat([z,in_pulse],dim=1)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, in_dazz: Tensor, in_pulse: Tensor,
                 current_device: int) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x D]
        :return: (Tensor) [B x D]
        """
        in_dazz = in_dazz.to(current_device)
        in_pulse = in_pulse.to(current_device)
        return self.forward(in_dazz, in_pulse)[0]