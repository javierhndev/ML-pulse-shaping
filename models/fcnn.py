import time

import torch
import torch.nn as nn
import torch.optim as optim

	
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

#X-> Dazzler parameters
#Y -> Pulse shape
class FWmodelNN:
    def __init__(self,X_train,y_train,X_test,y_test,device):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.device = device

        self.in_features = len(self.X_train.columns)
        self.out_features = len(self.y_train.columns)
        
    def train(self,n_epochs,batch_size,print_freq,learning_rate):
        #normalize the X (using the X_train)
        X_train_norm=norm_data(self.X_train,self.X_train)
        X_test_norm=norm_data(self.X_test,self.X_train) #normalized by the SAME values

        #convert data to tensors
        X_train_tensor=torch.tensor(X_train_norm.values,dtype=torch.float32)
        y_train_tensor=torch.tensor(self.y_train.values,dtype=torch.float32)

        X_test_tensor=torch.tensor(X_test_norm.values,dtype=torch.float32)
        y_test_tensor=torch.tensor(self.y_test.values,dtype=torch.float32)

        #define the model
        model=perceptron_fwd(self.in_features,self.out_features)

        #training
        self.nn_fc_model, self.train_error, self.test_error=train_nn(X_train_tensor,y_train_tensor,X_test_tensor,y_test_tensor,
			model,self.device,n_epochs,batch_size,print_freq,learning_rate)
        
    def predict(self,X_data):
        #normalize the X_data using the X_train
        X_data_norm=norm_data(X_data,self.X_train)

        #convert it to tensor
        X_data_tensor=torch.tensor(X_data_norm.values,dtype=torch.float32)
        X_data_tensor=X_data_tensor.to(self.device)

        #make the prediction
        y_predict_tensor=self.nn_fc_model(X_data_tensor)
        self.y_predict_nn=y_predict_tensor.detach().cpu().numpy()
        return self.y_predict_nn
        
    def error_calc_mae(self):
        print('Calculating the Mean Absolute Error')
        #Reset the index on y_test to have the same indexes as y_predict
        y_test_reset=self.y_test.reset_index(drop=True)
        #study the erro distribution
        mae_error=abs(y_test_reset-self.y_predict_nn)
        mae_error=mae_error.sum(axis=1)/self.y_test.shape[1] #sum error / num columns
        return mae_error
    

#############################3

#X-> Dazzler parameters
#Y -> Pulse shape
class ROUNDmodelNN:
    def __init__(self,X_train,y_train,X_test,y_test,device):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.device = device

        self.X_features = len(self.X_train.columns)
        self.y_features = len(self.y_train.columns)
        
    def train(self,n_epochs_fwd,batch_size_fwd,print_freq_fwd,learning_rate_fwd,
             n_epochs_bwd,batch_size_bwd,print_freq_bwd,learning_rate_bwd):
        #normalize the X (using the X_train)
        X_train_norm=norm_data(self.X_train,self.X_train)
        X_test_norm=norm_data(self.X_test,self.X_train) #normalized by the SAME values

        #convert data to tensors
        X_train_tensor=torch.tensor(X_train_norm.values,dtype=torch.float32)
        y_train_tensor=torch.tensor(self.y_train.values,dtype=torch.float32)

        X_test_tensor=torch.tensor(X_test_norm.values,dtype=torch.float32)
        y_test_tensor=torch.tensor(self.y_test.values,dtype=torch.float32)

        #define the models
        model_fwd=perceptron_fwd(self.X_features,self.y_features)
        model_bwd=perceptron_bwd(self.y_features,self.X_features)

        #training
        #train the fwd model
        print('Training the forward model')
        self.nn_fc_model_fwd, self.train_error_fwd, self.test_error_fwd=train_nn(X_train_tensor,y_train_tensor,X_test_tensor,y_test_tensor,
			model_fwd,self.device,n_epochs_fwd,batch_size_fwd,print_freq_fwd,learning_rate_fwd)
        #train the backward model  (reverse the order of the input and target)
        print('Training the backward model')
        self.nn_fc_model_bwd,  self.train_error_bwd, self.test_error_bwd=train_nn(y_train_tensor,X_train_tensor,y_test_tensor,X_test_tensor,
			model_bwd,self.device,n_epochs_bwd,batch_size_bwd,print_freq_bwd,learning_rate_bwd)
        
    def predict(self,y_data):
        #y_data is assumed to be a DataFrame.
        #  If it is a single shot in a list, use: pd.DataFrame([y_data[0]])

        #convert y to tensor
        y_data_tensor=torch.tensor(y_data.values,dtype=torch.float32)
        y_data_tensor=y_data_tensor.to(self.device)
        #predict X (using the bacward model)
        X_predict_tensor=self.nn_fc_model_bwd(y_data_tensor)

        #predict y (using the forward model and the X prediction
        y_predict_tensor=self.nn_fc_model_fwd(X_predict_tensor)
        self.y_predict_nn=y_predict_tensor.detach().cpu().numpy()

        #save also the X prediction
        #back to numpy
        X_predict_norm=X_predict_tensor.detach().cpu().numpy()
        #reverse norm
        self.X_predict_nn=renorm_data(X_predict_norm,self.X_train)
        
        return self.y_predict_nn
        
    def error_calc_mae(self):
        print('Calculating the Mean Absolute Error')
        #Reset the index on y_test to have the same indexes as y_predict
        y_test_reset=self.y_test.reset_index(drop=True)
        #study the erro distribution
        mae_error=abs(y_test_reset-self.y_predict_nn)
        mae_error=mae_error.sum(axis=1)/self.y_test.shape[1] #sum error / num columns
        return mae_error
########################################
#Here Y is the target and X is the input
#NOTE: that Y and X could be switched depending if it is forward or backward
def train_nn(X_train_tensor,y_train_tensor,X_test_tensor,y_test_tensor,
			model,device,n_epochs,batch_size,print_freq,learning_rate):
	
	startTime = time.time()
	#Create the model and define the loss and optimizer
	nn_fc_model=model.to(device)
	#nn_model=cnn().to(device)
	#print(nn_fc_model)

	loss_func=nn.MSELoss() #mean squared error
	optimizer = optim.Adam(nn_fc_model.parameters(), lr=learning_rate)

	#main training loop
	train_error=[]
	test_error=[]
	epoch_list=[]
	for epoch in range(n_epochs):
    	# set the model in training mode
		nn_fc_model.train()

		train_loss=0
		for i in range(0,len(y_train_tensor),batch_size):
			X_batch=X_train_tensor[i:i+batch_size]
			y_batch=y_train_tensor[i:i+batch_size]
			X_batch, y_batch =(X_batch.to(device), y_batch.to(device))
			y_predict_nn=nn_fc_model(X_batch)
			loss=loss_func(y_predict_nn,y_batch)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			train_loss += loss

    	#get training error
		train_loss=train_loss/(len(X_train_tensor)/batch_size)
		train_error.append(train_loss.detach().cpu().numpy())
    	#evaluate test error
		with torch.no_grad():
			nn_fc_model.eval()
			(X_test_tensor, y_test_tensor) = (X_test_tensor.to(device), y_test_tensor.to(device))
			y_predict_test=nn_fc_model(X_test_tensor)
			test_loss=loss_func(y_predict_test,y_test_tensor)
			test_error.append(test_loss.detach().cpu().numpy())

		epoch_list.append(epoch+1)
    
		if(epoch%print_freq==0 or epoch+1==n_epochs):
			print(f'Finished epoch {epoch},latest loss {train_loss}')
	#print(train_error)
	#print(test_error)
	endTime = time.time()
	print("Total time taken to train the model: {:.2f}s".format(endTime - startTime))
	
	return nn_fc_model, train_error, test_error
	
##################################
class perceptron_fwd(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        self.hidden1=nn.Linear(in_features,3*in_features)
        self.act1= nn.ReLU()
        self.hidden2=nn.Linear(3*in_features,20*in_features)
        self.act2=nn.ReLU()
        self.hidden3=nn.Linear(20*in_features,int(0.5*out_features))
        self.act3=nn.ReLU()
        self.hidden4=nn.Linear(int(0.5*out_features),out_features)
        self.act4=nn.ReLU()
        self.hidden5=nn.Linear(out_features,int(1.5*out_features))
        self.act5=nn.ReLU()
        self.hidden6=nn.Linear(int(1.5*out_features),int(1.3*out_features))
        self.act6=nn.ReLU()
        self.output=nn.Linear(int(1.3*out_features),out_features)

    def forward(self,x):
        x=self.act1(self.hidden1(x))
        x=self.act2(self.hidden2(x))
        x=self.act3(self.hidden3(x))
        x=self.act4(self.hidden4(x))
        x=self.act5(self.hidden5(x))
        x=self.act6(self.hidden6(x))
        x=self.output(x)
        return x
#######################################
class perceptron_bwd(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        self.hidden1=nn.Linear(in_features,in_features)
        self.act1= nn.ReLU()
        self.hidden2=nn.Linear(in_features,in_features)
        self.act2=nn.ReLU()
        self.hidden3=nn.Linear(in_features,int(0.5*in_features))
        self.act3=nn.ReLU()
        self.hidden4=nn.Linear(int(0.5*in_features),int(0.5*in_features))
        self.act4=nn.ReLU()
        self.hidden5=nn.Linear(int(0.5*in_features),int(0.25*in_features))
        self.act5=nn.ReLU()
        self.hidden6=nn.Linear(int(0.25*in_features),5*out_features)
        self.act6=nn.ReLU()
        self.output=nn.Linear(5*out_features,out_features)

    def forward(self,x):
        x=self.act1(self.hidden1(x))
        x=self.act2(self.hidden2(x))
        x=self.act3(self.hidden3(x))
        x=self.act4(self.hidden4(x))
        x=self.act5(self.hidden5(x))
        x=self.act6(self.hidden6(x))
        x=self.output(x)
        return x

