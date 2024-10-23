import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


import copy








class MLP(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        """
        Initializes the MLP.
        
        Args:
        - input_size (int): Size of the input layer.
        - layer_sizes (list of int): A list of integers where each element specifies the number of neurons in a hidden layer.
        - output_size (int): Size of the output layer.
        """
        super(MLP, self).__init__()
        
        # Create a list to hold all layers
        layers = []
        
        # Input layer to the first hidden layer
        layers.append(nn.Linear(input_size, layer_sizes[0]))
        
        # Add hidden layers
        for i in range(1, len(layer_sizes)):
            layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
        
        # Output layer
        layers.append(nn.Linear(layer_sizes[-1], output_size))
        
        # Save the layers as a ModuleList so PyTorch can track them
        self.layers = nn.ModuleList(layers)

        self.double()

    def init_weights(self):
        for i in range(len(self.layers)):
            torch.nn.init.normal_(self.layers[i].weight, mean=0.0, std=0.1)

    

        
    def forward(self, x):
        # Pass input through each layer with ReLU activation except the output layer
        for layer in self.layers[:-1]:
            #x = torch.sigmoid(layer(x))
            x = F.relu(layer(x))
        
        # Pass through the final layer without activation
        x = self.layers[-1](x)
        
        return x.flatten() 



    def train_and_validate(self, inputs, targets, validation_inputs, validation_targets, num_epochs, criterion, optimizer, early_stopping):

        train_loss_list = []
        valid_loss_list = []

        n_train_samples = inputs.shape[-1]
        n_val_samples = validation_inputs.shape[-1]


        MAX_PATIENCE = 14

        
        #Initialize Variables for EarlyStopping
        best_loss = float('inf')
        best_model_weights = None
        patience = MAX_PATIENCE



        for epoch in range(num_epochs):

            
            #Validation
            running_vloss = 0.0
            self.eval()
            with torch.no_grad():
                validation_outputs = self(validation_inputs)
                val_loss = criterion(validation_outputs, validation_targets)#/ n_val_samples



            self.train()

            #Reset gradient
            optimizer.zero_grad()


            #Training
            outputs = self(inputs)
            train_loss = criterion(outputs, targets.t()) #/ n_train_samples
            train_loss.backward()
            optimizer.step()

            if early_stopping:
                # Early stopping
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_weights = copy.deepcopy(self.state_dict())  # Deep copy here
                    best_model_epoch = epoch
                    patience = MAX_PATIENCE  # Reset patience counter
                else:
                    patience -= 1
                    if patience == 0:
                        print("Early Stopping at epoch:", epoch, " saved model is from epoch:", best_model_epoch)
                        break

            #print loss
            print("Epoch", epoch, "\t: training loss =", train_loss.item(), "\t: validation loss =", val_loss.item())

            train_loss_list.append(train_loss.item())
            valid_loss_list.append(val_loss.item())
    

        print("Finished training\tfinal training error:", train_loss_list[-1], "\tfinal validation error:", valid_loss_list[-1])


        #If no early_Stopping --> just take the last sample as best
        if not early_stopping:
            best_model_weights = copy.deepcopy(self.state_dict())
        

            
        

        return best_model_weights, train_loss_list, valid_loss_list

