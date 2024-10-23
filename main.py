import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


from mackey_glass import Dataset
from mlp import MLP







#Model Parameters
layer_sizes = [10,10,10]

#Training parameters
NUM_EPOCHS = 100
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.001
EARLY_STOPPING = True

#Mackey Glass Parameters
t_start = 301
t_end = 1500
valid_dist = 0.25
ADD_ZERO_MEAN_NOISE = False






#Number of data samples
n_samples = t_end - t_start + 1


#Generate Mackey Glass Time Series data points
dataset = Dataset(beta=0.2, gamma=0.1, n=10, tau=25, guess_length=5)
x = dataset.eulers_method(t_end)

X_in, X_out = dataset.organise_to_in_out_matrices(x, t_start, t_end)
num_samples, input_size = X_in.shape
_, output_size = X_out.shape

np.random.seed(84)




if ADD_ZERO_MEAN_NOISE:
    X_in_run = X_in + np.random.normal(0, 0.15, (num_samples, input_size))
    X_out_run = X_out + np.random.normal(0, 0.15, (n_samples, output_size))
else:
    X_in_run = X_in
    X_out_run = X_out

#input_training, input_validation, input_testing, output_training, output_validation, output_testing = dataset.split_samples_no_shuffle_test(X_in, X_out, valid_dist, n_samples)
input_training, input_validation, input_testing, output_training, output_validation, output_testing = dataset.split_samples(X_in_run, X_out_run, valid_dist, n_samples)



#Initalize model
model = MLP(input_size=input_size, layer_sizes=layer_sizes, output_size=output_size)
model.init_weights()

#Use MSE for Loss function and use Adam for training
criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

#Start training
best_model_weights, train_loss_list, valid_loss_list = model.train_and_validate(input_training, output_training, input_validation, output_validation, NUM_EPOCHS, criterion, optimizer, early_stopping=EARLY_STOPPING)



#Save best model
torch.save(best_model_weights, "trained_mlp.pth")


# Create new model and load states
best_model = MLP(input_size=input_size, layer_sizes=layer_sizes, output_size=output_size)

# Load the state dict
best_state_dict = torch.load('trained_mlp.pth', weights_only=True)

# Apply the loaded state dict to your model
best_model.load_state_dict(best_state_dict)
best_model.eval()


# Plot training and validation loss
t = np.arange(1, len(train_loss_list)+1)
plt.plot(t, train_loss_list, c='blue', label='Training Loss')
plt.plot(t, valid_loss_list, c='red', label='Validation Loss')
plt.legend()

plt.title("Training and Validation Loss")
plt.ylabel("MSE Loss")
plt.xlabel("Epoch")
plt.show()


#Test on unseen testing data
prediction_unseen = best_model(input_testing)

plt.plot(output_testing)
plt.plot(prediction_unseen.detach())

plt.title("Mackey-Glass Time Series Prediction with MLP, unseen sequence")
plt.ylabel("Value")
plt.xlabel("Time Step")
plt.show()


#Test on whole
prediction_whole_sequence = best_model(torch.tensor(X_in))

plt.plot(X_out)
plt.plot(prediction_whole_sequence.detach())

plt.title("Mackey-Glass Time Series Prediction with MLP, full sequence")
plt.ylabel("Value")
plt.xlabel("Time Step")
plt.show()