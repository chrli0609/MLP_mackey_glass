import numpy as np
import torch


class Dataset:

    def __init__(self, beta, gamma, n, tau, guess_length):
        self.beta = beta
        self.gamma = gamma
        self.n = n
        self.tau = tau

        self.guess_length = guess_length




    def eulers_method(self, end_t):
        x = np.zeros(end_t + self.guess_length)
        #x = np.array([None]*(end_t+self.guess_length))
    
        x[0] = 1.5    
        for t in range(end_t):
            x[t + 1] = x[t] + self.beta*x[t - self.tau] / (1 + x[t - self.tau]**self.n) - 0.1 * x[t]

        return x



    #Organise such that:
    #X_in = [[x(301), x(296), x(291), x(286), x(281)],
    #        [x(302), x(297), x(292), x(287), x(282)],
    #        [x(303), x(298), x(293), x(288), x(283)],
    #           ...
    #        [x(1499), x(1494), x(1489), x(1484), x(1479)]
    #        [x(1500), x(1495), x(1490), x(1485), x(1480)]]

    #X_out = [x(306),
    #         x(307),
    #         x(308),
    #         ... 
    #         x(1504),
    #         x(1505)]
    def organise_to_in_out_matrices(self, x, start_t, end_t):
        
        num_samples = (end_t + 1) - start_t 


        X_in = np.ones((num_samples, self.guess_length))
        X_out = np.ones((num_samples, 1))


        row_it = 0
        for i in range(start_t-1, end_t):

            X_out[row_it] = x[i + self.guess_length]

            for j in range(self.guess_length):

                X_in[row_it][j] = x[i-5*j]
            
            row_it += 1

        return X_in, X_out



    #Expected in and out dimensions: (5, 1200) and (1, 1200) respectively
    #split into training(1|5, 800) validation(1|5, 200) testing(1|5, 200)
    #Args are np.array but function returns torch.tensor
    def split_samples(self, X_in, X_out, valid_dist, n_samples):

        X = np.hstack((X_in, X_out))
        np.random.shuffle(X)
        X = X.T

        X_in = X[:-1, :]
        X_out = X[-1, :]
        #X_out = X_out.reshape(-1,1)


        
        training_samples = int(n_samples * (1-valid_dist))
        testing_samples = 200
        
        input_training = X_in[:, :training_samples]
        input_validation = X_in[:, training_samples:(n_samples - testing_samples)]
        input_testing = X_in[:, (n_samples - testing_samples):]

        output_training = X_out[:training_samples]
        output_validation = X_out[training_samples:(n_samples - testing_samples)]
        output_testing = X_out[(n_samples - testing_samples):]

        #print("output_training", output_training.shape)
        #print("output_validation", output_validation.shape)
        #print("output_testing", output_testing.shape)
        
        return torch.tensor(input_training.T), torch.tensor(input_validation.T), torch.tensor(input_testing.T), torch.tensor(output_training.T), torch.tensor(output_validation.T), torch.tensor(output_testing.T)
        

        
    def split_samples_no_shuffle_test(self, X_in, X_out, valid_dist, n_samples):
        training_samples = int(n_samples * (1-valid_dist))
        testing_samples = 200


        #Carve out last 200 for testing
        input_testing = X_in[(n_samples-testing_samples):, :]
        output_testing = X_out[(n_samples-testing_samples):]

        X_in = X_in[:(n_samples-testing_samples), :]
        X_out = X_out[:(n_samples-testing_samples)]




        #Stack together and shuffle
        X = np.hstack((X_in, X_out))


        np.random.shuffle(X)
        X = X.T

        X_in = X[:-1, :]
        X_out = X[-1, :]
        X_out = X_out.reshape(-1,1)


        
        
        
        input_training = X_in[:, :training_samples]
        input_validation = X_in[:, training_samples:(n_samples - testing_samples)]
        

        output_training = X_out[:training_samples]
        output_validation = X_out[training_samples:(n_samples - testing_samples)]
        

        #print("output_training", output_training.shape)
        #print("output_validation", output_validation.shape)
        #print("output_testing", output_testing.shape)
        
        return torch.tensor(input_training.T), torch.tensor(input_validation.T), torch.tensor(input_testing.T), torch.tensor(output_training.T), torch.tensor(output_validation.T), torch.tensor(output_testing.T)
