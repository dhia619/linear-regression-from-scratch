import numpy as np 
import os

class LinearRegression:
    def __init__(self,num_features):
        self.weights = np.random.randn(1,num_features)
        self.bias = 0

    def forward_pass(self,x):
        return np.dot(self.weights,x) + self.bias

    def mse(self,predicted, actual):
        return np.mean((predicted - actual) ** 2)

    def fit(self,epochs,lr,x,actual):
        n = np.shape(x)[1]
        for epoch in range(epochs):
            z = self.forward_pass(x)
            dw = 2/n * np.dot(z-actual,x.T)
            db = 2/n * np.sum(z-actual)
            self.weights -= lr * dw
            self.bias -= lr * db

            if epoch % 10000 == 0:
                print(f"epoch {epoch} | loss = {self.mse(z, actual)}")

    def r2(self,x,actual):
        z = self.forward_pass(x)
        ss_res = np.sum((z-actual)**2)
        ss_tot = np.sum((actual-np.mean(actual))**2)
        return 1 - (ss_res / ss_tot)
    
    def save_model(self, file_prefix='model'):
        os.mkdir("model")
        os.chdir("model")
        for i, layer in enumerate(self.layers):
            np.save(f'{file_prefix}_weights_{i}.npy', layer.weights)
            np.save(f'{file_prefix}_biases_{i}.npy', layer.biases)
        os.chdir("..")
    
    def load_model(self, file_prefix='model'):
        for i, layer in enumerate(self.layers):
            layer.weights = np.load(f'{file_prefix}_weights_{i}.npy')
            layer.biases = np.load(f'{file_prefix}_biases_{i}.npy')