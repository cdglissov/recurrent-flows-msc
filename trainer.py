'''Train script file'''
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

import os

from math import log, pi, exp
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision import transforms
import torch.distributions as td
from data_generators import stochasticMovingMnist
from RFN import RFN
from utils import *
n_bits = 7

n_frames = 6
three_channels=False
batch_size=64
testset = stochasticMovingMnist.MovingMNIST(False, 'Mnist', seq_len=n_frames, image_size=32, digit_size=24, num_digits=1, 
                                            deterministic=False, three_channels=three_channels, step_length=2, normalize=False)
trainset = stochasticMovingMnist.MovingMNIST(True, 'Mnist', seq_len=n_frames, image_size=32, digit_size=24, num_digits=1, 
                                              deterministic=False, three_channels=three_channels, step_length=2, normalize=False)
device = set_gpu(True)
class Solver(object):
    def __init__(self, learning_rate=0.0001, n_epochs=128):
        self.train_loader, self.test_loader = self.create_loaders()
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.n_batches_in_epoch = len(self.train_loader)

    def build(self):
        self.model = RFN(batch_size=batch_size).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=4, factor = 0.5)

    def create_loaders(self):
        train_loader=DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last = True)
        test_loader=DataLoader(testset, batch_size=batch_size, shuffle=True, drop_last = True)
        return train_loader, test_loader
    
    def uniform_binning_correction(self, x, n_bits=8):
        b, t, c, h, w = x.size()
        n_bins = 2 ** n_bits
        chw = c * h * w
        x_noise = x + torch.zeros_like(x).uniform_(0, 1.0 / n_bins)
        objective = -np.log(n_bins) * chw * torch.ones(b, device=x.device)
        return x_noise, objective

    def preprocess(self, x, reverse=False, range="0.5", n_bits = 8):

        n_bins = 2 ** n_bits
        if range == "0.5":
          if reverse == False:
            x = x * 255 # scale to 255
            if n_bits < 8:
              x = torch.floor( x/2 ** (8 - n_bits))
            x = x / n_bins  - 0.5
          else:
            x = torch.clamp(x, -0.5, 0.5)
            x = x + 0.5
            x = x * n_bins
            x = torch.clamp(x * (255 / n_bins), 0, 255).byte()
        elif range == "1.0":
          if reverse == False:
            x = x * 255 # scale to 255
            if n_bits < 8:
              x = torch.floor( x/2 ** (8 - n_bits))
            x = x / n_bins  
          else:
            x = torch.clamp(x, 0, 1)
            x = x * n_bins
            x = torch.clamp(x * (255 / n_bins), 0, 255).byte()
        return x

    def train(self):
        self.losses = []
        train_losses = []
        val_losses = []
        self.kl_loss = []
        self.recon_loss = []
        
        # TODO: Scheduler
        counter = 0
        iterations = 2000

        for epoch_i in range(self.n_epochs):
            epoch_i += 1
            self.batch_loss_history = []
            for batch_i, image in enumerate(self.train_loader):
                image = image.to(device)

                image = self.preprocess(image, n_bits = n_bits)
                image, logdet = self.uniform_binning_correction(image, n_bits = n_bits)
                self.model.beta = min(0.01, 0.0001 + (counter+1)/iterations)
                loss = self.model.loss(image, logdet.to(device))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (batch_i+1) % 200 == 0:
                  self.plotter()     
                self.batch_loss_history.append(float(loss.data))
                self.losses.append(loss.data)
                self.kl_loss.append(self.model.book['kl'])
                self.recon_loss.append(self.model.book['nll'])
                counter += 1
                
            epoch_loss = np.mean(self.batch_loss_history)
            self.scheduler.step(loss)
            print(f'Epoch {epoch_i} Loss: {epoch_loss:.2f}')
            train_losses.append(epoch_loss)
        return train_losses

    def plotter(self):
      fig, ax = plt.subplots(1, 3 , figsize = (10,3))
      ax[0].plot(self.losses)
      ax[1].plot(self.kl_loss)
      ax[2].plot(self.recon_loss)
      print("kl: {} and {}".format(str(self.kl_loss[-1]), str(self.recon_loss[-1])))
      with torch.no_grad():
        self.model.eval()
        image = next(iter(self.test_loader)).to(device)
        time_steps = 5
        n_predictions = 6
        image  = self.preprocess(image, reverse=False, n_bits = n_bits)
        #image, _ = self.uniform_binning_correction(image, n_bits = n_bits)
        samples, samples_recon, predictions = self.model.sample(image, n_predictions = n_predictions)
        samples  = self.preprocess(samples, reverse=True, n_bits = n_bits)
        samples_recon  = self.preprocess(samples_recon, reverse=True, n_bits = n_bits)
        predictions  = self.preprocess(predictions, reverse=True, n_bits = n_bits)
        image  = self.preprocess(image, reverse=True, n_bits = n_bits)
        self.model.train()

      fig, ax = plt.subplots(1, 5 , figsize = (20,5))
      for i in range(0, 5):
        ax[i].imshow(samples[0, i, :, :, :].view(-1, 32).detach().cpu().numpy())
        ax[i].set_title("Random Sample")

      fig, ax = plt.subplots(1, time_steps , figsize = (20,5))
      for i in range(0, time_steps):
        ax[i].imshow(samples[i, 0, :, :, :].view(-1, 32).detach().cpu().numpy())
        ax[i].set_title("Sample at timestep t")
      
      fig, ax = plt.subplots(1, time_steps , figsize = (20,5))
      for i in range(0, time_steps):
        ax[i].imshow(image[0, i, :, :, :].view(-1, 32).detach().cpu().numpy())
        ax[i].set_title("True Image")

      fig, ax = plt.subplots(1, time_steps , figsize = (20,5))
      for i in range(0, time_steps):
        ax[i].imshow(samples_recon[i, 0, :, :, :].view(-1, 32).detach().cpu().numpy())
        ax[i].set_title("Reconstructed Image")

      fig, ax = plt.subplots(1, n_predictions , figsize = (20,5))
      for i in range(0, n_predictions):
        ax[i].imshow(predictions[i, 0, :, :, :].view(-1, 32).detach().cpu().numpy())
        ax[i].set_title("Prediction")
      plt.show()


solver = Solver(n_epochs=60, learning_rate=0.001)
solver.build()
train = solver.train()
