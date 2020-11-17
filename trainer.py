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
from data_generators import bair_push
import matplotlib.pyplot as plt
from RFN import RFN
from utils import *

choose_data='mnist'
batch_size=64
n_frames = 6

if choose_data=='mnist':
	three_channels=False
	testset = stochasticMovingMnist.MovingMNIST(False, 'Mnist', seq_len=n_frames, image_size=32, digit_size=24, num_digits=1, 
												deterministic=False, three_channels=three_channels, step_length=2, normalize=False)
	trainset = stochasticMovingMnist.MovingMNIST(True, 'Mnist', seq_len=n_frames, image_size=32, digit_size=24, num_digits=1, 
												  deterministic=False, three_channels=three_channels, step_length=2, normalize=False)
if choose_data=='bair':
	string=str(os.path.abspath(os.getcwd()))
	trainset = bair_push.PushDataset(split='train',dataset_dir=string+'/bair_robot_data/processed_data/',seq_len=n_frames)
	testset = bair_push.PushDataset(split='test',dataset_dir=string+'/bair_robot_data/processed_data/',seq_len=n_frames)


device = set_gpu(True)

class EarlyStopping():
	def __init__(self, min_delta=0, patience=50, verbose=True):
		super(EarlyStopping, self).__init__()
		# Early stopping will terminate training early based on specified conditions
		# min_delta : float minimum change in monitored value to qualify as improvement.
		# patience: integer number of epochs to wait for improvment before terminating.
		self.min_delta = min_delta
		self.patience = patience
		self.wait = 0
		self.best_loss = 1e15
		self.verbose = verbose

	def step(self, epoch, loss):
		current_loss = loss
		if current_loss is None:
			pass
		else: 
			if (current_loss - self.best_loss) < -self.min_delta:
				self.best_loss = current_loss
				self.wait = 1
			else:
				if self.wait >= self.patience:
					self.stop_training = True
					if self.verbose:
					  print("STOP! Criterion met at epoch " % (self.epoch))
					return self.stop_training
				self.wait += 1
 
class Solver(object):
    def __init__(self, learning_rate=0.001, n_epochs=40, verbose=False):
        self.train_loader, self.test_loader = self.create_loaders()
        self.n_bits = 7
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.plot_counter = 0
        self.path = str(os.path.abspath(os.getcwd()))+'/content/'
        self.losses = []
        self.kl_loss = []
        self.recon_loss = []
        if not os.path.exists(self.path + 'png_folder'):
          os.makedirs(self.path + 'png_folder')
        if not os.path.exists(self.path + 'model_folder'):
          os.makedirs(self.path + 'model_folder')
        self.epoch_i = 0
        self.best_loss = 1e15
 
    def build(self):
        self.model = RFN(batch_size=batch_size).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=4, 
                                                                    factor=0.5, min_lr=self.learning_rate*0.001)
        self.earlystopping = EarlyStopping(min_delta = 0, patience = 50, verbose = self.verbose)
 
    def create_loaders(self):
        train_loader = DataLoader(trainset,batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, drop_last=True)
        return train_loader, test_loader
 
    def uniform_binning_correction(self, x, n_bits=8):
        b, t, c, h, w = x.size()
        n_bins = 2 ** n_bits
        chw = c * h * w
        x_noise = x + torch.zeros_like(x).uniform_(0, 1.0 / n_bins)
        objective = -np.log(n_bins) * chw * torch.ones(b, device=x.device)
        return x_noise, objective
 
    def preprocess(self, x, reverse = False, range = "0.5", n_bits = 8, scale = 255):
        # Remember to change the scale parameter to make variable between 0..255
        n_bins = 2 ** n_bits
        if range == "0.5":
          if reverse == False:
            x = x * scale
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
            x = x * scale
            if n_bits < 8:
              x = torch.floor( x/2 ** (8 - n_bits))
            x = x / n_bins  
          else:
            x = torch.clamp(x, 0, 1)
            x = x * n_bins
            x = torch.clamp(x * (255 / n_bins), 0, 255).byte()
        return x
 
    def train(self):
      counter = 0
      max_value = 0.01
      min_value = 0.001
      num_steps = 2000
      
      for epoch_i in range(self.n_epochs):
          self.model.train()
          self.epoch_i += 1
          self.batch_loss_history = []
          for batch_i, image in enumerate(self.train_loader):
            batch_i += 1
            if choose_data=='bair':
                image = image[0].to(device)
            else:
                image = image.to(device)
            image = self.preprocess(image, n_bits = self.n_bits)
            image, logdet = self.uniform_binning_correction(image, n_bits = self.n_bits)
            self.model.beta = min(max_value, min_value + counter*(max_value - min_value) / num_steps)
            loss = self.model.loss(image, logdet)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
 
            store_loss = float(loss.data)
            self.losses.append(store_loss)
            self.kl_loss.append(self.model.book['kl'])
            self.recon_loss.append(self.model.book['nll'])
            counter += 1
            
          self.plotter()   
          epoch_loss = np.mean(self.losses)
 
          if self.epoch_i % 25 == 0:
            # Save model after each 25 epochs
            self.checkpoint('rfn.pt', self.epoch_i, epoch_loss) 
 
          # Do early stopping, if above patience stop training, if better loss, save model
          stop = self.earlystopping.step(self.epoch_i, epoch_loss)
          if stop:
            break
          if (self.earlystopping.best_loss < self.best_loss) and (self.epoch_i > 50):
            self.best_loss = self.earlystopping.best_loss
            self.checkpoint('rfn_best_model.pt', self.epoch_i, epoch_loss) 
          self.scheduler.step(loss)
 
          if self.verbose:
            print(f'Epoch {self.epoch_i} Loss: {epoch_loss:.2f}')
          else:
            self.status()
 
    def checkpoint(self, model_name, epoch, loss):
      torch.save({
          'epoch': epoch,
          'model_state_dict': self.model.state_dict(),
          'optimizer_state_dict': self.optimizer.state_dict(),
          'loss': loss,}, self.path + 'model_folder/' + model_name)
 
    def load(self, path):
      load_model = torch.load(path)
      self.model.load_state_dict(load_model['model_state_dict'])
      self.optimizer.load_state_dict(load_model['optimizer_state_dict'])
      self.epoch_i += load_model['epoch']
      loss = load_model['loss']
      self.best_loss = loss
      return (epoch, loss)
 
    def status(self):
      # Only works for python 3.x
      lr = self.optimizer.param_groups[0]['lr']
      with open(self.path + 'model_folder/status.txt', 'a') as f:
        print("STATUS:", file=f)
        print("\tKL and Reconstruction loss: {:.4f}, {:.4f}".format(self.kl_loss[-1].data, self.recon_loss[-1].data), file=f)
        print(f'\tEpoch {self.epoch_i}, Beta value {self.model.beta}, Learning rate {lr}', file=f)
 
    def plotter(self):
      n_plot = str(self.plot_counter)
      with torch.no_grad():
        self.model.eval()
        image = next(iter(self.test_loader)).to(device)
        time_steps = 5
        n_predictions = time_steps
        image  = self.preprocess(image, reverse=False, n_bits = self.n_bits)
        samples, samples_recon, predictions = self.model.sample(image, n_predictions = n_predictions,encoder_sample = False)
        samples  = self.preprocess(samples, reverse=True, n_bits = self.n_bits)
        samples_recon  = self.preprocess(samples_recon, reverse=True, n_bits = self.n_bits)
        predictions  = self.preprocess(predictions, reverse=True, n_bits = self.n_bits)
        # With x
        samples_x, samples_recon_x, predictions_x = self.model.sample(image, n_predictions = n_predictions,encoder_sample = True)
        samples_x  = self.preprocess(samples_x, reverse=True, n_bits = self.n_bits)
        samples_recon_x  = self.preprocess(samples_recon_x, reverse=True, n_bits = self.n_bits)
        predictions_x  = self.preprocess(predictions_x, reverse=True, n_bits = self.n_bits)
        image  = self.preprocess(image, reverse=True, n_bits = self.n_bits)
        
 
      fig, ax = plt.subplots(1, 4 , figsize = (20,5))
      ax[0].plot(self.losses)
      ax[0].set_title("Loss")
      ax[0].grid()
      ax[1].plot(self.losses)
      ax[1].set_title("log-Loss")
      ax[1].set_yscale('log')
      ax[1].grid()
      ax[2].plot(self.kl_loss)
      ax[2].set_title("KL Loss")
      ax[2].grid()
      ax[3].plot(self.recon_loss)
      ax[3].set_title("Reconstruction Loss")
      ax[3].grid()
      if not self.verbose:
        fig.savefig(self.path + 'png_folder/losses' + '.png', bbox_inches='tight')
        plt.close(fig)
      
      fig, ax = plt.subplots(5, time_steps , figsize = (20,5*5))
      for i in range(0, time_steps):
        ax[0,i].imshow(samples[0, i, :, :, :].permute(1,2,0).squeeze().detach().cpu().numpy())
        ax[0,i].set_title("Random Sample")
        ax[1,i].imshow(samples[i, 0, :, :, :].permute(1,2,0).squeeze().detach().cpu().numpy())
        ax[1,i].set_title("Sample at timestep t")
        ax[2,i].imshow(image[0, i, :, :, :].permute(1,2,0).squeeze().detach().cpu().numpy())
        ax[2,i].set_title("True Image")
        ax[3,i].imshow(samples_recon[i, 0, :, :, :].permute(1,2,0).squeeze().detach().cpu().numpy())
        ax[3,i].set_title("Reconstructed Image")
        ax[4,i].imshow(predictions[i, 0, :, :, :].permute(1,2,0).squeeze().detach().cpu().numpy())
        ax[4,i].set_title("Prediction")
      if not self.verbose:
        fig.savefig(self.path +'png_folder/samples' + n_plot + '.png', bbox_inches='tight')
        plt.close(fig)

      fig, ax = plt.subplots(5, time_steps , figsize = (20,5*5))
      for i in range(0, time_steps):
        ax[0,i].imshow(samples_x[0, i, :, :, :].permute(1,2,0).squeeze().detach().cpu().numpy())
        ax[0,i].set_title("Random Sample")
        ax[1,i].imshow(samples_x[i, 0, :, :, :].permute(1,2,0).squeeze().detach().cpu().numpy())
        ax[1,i].set_title("Sample at timestep t")
        ax[2,i].imshow(image[0, i, :, :, :].permute(1,2,0).squeeze().detach().cpu().numpy())
        ax[2,i].set_title("True Image")
        ax[3,i].imshow(samples_recon_x[i, 0, :, :, :].permute(1,2,0).squeeze().detach().cpu().numpy())
        ax[3,i].set_title("Reconstructed Image")
        ax[4,i].imshow(predictions_x[i, 0, :, :, :].permute(1,2,0).squeeze().detach().cpu().numpy())
        ax[4,i].set_title("Prediction")
      if not self.verbose:
        fig.savefig(self.path +'png_folder/samples_with_x'+ n_plot + '.png', bbox_inches='tight')
        plt.close(fig)
      if self.verbose:
        print("\tKL and Reconstruction loss: {:.4f}, {:.4f}".format(self.kl_loss[-1].data, self.recon_loss[-1].data))
        plt.show()
 
      self.plot_counter += 1
      self.model.train()
 
 
solver = Solver(n_epochs=1000, learning_rate=0.0007)
solver.build()
# uncomment this if we want to load a model
#path_model = '/content/model_folder/rfn.pt'
#solver.load(path_model)
solver.train()   
