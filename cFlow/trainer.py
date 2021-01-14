#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 20:29:27 2021

@author: s144077
"""

import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import os
from torch.utils.data import DataLoader


import matplotlib.pyplot as plt
from .cGlow import cGlow
from Utils import set_gpu
from tqdm import tqdm 
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
    def __init__(self, args):
        self.args = args
        self.params = args
        self.n_bits = args.n_bits
        self.n_epochs = args.n_epochs
        self.learning_rate = args.learning_rate
        self.verbose = args.verbose
        self.plot_counter = 0
        self.path = str(os.path.abspath(os.getcwd())) + args.path
        self.losses = []
        
        self.epoch_i = 0
        self.best_loss = 1e15
        self.batch_size = args.batch_size
        self.patience_lr = args.patience_lr
        self.factor_lr = args.factor_lr
        self.min_lr = args.min_lr
        self.patience_es = args.patience_es

        self.choose_data = args.choose_data
        
        self.image_size = args.image_size
        self.preprocess_range = args.preprocess_range
        self.preprocess_scale = args.preprocess_scale
        self.num_workers=args.num_workers
        self.multigpu = args.multigpu
        
    def build(self):
        self.train_loader, self.test_loader = self.create_loaders()
        
        if not os.path.exists(self.path + 'png_folder'):
          os.makedirs(self.path + 'png_folder')
        if not os.path.exists(self.path + 'model_folder'):
          os.makedirs(self.path + 'model_folder')
        
        if self.multigpu and torch.cuda.device_count() > 1:
            print("Using:", torch.cuda.device_count(), "GPUs")
            self.model = nn.DataParallel(cGlow(self.params)).to(device)
        else:
            self.model = cGlow(self.params).to(device)
        
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', 
                                                                    patience=self.patience_lr, 
                                                                    factor=self.factor_lr, 
                                                                    min_lr=self.min_lr)
        self.earlystopping = EarlyStopping(min_delta = 0, patience = self.patience_es, 
                                           verbose = self.verbose)
        self.counter = 0

    def get_joint_conditioned_data(self, data, box_size = (8, 24)):
        i, j = box_size
        x = data.clone()
        y = data.clone()
        h, w=data.shape[-2:]
        # Set the inner square to 0
        x[:, :, i:j, i:j] = 0
        # Set the outer square to 0
        get_ids = np.concatenate((np.arange(0, i, 1),np.arange(j, w, 1)))
        y[:, :, get_ids, :] = 0
        y[:, :, :, get_ids] = 0
        # x is rim, y is center
        return x, y
    def create_loaders(self):

        if self.choose_data=='celeba32':

            from data_generators import celeba
            # This is not good code, is very memory intensive.
            trainset, testset = celeba.get_celeba(plot_sample=False)
            
            
        train_loader = DataLoader(trainset,batch_size=self.batch_size,num_workers=self.num_workers, shuffle=True, drop_last=True)
        test_loader = DataLoader(testset, batch_size=self.batch_size,num_workers=self.num_workers, shuffle=True, drop_last=True)
        return train_loader, test_loader
 
    def uniform_binning_correction(self, x):
        n_bits = self.n_bits
        b, c, h, w = x.size()
        n_bins = 2 ** n_bits
        chw = c * h * w
        x_noise = x + torch.zeros_like(x).uniform_(0, 1.0 / n_bins)
        objective = -np.log(n_bins) * chw * torch.ones(b, device=x.device)
        return x_noise, objective
 
    def preprocess(self, x, reverse = False):
        # Remember to change the scale parameter to make variable between 0..255
        n_bits = self.n_bits
        n_bins = 2 ** n_bits
        if self.preprocess_range == "0.5":
          if reverse == False:
            x = x * self.preprocess_scale
            if n_bits < 8:
              x = torch.floor( x/2 ** (8 - n_bits))
            x = x / n_bins  - 0.5
          else:
            x = x + 0.5
            x = x * n_bins
            x = torch.clamp(x * (255 / n_bins), 0, 255).byte()
        elif self.preprocess_range == "1.0":
          if reverse == False:
            x = x * self.preprocess_scale
            if n_bits < 8:
              x = torch.floor( x/2 ** (8 - n_bits))
            x = x / n_bins  
          else:
            x = x * n_bins
            x = torch.clamp(x * (255 / n_bins), 0, 255).byte()
        return x
 
    def train(self):
      

      for epoch_i in range(self.n_epochs):
          self.model.train()
          self.epoch_i += 1
          self.batch_loss_history = []
          for batch_i, image in enumerate(tqdm(self.train_loader, desc="Epoch " + str(self.epoch_i), position=0, leave=False)):
            batch_i += 1
            image = image.to(device)
            if self.choose_data=='celeba32':
                image = image/3 # Get to range [0,1]
            image = self.preprocess(image)
            image, logdet = self.uniform_binning_correction(image)
            condition, image = self.get_joint_conditioned_data(image)
            logdet =  logdet.to(device)
            
            
            
            if self.multigpu and torch.cuda.device_count() > 1:
                nll_loss = self.model.module.loss(image, condition, logdet)
            else:
                nll_loss = self.model.loss(image, condition, logdet)
            loss= nll_loss
            self.optimizer.zero_grad()
            loss.backward()
            
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10, 2)
            #torch.nn.utils.clip_grad_value_(self.model.parameters(), 5)
            self.optimizer.step()
            dims = float(np.log(2.) * torch.prod(torch.tensor(image.shape[1:])))
            bits_per_dim_loss =  float(loss.data/dims)
            
            self.losses.append(bits_per_dim_loss)

            #if (batch_i % 5)==0:
          
          self.plotter()   
          epoch_loss = np.mean(self.losses)
 
          if self.epoch_i % 1 == 0:
            # Save model after each 25 epochs
            self.checkpoint('cGlow.pt', self.epoch_i, epoch_loss) 
 
          # Do early stopping, if above patience stop training, if better loss, save model
          stop = self.earlystopping.step(self.epoch_i, epoch_loss)
          if stop:
            break
          if (self.earlystopping.best_loss < self.best_loss) and (self.epoch_i > 50):
            self.best_loss = self.earlystopping.best_loss
            self.checkpoint('cGlow_best_model.pt', self.epoch_i, epoch_loss) 
          self.scheduler.step(loss)
          
          if self.verbose:
            print('Epoch {} Loss: {:.2f}'.format(self.epoch_i, epoch_loss))
          else:
            self.status()
 
    def checkpoint(self, model_name, epoch, loss):
      torch.save({
          'epoch': epoch,
          'model_state_dict': self.model.state_dict(),
          'optimizer_state_dict': self.optimizer.state_dict(),
          'loss': loss,
          'losses': self.losses,
          'plot_counter': self.plot_counter,
          'args': self.args,
          }, self.path + 'model_folder/' + model_name)
      torch.save({
          'epoch': epoch,
          'loss': loss,
          'losses': self.losses,
          'annealing_counter': self.counter,
          'args': self.args,
          }, self.path + 'model_folder/eval_dict.pt')
      
    def load(self, load_model):
      self.model.load_state_dict(load_model['model_state_dict'])
      self.optimizer.load_state_dict(load_model['optimizer_state_dict'])
      self.epoch_i += load_model['epoch']
      loss = load_model['loss']
      self.losses = load_model['losses']
      self.plot_counter = load_model['plot_counter']
      
      self.best_loss = loss
      self.model.to(device)
      return (self.epoch_i, loss)
 
    def status(self):
      # Only works for python 3.x
      lr = self.optimizer.param_groups[0]['lr']
      with open(self.path + 'model_folder/status.txt', 'a') as f:
        print("STATUS:", file=f)
        print("\t NLL loss: {:.4f}".format(self.losses[-1]), file=f)
        print(f'\tEpoch {self.epoch_i}, Learning rate {lr}', file=f)
 
    def plotter(self):
      n_plot = str(self.plot_counter)
      with torch.no_grad():
        self.model.eval()
        image = next(iter(self.test_loader))
        image = image.to(device)
        if self.choose_data=='celeba32':
            image = image/3 # Get to range [0,1]
        imagetrue = image 
        image = self.preprocess(image)
        condition, image = self.get_joint_conditioned_data(image)

        
        n_samples = 2

        samples, samples_recon = self.model.sample(image, condition, n_samples=n_samples)
        samples  = self.preprocess(samples, reverse=True)
        samples_recon  = self.preprocess(samples_recon, reverse=True)

    
 
      plt.figure()
      plt.plot(self.losses)
      plt.title("Loss")
      plt.grid()

      if not self.verbose:
        plt.savefig(self.path + 'png_folder/losses' + '.png', bbox_inches='tight')
        plt.close()
      
      fig, ax = plt.subplots(2*n_samples+1, 10 , figsize = ((2*n_samples+1)*2,10))
      
      for i in range(0, 10):
        ax[0,i].imshow(self.convert_to_numpy(imagetrue[i, :, :, :]))
        ax[0,i].set_title("True")
        for k in range(0,n_samples):

            #combined_samples = self.join_condition_and_samples(imagetrue[i, :, :,:], samples[k, i,:,:,:])
            ax[1+k,i].imshow(self.convert_to_numpy(samples[k, i,:,:,:]))
            ax[1+k,i].set_title("Sample")
            #combined_recon = self.join_condition_and_samples(imagetrue[i, :, :,:], samples_recon[k, i, :, :, :])
            ax[2+k+n_samples-1,i].imshow(self.convert_to_numpy(samples_recon[k, i, :, :, :]))
            ax[2+k+n_samples-1,i].set_title("Recon")

            
      if not self.verbose:
        fig.savefig(self.path +'png_folder/samples' + n_plot + '.png', bbox_inches='tight')
        plt.close(fig)
      
     
      if self.verbose:
        plt.show()
 
      self.plot_counter += 1
      self.model.train()
    def join_condition_and_samples(self,condition,samples):
        condition = condition.clone()
        samples = samples.clone()
        condition[:, 8:24,8:24] = samples[:,8:24,8:24]
        return condition
    def convert_to_numpy(self, x):
        return x.permute(1,2,0).squeeze().detach().cpu().numpy()

