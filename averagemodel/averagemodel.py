#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run this model in evalution environment! Is very small model so it is easy to 
load it.
"""

import torch

import torch.nn as nn
from data_generators import stochasticMovingMnist
from torch.utils.data import DataLoader
import numpy as np
from pytorch_msssim import ssim
from tqdm import tqdm 
from utils import set_gpu
import matplotlib.pyplot as plt
import os
from data_generators import bair_push
device = set_gpu(True)

seq_len_train = 6
seq_len_test = 20
# Takes for ever to compile the batches in bair
trainset_str = 'bair'
if trainset_str == 'mnist':
    trainset = stochasticMovingMnist.MovingMNIST(True, 'Mnist', 
                                                              seq_len = seq_len_train, 
                                                              image_size=64, 
                                                              digit_size=28, 
                                                              num_digits=2,
                                                              deterministic=False, 
                                                              three_channels=False, 
                                                              step_length=4, 
                                                              normalize=False)
    testset = stochasticMovingMnist.MovingMNIST(False, 'Mnist', 
                                                              seq_len = seq_len_test, 
                                                              image_size=64, 
                                                              digit_size=28, 
                                                              num_digits=2,
                                                              deterministic=False, 
                                                              three_channels=False, 
                                                              step_length=4, 
                                                              normalize=False)
if trainset_str == 'bair':
    string = str(os.path.abspath(os.getcwd()))
    trainset = bair_push.PushDataset(split='train',
                                              dataset_dir=string+'/bair_robot_data/processed_data/',
                                              seq_len=seq_len_train)
    testset = bair_push.PushDataset(split='test',
                                 dataset_dir=string+'/bair_robot_data/processed_data/',
                                 seq_len=seq_len_test)

train_loader = DataLoader(trainset,batch_size=128*2,num_workers=10, shuffle=True, drop_last=True)
test_loader = DataLoader(testset,batch_size=128*2,num_workers=10, shuffle=True, drop_last=True)
class SimpleLinearModel(nn.Module):
    def __init__(self, num_cond_frames = 9):
        super(SimpleLinearModel, self).__init__()
        num_features = int(num_cond_frames + (num_cond_frames-1) * (num_cond_frames) / 2 ) 
        self.W = nn.Parameter(torch.zeros(1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, 1))
        self.loss = nn.MSELoss()
        self.c = 3
        self.num_cond_frames = num_cond_frames
    def get_lagged_differences(self,x):
        laggeddif = []
        length = x.shape[1]
        for k in range(0,(x.shape[1] -1)):
            diff = x[:,0:(length-k -1 )]-x[:,(k+1):(length)]
            laggeddif.append(diff)
        return torch.cat(laggeddif,dim = 1)
    def get_prediction(self,x_use):
        diffs = self.get_lagged_differences(x_use)
        
        features = torch.cat((x_use,diffs),dim=1)

        features = features.view(*features.shape[0:2],-1)
        prediction = torch.matmul(self.W, features) + self.bias

        prediction = prediction.view(x_use.shape[0],self.c,*x_use.shape[3:5])
        return prediction
    def forward(self,x):
        x_use = x[:,0:self.num_cond_frames]
        x_true = x[:,self.num_cond_frames]
        
        prediction = self.get_prediction(x_use)
        
        loss = self.loss(prediction,x_true)

        return loss
    def ssim_val(self, X,Y,n_bits=8):
      # Is communative so [X,Y]=0
      # Or like it does not matter which one.. :P
      data_range = 2**n_bits-1
      ssim_val = ssim( X, Y, data_range=data_range)
      return ssim_val
    def PSNRbatch(self, X, Y, n_bits=8):
      # Is communative so [X,Y]=0
      bs, cs, h, w = X.shape
      maxi = 2**n_bits-1
      MSB = torch.mean( (X - Y)**2, dim = [1, 2, 3]) # Perbatch
      PSNR = 10 * torch.log10(maxi**2 / MSB).mean()
      
      return PSNR
    def sample(self,x, num_predictions):
        x_use = x[:,0:self.num_cond_frames]
        predictions = []
        
        for t in range(0,num_predictions):
            prediction = self.get_prediction(x_use)
            x_use = torch.cat((x_use,prediction.unsqueeze(1)), dim = 1)[:,1:]
            predictions.append(prediction.unsqueeze(1).detach())
        return torch.cat(predictions,dim = 1)
# It is conditioned on seq_len-1 as the models' loss is calculated from one-step predictions.
model = SimpleLinearModel(num_cond_frames = seq_len_train - 1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
                                                                    patience=10, 
                                                                    factor=0.2, 
                                                                    min_lr=0.000001)
losses = []
for epoch_i in range(3): # I does not need more than three epochs. 
    loss_epoch = []
    for batch_i, image in enumerate(tqdm(train_loader, desc="Epoch " + str(epoch_i), position=0, leave=True)):
        if trainset_str=='bair':
            image = image[0].to(device)
        else:
            image = image.to(device)
        loss = model(image)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_epoch.append(float(loss.data))
    with torch.no_grad():
        losses.append(loss_epoch)
        plt.figure()
        plt.plot(np.array(np.ravel(losses)))
        plt.savefig('/work1/s144077/testermeget/losses' + '.png', bbox_inches='tight')
        plt.close()
        xpredicted = model.sample(image,num_predictions = 10)
        x_true = image[:,-1]
        fig, ax = plt.subplots(1, 3 , figsize = (10,20))
        ax[0].imshow(xpredicted[0,0,:,:,:].permute(1,2,0).squeeze().detach().cpu().numpy())
        ax[0].set_title('Prediction')
        ax[1].imshow(x_true[0,:,:,:].permute(1,2,0).squeeze().detach().cpu().numpy())
        ax[1].set_title('True')
        ax[2].imshow(torch.abs(x_true[0,:,:,:]-xpredicted[0,0,:,:,:]).permute(1,2,0).squeeze().detach().cpu().numpy())
        ax[2].set_title('Difference')
        plt.savefig('/work1/s144077/testermeget/figures' +str(epoch_i)+ '.png', bbox_inches='tight')
        plt.close()
        fig, ax = plt.subplots(1, xpredicted.shape[1]-1 , figsize = (10*10,20))
        for k in range(0,xpredicted.shape[1]-1):
        	ax[k].imshow(xpredicted[0,k,:,:,:].permute(1,2,0).squeeze().detach().cpu().numpy())
        	ax[k].set_title(str(k+1)+'-Predict')
        	  
        plt.savefig('/work1/s144077/testermeget/predictions' +str(epoch_i)+ '.png', bbox_inches='tight')
        plt.close()

num_predictions = 10
# Training Evaluations
PSNR_values = []
SSIM_values = []
with torch.no_grad():
    model.eval()

    for batch_i, image in enumerate(tqdm(test_loader, desc="Tester", position=0, leave=True)):
            SSIM_values_batch = []
            PSNR_values_batch = []
            if trainset_str=='bair':
                image = image[0].to(device)
            else:
                image = image.to(device)
            xpredicted = model.sample(image,num_predictions = num_predictions)
            xpredicted = xpredicted * 255
            image = image * 255
            xpredicted = torch.clamp(xpredicted,0,255)
            image = torch.clamp(image,0,255)
            trueimage_predict = image[:,model.num_cond_frames:(model.num_cond_frames + num_predictions )]
            for i in range(0, xpredicted.shape[1]):
                #SSIM_values_batch = []

              PSNR_values_batch.append(model.PSNRbatch(xpredicted[:,i],trueimage_predict[:, i]))
              SSIM_values_batch.append(model.ssim_val(xpredicted[:,i],trueimage_predict[:, i]))
            PSNR_values_batch = torch.stack(PSNR_values_batch, dim = 0)
            SSIM_values_batch = torch.stack(SSIM_values_batch, dim = 0)

            PSNR_values.append(PSNR_values_batch)
            SSIM_values.append(SSIM_values_batch)
    PSNR_values = torch.stack(PSNR_values, dim = 0)
    SSIM_values = torch.stack(SSIM_values, dim = 0)

# This is then meaned over all the batches.. 
Savedict = {
  "SSIM_values": SSIM_values,
  "PSNR_values": PSNR_values,
  "SSIM_values_mean": SSIM_values.mean(0),  # We dont need to save this, but w.e.
  "PSNR_values_mean": PSNR_values.mean(0)
}
torch.save(Savedict,'SSIM_PSNR_bair_averagemodel_trained_on_6_frames.pt')
#print(SSIM_values)
#print(PSNR_values)
#print(PSNR_values.mean(0)) 
#print(SSIM_values.mean(0))        
#print(PSNR_values.shape)
