#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 23:10:07 2021

@author: s144077
"""

import matplotlib.pyplot as plt
plt.rcParams['image.cmap']='gray'
import torch
import sys
import numpy as np
sys.path.insert(1, './deepflows_02_02_v2/')
plt.rcParams.update({'text.usetex': True})
from Utils import set_gpu
device = set_gpu(True)
# Adding deepflows to system path

from RFN.trainer import Solver
#from evaluation_metrics.error_metrics import Evaluator

def convert_to_numpy(x):
    return x.permute(1,2,0).squeeze().detach().cpu().numpy()


path1 = '/work1/s144077/RFNtest/final_models/KTH/model_folder/rfn.pt'
path2 = '/work1/s144077/content/model_folder/rfn_epoch_10.pt'
path3 = '/work1/s144077/content/model_folder/rfn_epoch_20.pt'
savepath = '/work1/s144077/content/png_folder/BPPcompare' + '.pdf'
# Compare three models
with torch.no_grad():
    load_model_1 = torch.load(path1)
    args_1 = load_model_1['args']
    solver_1 = Solver(args_1)
    solver_1.build()
    solver_1.load(load_model_1)
    
    model_1 = solver_1.model.to(device).eval()
    
    load_model_2 = torch.load(path2)
    args_2 = load_model_2['args']
    solver_2 = Solver(args_2)
    solver_2.build()
    solver_2.load(load_model_2)
    
    model_2 = solver_2.model.to(device).eval()
    
    load_model_3 = torch.load(path3)
    args_3 = load_model_3['args']
    solver_3 = Solver(args_3)
    solver_3.build()
    solver_3.load(load_model_3)
    
    model_3 = solver_3.model.to(device).eval()

    
    ## Get loaded values
    image_1 = next(iter(solver_1.train_loader))
    image_2 = next(iter(solver_2.train_loader))
    image_3 = next(iter(solver_3.train_loader))
    ## Need to match the batches properly, as the model might be trained with different batch size...
    batch_1 = image_1.shape[0]
    batch_2 = image_2.shape[0]
    batch_3 = image_3.shape[0]
    
    images = [image_1,image_2,image_3]
    batches = np.array([batch_1,batch_2,batch_3])
    maxbatch = np.max(batches)
    
    idxmaxbatch = np.where(batches == maxbatch)

    image = solver_1.preprocess(images[idxmaxbatch[0][0]])
    image = image.to(device)
    ## Make sure we have the same batch, to all the models 
    image_1 = image[0:batch_1,:,:,:,:]
    image_2 = image[0:batch_2,:,:,:,:]
    image_3 = image[0:batch_3,:,:,:,:]
    recons_1, recons_flow_1, averageKLDseq_1, averageNLLseq_1 = model_1.reconstruct_elbo_gap(image_1,sample = True)
    recons_2, recons_flow_2, averageKLDseq_2, averageNLLseq_2 = model_2.reconstruct_elbo_gap(image_2,sample = True)
    recons_3, recons_flow_3, averageKLDseq_3, averageNLLseq_3 = model_3.reconstruct_elbo_gap(image_3,sample = True)
    # Restrict the length of the image.
    t = 10
    image = image[:,0:t,:,:,:]

    dims = image.shape[2:]
    averageNLLseq_1 = averageNLLseq_1/(np.log(2.)*torch.prod(torch.tensor(dims)))
    averageNLLseq_2 = averageNLLseq_2/(np.log(2.)*torch.prod(torch.tensor(dims)))
    averageNLLseq_3 = averageNLLseq_3/(np.log(2.)*torch.prod(torch.tensor(dims)))
    recons_1  = solver_1.preprocess(recons_1, reverse=True)
    recons_2  = solver_2.preprocess(recons_2, reverse=True)
    recons_3  = solver_3.preprocess(recons_3, reverse=True)
    image  = solver_1.preprocess(image, reverse=True)
    time_steps = image.shape[1]
    fig, ax = plt.subplots(4, time_steps , figsize = (2*time_steps, 1.62*5))
    #fig, ax = plt.subplots(4, time_steps , figsize = (time_steps, 4))
    for i in range(0, time_steps):
        ax[0,i].imshow(convert_to_numpy(image[0, i, :, :, :]))
        ax[0,i].set_xticks([])
        ax[0,i].set_yticks([])
        for z, recons in zip(range(0,3),list([recons_2,recons_3, recons_1])):
            if i == 0:
                ax[1+z,i].axis('off')
            else:
                ax[1+z,i].imshow(convert_to_numpy(recons[0, i, 0, :, :, :]))
            #ax[1+z,i].set_title(str(zname))
            ax[1+z,i].set_xticks([])
            ax[1+z,i].set_yticks([])
    
    fontsize = 30
    rotation = 0
    labelpad = 60
    ax[0,0].set_ylabel(r'$GT$:',fontsize = fontsize, rotation = rotation, labelpad = 35)
    ax[1,1].set_ylabel(r'$BPP='+str(np.round(averageNLLseq_2[0, :t, 0].mean().numpy(),2)) + '$',fontsize = 25, rotation = rotation, labelpad = labelpad+20)
    ax[2,1].set_ylabel(r'$BPP='+str(np.round(averageNLLseq_3[0, :t, 0].mean().numpy(),2)) + '$',fontsize = 25, rotation = rotation, labelpad = labelpad+20)
    ax[3,1].set_ylabel(r'$BPP='+str(np.round(averageNLLseq_1[0, :t, 0].mean().numpy(),2))+ '$',fontsize = 25, rotation = rotation, labelpad = labelpad+20)
    #plt.tight_layout()
    plt.subplots_adjust(wspace=0.03, hspace=0.03)
    fig.savefig(savepath, bbox_inches='tight')
    plt.close(fig)
