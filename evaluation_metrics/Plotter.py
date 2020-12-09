#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 22:29:03 2020

@author: s144077
"""

import matplotlib.pyplot as plt
import torch 
import numpy as np
RFN_dict = torch.load('/work1/s144077/errormeasures_bair_something_trained_on_6_frames.pt')
Average_dict = torch.load('/work1/s144077/SSIM_PSNR_bair_averagemodel_trained_on_6_frames.pt')

SSIM_1  = RFN_dict['SSIM_values']
SSIM_2  = Average_dict['SSIM_values']

PSNR_1  = RFN_dict['PSNR_values']
PSNR_2 = Average_dict['PSNR_values']

std_1 = torch.std(SSIM_1, dim=0).cpu().numpy() 
std_2 = torch.std(SSIM_2, dim=0).cpu().numpy()
plt.figure()
plt.errorbar(np.arange(len(std_1)), SSIM_1.mean(0).cpu().numpy(), yerr=std_1*1.96, label='RFN - SSIM')
plt.errorbar(np.arange(len(std_2)), SSIM_2.mean(0).cpu().numpy(), yerr=std_2*1.96, label='Average - SSIM')
plt.legend()

plt.savefig('/work1/s144077/testermeget/comparison_SSIM_bair.png', bbox_inches='tight')
plt.close()

std_1 = torch.std(PSNR_1, dim=0).cpu().numpy() 
std_2 = torch.std(PSNR_2, dim=0).cpu().numpy()
plt.figure()
plt.errorbar(np.arange(len(std_1)), PSNR_1.mean(0).cpu().numpy(), yerr=std_1*1.96, label='RFN - PSNR')
plt.errorbar(np.arange(len(std_2)), PSNR_2.mean(0).cpu().numpy(), yerr=std_2*1.96, label='Average - PSNR')
plt.legend()
plt.savefig('/work1/s144077/testermeget/comparison_PSNR_bair.png', bbox_inches='tight')
plt.close()
