import matplotlib.pyplot as plt
import torch 
import numpy as np
# For the loss now
from scipy.signal import savgol_filter

namelist = ['tanh_no','tanhup_down','tanh_no_conv','tanhup_down_conv']
path = './tanhtest/'
fig, ax = plt.subplots(3, 2 ,figsize = (20,10)) #, figsize = (time_steps,5*num_samples))
for i in range(0,len(namelist)):
    name = namelist[i]
    plot_dict = torch.load(path+name+'/PSNR_SSIM_LPIPS.pt')
    SSIM  = plot_dict['SSIM_values_sklearn']
    PSNR  = plot_dict['PSNR_values_sklearn']
    LPIPS  = plot_dict['LPIPS_values']
    y = SSIM.mean(0).numpy()
    #print(np.shape(np.std(PSNR.numpy(),0)))
    #print()

    twostd = 1.96 * np.std(SSIM.numpy(),0)#/np.sqrt(np.shape(SSIM.numpy())[0])
    #print(twostd)
    ax[0,0].plot(np.arange(0,len(y)),y,label = name)
    alpha = 0.05
    
    ax[0,0].fill_between(np.arange(0,len(y)), y-twostd, y+twostd, alpha=.1)
    
    y = np.median(SSIM.numpy(),0)
    ax[0,1].plot(np.arange(0,len(y)),y,label = name)
    alpha = 0.05
    
    ax[0,1].fill_between(np.arange(0,len(y)), np.quantile(SSIM.numpy(),alpha/2,axis = 0), np.quantile(SSIM.numpy(),1-alpha/2,axis = 0), alpha=.1)
    

    
    
    
    y = PSNR.mean(0).numpy()
    twostd = 1.96 * np.std(PSNR.numpy(),0)#/np.sqrt(np.shape(PSNR.numpy())[0])
    ax[1,0].plot(np.arange(0,len(y)),y,label = name)
    #ax[1].fill_between(np.arange(0,len(y)), np.quantile(PSNR.numpy(),alpha/2,axis = 0), np.quantile(PSNR.numpy(),1-alpha/2,axis = 0), alpha=.1)
    ax[1,0].fill_between(np.arange(0,len(y)), y-twostd, y+twostd, alpha=.1)
    
    y = np.median(PSNR.numpy(),0)
    ax[1,1].plot(np.arange(0,len(y)),y,label = name)
    ax[1,1].fill_between(np.arange(0,len(y)), np.quantile(PSNR.numpy(),alpha/2,axis = 0), np.quantile(PSNR.numpy(),1-alpha/2,axis = 0), alpha=.1)
    
    
    y = LPIPS.mean(0).numpy()
    #print(np.shape(np.std(PSNR.numpy(),0)))
    #print()

    twostd = 1.96 * np.std(LPIPS.numpy(),0)#/np.sqrt(np.shape(LPIPS.numpy())[0])
    #print(twostd)
    ax[2,0].plot(np.arange(0,len(y)),y,label = name)
    alpha = 0.05
    
    ax[2,0].fill_between(np.arange(0,len(y)), y-twostd, y+twostd, alpha=.1)
    
    y = np.median(LPIPS.numpy(),0)
    ax[2,1].plot(np.arange(0,len(y)),y,label = name)
    alpha = 0.05
    
    ax[2,1].fill_between(np.arange(0,len(y)), np.quantile(LPIPS.numpy(),alpha/2,axis = 0), np.quantile(LPIPS.numpy(),1-alpha/2,axis = 0), alpha=.1)
    
    

ax[0,0].set_title('SSIM')
ax[0,0].legend()
ax[0,0].grid()
ax[0,1].set_title('SSIM-quantile-plot')
ax[0,1].legend()
ax[0,1].grid()
ax[1,0].set_title('PSNR')
ax[1,0].legend()
ax[1,0].grid()
ax[1,1].set_title('PSNR-quantile-plot')
ax[1,1].legend()
ax[1,1].grid()

ax[2,0].set_title('LPIPS')
ax[2,0].legend()
ax[2,0].grid()
ax[2,1].set_title('LPIPS-quantile-plot')
ax[2,1].legend()
ax[2,1].grid()
fig.savefig(path +  'SSIM_PSNR_LPIPS.png', bbox_inches='tight')  




# Check the different losses
#namelist = ['tanh_no','tanhup_down']
fig, ax = plt.subplots(3, 1)
fig2, ax2 = plt.subplots(3, 1)
for i in range(0,len(namelist)):
	name = namelist[i]
	pathcd = path #'/work1/s144077/tanhtest/'
	pathmodel = pathcd+namelist[i]+'/model_folder/eval_dict.pt'
    
	load_dict = torch.load(pathmodel)
	loss = load_dict['losses']
	kl_loss = load_dict['kl_loss']
	recon_loss = load_dict['recon_loss']
	ax[0].plot(loss, label = name)
	ax[1].plot(kl_loss, label = name)
	ax[2].plot(recon_loss, label = name)
	ax2[0].plot(savgol_filter(tuple(loss), 301, 3), label = name)
	ax2[1].plot(savgol_filter(tuple(kl_loss), 301, 3), label = name)
	ax2[2].plot(savgol_filter(tuple(recon_loss), 301, 3), label = name)
ylim1 = [654,655]
ax[0].set_title('Total loss')
ax[0].legend()
ax[0].set_ylim(ylim1)
ax[0].grid()
ax[1].set_title('KL loss')
ax[1].legend()
ax[1].grid()
ax[1].set_ylim([0, 0.2])
ax[2].set_title('Recon loss')
ax[2].legend()
ax[2].grid()
ax[2].set_ylim(ylim1)
fig.savefig(path +  'Losses.png', bbox_inches='tight')  
  
ax2[0].set_title('Total loss')
ax2[0].legend()
ax2[0].set_ylim(ylim1)
ax2[0].grid()
ax2[1].set_title('KL loss')
ax2[1].legend()
ax2[1].grid()
ax2[1].set_ylim([0, 0.2])
ax2[2].set_title('Recon loss')
ax2[2].legend()
ax2[2].grid()
ax2[2].set_ylim(ylim1)
fig2.savefig(path +  'SmoothLosses.png', bbox_inches='tight')  
    

