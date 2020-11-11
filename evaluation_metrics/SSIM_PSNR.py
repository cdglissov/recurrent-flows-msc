
import torch 
#!pip install pytorch-msssim
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM



def PSNRbatch( X, Y, n_bits=8):
  bs, cs, h, w = X.shape
  maxi = 2**n_bits-1
  MSB = 1 / (cs * h * w) * torch.sum( (X - Y)**2, [1, 2, 3]) # Perbatch
  PSNR = 10 * torch.log10(maxi / MSB).mean()
  
  return PSNR

def ssim_val(X,Y,n_bits):
  data_range = 2**n_bits-1
  ssim_val = ssim( X, Y, data_range=data_range)
  
  return ssim_val
