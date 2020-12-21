
import torch 
#!pip install pytorch-msssim
from pytorch_msssim import ssim
import torch.utils.data
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm 

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/work1/s144077/deepflows/')


from Utils import set_gpu
device = set_gpu(True)

import matplotlib.pyplot as plt
from RFN.trainer import Solver


from data_generators import MovingMNIST
from data_generators import PushDataset
import os
from skimage.metrics import peak_signal_noise_ratio as skPSNR
from skimage.measure import compare_ssim as skSSIM
import numpy as np

### Load model
  
# RFN
# Path to model



class Evaluator(object):
    def __init__(self,solver, args):
        self.args = args
        self.params = args
        self.n_bits = args.n_bits
        self.solver = solver

        self.verbose = args.verbose
        
        self.path = str(os.path.abspath(os.getcwd())) + args.path

        self.batch_size = args.batch_size

        
        self.choose_data = args.choose_data
        #self.n_frames = args.n_frames
        self.digit_size = args.digit_size
        self.step_length = args.step_length
        self.num_digits = args.num_digits
        self.image_size = args.image_size
        self.preprocess_range = args.preprocess_range
        self.preprocess_scale = args.preprocess_scale
        self.num_workers=args.num_workers
        self.multigpu = args.multigpu

    def build(self):
        self.test_loader = self.create_loaders()
        
        if not os.path.exists(self.path + 'png_folder'):
          os.makedirs(self.path + 'png_folder')
        if not os.path.exists(self.path + 'model_folder'):
          os.makedirs(self.path + 'model_folder')
        
        if self.multigpu and torch.cuda.device_count() > 1:
            print("Using:", torch.cuda.device_count(), "GPUs")
            self.model = self.solver.model.to(device)
        else:
            self.model = self.solver.model.to(device)
        self.model.eval()
    def create_loaders(self):
        self.n_frames = 30
        if self.choose_data=='mnist':
            	testset = MovingMNIST(False, 'Mnist', 
                                                         seq_len=self.n_frames, 
                                                         image_size=self.image_size, 
                                                         digit_size=self.digit_size, 
                                                         num_digits=self.num_digits, 
            												 deterministic=False, 
                                                         three_channels=False, 
                                                         step_length=self.step_length, 
                                                         normalize=False)
        if self.choose_data=='bair':
            	string=str(os.path.abspath(os.getcwd()))
            	testset = PushDataset(split='test',
                                             dataset_dir=string+'/bair_robot_data/processed_data/',
                                             seq_len=self.n_frames)

        test_loader = DataLoader(testset, batch_size=self.batch_size,num_workers=self.num_workers, shuffle=True, drop_last=True)
        return test_loader    
#    def preprocess(self, x, reverse = False):
#        # Remember to change the scale parameter to make variable between 0..255
#        preprocess_range = self.preprocess_range
#        scale = self.preprocess_scale
#        n_bits = self.n_bits
#        n_bins = 2 ** n_bits
#        if preprocess_range == "0.5":
#          if reverse == False:
#            x = x * scale
#            if n_bits < 8:
#              x = torch.floor( x/2 ** (8 - n_bits))
#            x = x / n_bins  - 0.5
#          else:
#            #x = torch.clamp(x, -0.5, 0.5)
#            x = x + 0.5
#            x = x * n_bins
#            x = torch.clamp(x * (255 / n_bins), 0, 255).byte()
#            x = x * 255
#        elif preprocess_range == "1.0":
#          if reverse == False:
#            x = x * scale
#            if n_bits < 8:
#              x = torch.floor( x/2 ** (8 - n_bits))
#            x = x / n_bins  
#          else:
#            x = torch.clamp(x, 0, 1)
#            x = x * n_bins
#            x = torch.clamp(x * (255 / n_bins), 0, 255).byte()
#        return x
    
    def Evalplotter(self, predictions,true_predicted):
      num_samples = 20 # The number of examples plotted
      time_steps = predictions.shape[1]
      fig, ax = plt.subplots(num_samples*2, time_steps , figsize = (time_steps,5*num_samples))
      for k in range(0,num_samples):
          for i in range(0, time_steps):
            ax[2*(k),i].imshow(self.convert_to_numpy(true_predicted[k, i, :, :, :]))
            ax[2*(k),i].set_title("True Image")
            ax[2*(k)+1,i].imshow(self.convert_to_numpy(predictions[k, i, :, :, :]))
            ax[2*(k)+1,i].set_title(str(i)+"-step Prediction")
      fig.savefig(self.path +'png_folder/eval_samples_v2' +  '.png', bbox_inches='tight')
      plt.close(fig)

    def convert_to_numpy(self, x):
        return x.permute(1,2,0).squeeze().detach().cpu().numpy()
    def PSNRbatch(self, X, Y, n_bits=8):
      # Is communative so [X,Y]=0
      bs, cs, h, w = X.shape
      maxi = 2**n_bits-1
      MSB = torch.mean( (X - Y)**2, dim = [1, 2, 3]) # size batch
      PSNR = 10 * torch.log10(maxi**2 / MSB).mean()
      
      return PSNR
    def mse_metric(self, x1, x2):
        err = np.sum((x1 - x2) ** 2)
        err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
        return err
    # Eval From SVG
    def eval_seq(self, gt, pred):
        # Takes a gt of size [bs, time, c,h,w]
        T = gt.shape[1]
        bs = gt.shape[0]
        ssim = torch.zeros((bs, T))
        psnr = torch.zeros((bs, T))
        mse = torch.zeros((bs, T))
        for i in range(bs):
            for t in range(T):
                for c in range(gt.shape[2]):
                    image_gen = np.uint8(gt[i,t,c,:,:].cpu().numpy())
                    image_true = np.uint8(pred[i, t, c,:, :].cpu().numpy())
                    ssim[i, t] += skSSIM(image_gen, image_true)
                    psnr[i, t] += skPSNR(image_gen, image_true)                    
                ssim[i, t] /= gt.shape[2] ## Num channels
                psnr[i, t] /= gt.shape[2] 
                mse[i, t] = torch.mean( (gt[i, t, :, : ,:] - pred[i, t, :, : ,:])**2, dim = [0, 1, 2])

        return mse, ssim, psnr
    
    def ssim_val(self, X,Y,n_bits=8):
      # Is communative so [X,Y]=0
      # Or like it does not matter which one.. :P
      data_range = 2**n_bits-1
      ssim_val = ssim( X, Y, data_range=data_range)
      return ssim_val
  
    def EvaluatorPSNR_SSIM(self):
      start_predictions = 6 # After how many frames the the models starts condiitioning on itself.
      times = 1 # This is to loop over the test set more than once, if you want to have than one.
      SSIM_values = []
      PSNR_values = []
      SSIM_values_sklearn = []
      PSNR_values_sklearn = []
      MSE_values_sklearn = []
      with torch.no_grad():
          for time in range(0, times):
              for batch_i, image in enumerate(tqdm(self.test_loader, desc="Tester", position=0, leave=True)):
                SSIM_values_batch = []
                PSNR_values_batch = []
                batch_i += 1
                if self.choose_data=='bair':
                    image_unprocessed = image[0].to(device)
                else:
                    image_unprocessed = image.to(device)
                image = self.solver.preprocess(image_unprocessed)
                samples, samples_recon, predictions = self.model.sample(image, n_predictions=self.n_frames-start_predictions, encoder_sample = False, start_predictions = start_predictions)
                #print(predictions[0,0,0,:,0])
                #print(predictions.permute(1,0,2,3,4).reshape((samples_recon.shape[1],-1)).min(dim=1)[0])
                image  = self.solver.preprocess(image, reverse=True)
                predictions  = self.solver.preprocess(predictions, reverse=True)
                true_predicted = image[:,start_predictions:,:,:,:].type(torch.FloatTensor).to(device)
                predictions = predictions.permute(1,0,2,3,4).type(torch.FloatTensor).to(device)
                mse, ssim, psnr = self.eval_seq(predictions, true_predicted)
                SSIM_values_sklearn.append(ssim)
                PSNR_values_sklearn.append(psnr)
                MSE_values_sklearn.append(mse)
                for i in range(0,predictions.shape[1]):
                    SSIM_values_batch.append(self.ssim_val(predictions[:,i,:,:,:], true_predicted[:,i,:,:,:]))
                    PSNR_values_batch.append(self.PSNRbatch(predictions[:,i,:,:,:], true_predicted[:,i,:,:,:]))
                SSIM_values_batch = torch.stack(SSIM_values_batch, dim = 0)
                PSNR_values_batch = torch.stack(PSNR_values_batch, dim = 0)
                SSIM_values.append(SSIM_values_batch)
                PSNR_values.append(PSNR_values_batch)
                
      SSIM_values = torch.stack(SSIM_values)
      PSNR_values = torch.stack(PSNR_values)
      MSE_values_sklearn = torch.stack(MSE_values_sklearn, dim = 1)
      PSNR_values_sklearn = torch.stack(PSNR_values_sklearn, dim = 1)
      SSIM_values_sklearn = torch.stack(SSIM_values_sklearn, dim = 1)
      bs,times, seq = MSE_values_sklearn.shape
      MSE_values_sklearn = MSE_values_sklearn.view(bs*times,seq) #torch.stack(MSE_values_sklearn, dim = 1)
      PSNR_values_sklearn = PSNR_values_sklearn.view(bs*times,seq)
      SSIM_values_sklearn = SSIM_values_sklearn.view(bs*times,seq)
      
      # Plot some samples to make sure the loader works. And the eval
      self.Evalplotter(predictions,true_predicted)
      
      return SSIM_values, PSNR_values, MSE_values_sklearn, PSNR_values_sklearn, SSIM_values_sklearn
                
namelist = ['conv','pool','squeeze']

for i in range(0,len(namelist)):
	pathcd ='/work1/s144077/architechturetest/'
	pathmodel = pathcd+namelist[i]+'/model_folder/rfn.pt'
	patherrormeasure = pathcd+namelist[i]+'/PSNR_SSIM.pt'
	#name_string = 'errormeasures_architecture_'+namelist[i]+'_trained_on_6_frames.pt'
	load_model = torch.load(pathmodel)
	args = load_model['args']
	solver = Solver(args)
	solver.build()
	solver.load(load_model)               
				   
					
	MetricEvaluator = Evaluator(solver, args)
	MetricEvaluator.build()

	SSIM_values, PSNR_values, MSE_values_sklearn, PSNR_values_sklearn, SSIM_values_sklearn = MetricEvaluator.EvaluatorPSNR_SSIM()
	Savedict = {
	  "SSIM_values": SSIM_values.cpu(),
	  "PSNR_values": PSNR_values.cpu(),
      "SSIM_values_sklearn": SSIM_values_sklearn.cpu(),
	  "PSNR_values_sklearn": PSNR_values_sklearn.cpu(),
      "MSE_values_sklearn": MSE_values_sklearn.cpu(),
	  "SSIM_values_mean": SSIM_values.mean(0).cpu(),  # We dont need to save this, but w.e.
	  "PSNR_values_mean": PSNR_values.mean(0).cpu()
	}
	torch.save(Savedict,patherrormeasure)    
	print(SSIM_values.mean(0))
	print(PSNR_values.mean(0))
	print(MSE_values_sklearn.mean(0))
	print(PSNR_values_sklearn.mean(0))
	print(SSIM_values_sklearn.mean(0))
       
            
