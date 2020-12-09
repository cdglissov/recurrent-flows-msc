
import torch 
#!pip install pytorch-msssim
from pytorch_msssim import ssim
import torch.utils.data
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm 

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/work1/s144077/deepflows/')


from utils import set_gpu
device = set_gpu(True)

from trainer import Solver

from data_generators import stochasticMovingMnist
from data_generators import bair_push
import os




### Load model
  
# RFN
# Path to model
path = '/work1/s144077/bairL4_64_128_skip_conditional_squeeze_batchnorm_v4/model_folder/rfn.pt'
name_string = 'errormeasures_bair_something_trained_on_6_frames.pt'
load_model = torch.load(path)
args = load_model['args']
solver = Solver(args)
solver.build()
solver.load()


class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.params = args
        self.n_bits = args.n_bits

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
            self.model = solver.model
        else:
            self.model = solver.model
        self.model.eval()
    def create_loaders(self):
        self.n_frames = 20
        if self.choose_data=='mnist':
            	testset = stochasticMovingMnist.MovingMNIST(False, 'Mnist', 
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
            	testset = bair_push.PushDataset(split='test',
                                             dataset_dir=string+'/bair_robot_data/processed_data/',
                                             seq_len=self.n_frames)

        test_loader = DataLoader(testset, batch_size=self.batch_size,num_workers=self.num_workers, shuffle=True, drop_last=True)
        return test_loader    
    def preprocess(self, x, reverse = False):
        # Remember to change the scale parameter to make variable between 0..255
        preprocess_range = self.preprocess_range
        scale = self.preprocess_scale
        n_bits = self.n_bits
        n_bins = 2 ** n_bits
        if preprocess_range == "0.5":
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
        elif preprocess_range == "1.0":
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
    def PSNRbatch(self, X, Y, n_bits=8):
      # Is communative so [X,Y]=0
      bs, cs, h, w = X.shape
      maxi = 2**n_bits-1
      MSB = torch.mean( (X - Y)**2, dim = [1, 2, 3]) # Perbatch
      PSNR = 10 * torch.log10(maxi**2 / MSB).mean()
      
      return PSNR
    
    def ssim_val(self, X,Y,n_bits=8):
      # Is communative so [X,Y]=0
      # Or like it does not matter which one.. :P
      data_range = 2**n_bits-1
      ssim_val = ssim( X, Y, data_range=data_range)
      return ssim_val
  
    def EvaluatorPSNR_SSIM(self):
      start_predictions = 6 # After how many frames the the models starts condiitioning on itself.
      times = 1
      SSIM_values = []
      PSNR_values = []
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
                image = self.preprocess(image_unprocessed)
                samples, samples_recon, predictions = self.model.sample(image, n_predictions=self.n_frames-start_predictions, encoder_sample = False, start_predictions = start_predictions)
                #samples  = self.preprocess(samples, reverse=True)
                image  = self.preprocess(image, reverse=True)
                #samples_recon  = self.preprocess(samples_recon, reverse=True)
                predictions  = self.preprocess(predictions, reverse=True)
                true_predicted = image[:,start_predictions:,:,:,:].type(torch.FloatTensor).to(device)
                predictions = predictions.permute(1,0,2,3,4).type(torch.FloatTensor).to(device)
                for i in range(0,predictions.shape[1]):
                    SSIM_values_batch.append(self.ssim_val(predictions[:,i,:,:,:], true_predicted[:,i,:,:,:]))
                    PSNR_values_batch.append(self.PSNRbatch(predictions[:,i,:,:,:], true_predicted[:,i,:,:,:]))
                SSIM_values_batch = torch.stack(SSIM_values_batch, dim = 0)
                PSNR_values_batch = torch.stack(PSNR_values_batch, dim = 0)
                SSIM_values.append(SSIM_values_batch)
                PSNR_values.append(PSNR_values_batch)
      SSIM_values = torch.stack(SSIM_values)
      PSNR_values = torch.stack(PSNR_values)
      return SSIM_values, PSNR_values
                
                
MetricEvaluator = Evaluator(args)
MetricEvaluator.build()

SSIM_values, PSNR_values = MetricEvaluator.EvaluatorPSNR_SSIM()
Savedict = {
  "SSIM_values": SSIM_values,
  "PSNR_values": PSNR_values,
  "SSIM_values_mean": SSIM_values.mean(0),  # We dont need to save this, but w.e.
  "PSNR_values_mean": PSNR_values.mean(0)
}
torch.save(Savedict,name_string)    
print(SSIM_values)
print(PSNR_values)       
            
