'''Train script file'''
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from data_generators import MovingMNIST
from data_generators import PushDataset
import matplotlib.pyplot as plt
from .SVG import SVG
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
        self.params = args
        self.n_bits = args.n_bits
        self.n_epochs = args.n_epochs
        self.learning_rate = args.learning_rate
        self.verbose = args.verbose
        self.plot_counter = 0
        self.path = str(os.path.abspath(os.getcwd())) + args.path
        self.losses = []
        self.kl_loss = []
        self.recon_loss = []
        self.epoch_i = 0
        self.best_loss = 1e15
        self.batch_size = args.batch_size
        self.patience_es = args.patience_es
        self.beta_max = args.beta_max
        self.beta_min = args.beta_min
        self.beta_steps = args.beta_steps
        self.choose_data = args.choose_data
        self.n_frames = args.n_frames
        self.digit_size = args.digit_size
        self.step_length = args.step_length
        self.num_digits = args.num_digits
        self.image_size = args.image_size
        self.preprocess_range = args.preprocess_range
        self.preprocess_scale = args.preprocess_scale
        self.multigpu = args.multigpu
        self.variable_beta = args.variable_beta
        self.counter = 0
        
    def build(self):
        self.train_loader, self.test_loader = self.create_loaders()
        
        if not os.path.exists(self.path + 'png_folder'):
          os.makedirs(self.path + 'png_folder')
        if not os.path.exists(self.path + 'model_folder'):
          os.makedirs(self.path + 'model_folder')
        
        if self.multigpu and torch.cuda.device_count() > 1:
            print("Using:", torch.cuda.device_count(), "GPUs")
            self.model = nn.DataParallel(SVG(self.params)).to(device)
        else:
            self.model = SVG(self.params).to(device)
        
        
        self.earlystopping = EarlyStopping(min_delta = 0, patience = self.patience_es, 
                                           verbose = self.verbose)
        
        
    def create_loaders(self):
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
            	trainset = MovingMNIST(True, 'Mnist', 
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
            	trainset = PushDataset(split='train',
                                              dataset_dir=string+'/bair_robot_data/processed_data/',
                                              seq_len=self.n_frames)
            	testset = PushDataset(split='test',
                                             dataset_dir=string+'/bair_robot_data/processed_data/',
                                             seq_len=self.n_frames)

        train_loader = DataLoader(trainset,batch_size=self.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(testset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        return train_loader, test_loader
 
    def uniform_binning_correction(self, x):
        n_bits = self.n_bits
        b, t, c, h, w = x.size()
        n_bins = 2 ** n_bits
        chw = c * h * w
        x_noise = x + torch.zeros_like(x).uniform_(0, 1.0 / n_bins)
        objective = -np.log(n_bins) * chw * torch.ones(b, device=x.device)
        return x_noise, objective
 
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
            x = x.div(x.max()) - 0.5 #torch.clamp(x, -0.5, 0.5)
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
 
    def train(self):
      max_value = self.beta_max
      min_value = self.beta_min
      num_steps = self.beta_steps
      
      for epoch_i in range(self.n_epochs):
          self.model.frame_predictor.train()
          self.model.posterior.train() 
          self.model.prior.train()
          self.model.encoder.train()
          self.model.decoder.train()
          self.model.flow.train()
          epoch_nll = 0
          epoch_kl = 0
          epoch_loss = 0
          self.epoch_i += 1
          for batch_i, image in enumerate(tqdm(self.train_loader, desc="Epoch " + str(self.epoch_i), position=0, leave=False)):
            
            batch_i += 1
            if self.choose_data=='bair':
                image = image[0].to(device)
            else:
                image = image.to(device)
            image = self.preprocess(image)
            image, logdet = self.uniform_binning_correction(image)

            if self.variable_beta: 
                self.model.beta = min(max_value, min_value + self.counter*(max_value - min_value) / num_steps)
            else:
                self.model.beta = max_value
            
            if self.multigpu and torch.cuda.device_count() > 1:
                kl, nll, loss = self.model.module.loss(image, logdet)
            else:
                kl, nll, loss = self.model.loss(image, logdet)
            epoch_nll += nll
            epoch_kl += kl
            epoch_loss = (epoch_nll+epoch_kl)/batch_i
            
            self.losses.append(loss)
            self.kl_loss.append(kl)
            self.recon_loss.append(nll)
            self.counter += 1
            
          self.plotter()
 
          if self.epoch_i % 10 == 0:
            # Save model after each 25 epochs
            self.checkpoint('svg.pt', self.epoch_i, epoch_loss) 
            
          # Do early stopping, if above patience stop training, if better loss, save model
          stop = self.earlystopping.step(self.epoch_i, epoch_loss)
          
          if stop:
            break
          if (self.earlystopping.best_loss < self.best_loss) and (self.epoch_i > 100):
            self.best_loss = self.earlystopping.best_loss
            self.checkpoint('svg_best_model.pt', self.epoch_i, epoch_loss) 
          
            
          if self.verbose:
            print('Epoch {} Loss: {:.2f}'.format(self.epoch_i, epoch_loss))
          else:
            self.status(epoch_loss)
 
    def checkpoint(self, model_name, epoch, loss):
      torch.save({
          'epoch': epoch,
          'model_state_dict': self.model.state_dict(),
          'frame_optimizer_state_dict': self.model.frame_predictor.state_dict(),
          'posterior_optimizer_state_dict': self.model.posterior.state_dict(),
          'prior_optimizer_state_dict': self.model.prior.state_dict(),
          'encoder_optimizer_state_dict': self.model.encoder.state_dict(),
          'decoder_optimizer_state_dict': self.model.decoder.state_dict(),
          'flow_optimizer_state_dict': self.model.flow.state_dict(),
          'loss': loss,
          'kl_loss': self.kl_loss,
          'recon_loss': self.recon_loss,
          'losses': self.losses,
          'plot_counter': self.plot_counter,
          'annealing_counter': self.counter,
          'args': self.args,
          }, self.path + 'model_folder/' + model_name)
      
      
    def load(self, load_model):
      self.model.load_state_dict(load_model['model_state_dict'])
      self.model.frame_predictor.load_state_dict(load_model['frame_optimizer_state_dict'])
      self.model.posterior.load_state_dict(load_model['posterior_optimizer_state_dict'])
      self.model.prior.load_state_dict(load_model['prior_optimizer_state_dict'])
      self.model.encoder.load_state_dict(load_model['encoder_optimizer_state_dict'])
      self.model.decoder.load_state_dict(load_model['decoder_optimizer_state_dict'])
      self.model.flow.load_state_dict(load_model['flow_optimizer_state_dict'])
      self.epoch_i += load_model['epoch']
      loss = load_model['loss']
      self.kl_loss = load_model['kl_loss']
      self.recon_loss = load_model['recon_loss']
      self.losses = load_model['losses']
      self.plot_counter = load_model['plot_counter']
      self.counter = load_model['annealing_counter']
      self.best_loss = loss
      self.model.to(device)
      return (self.epoch_i, loss)
  
    
    def status(self, epoch_loss):
      # Only works for python 3.x
      # TODO: Fix LR
      #lr = self.model.frame_predictor.param_groups[0]['lr'] 
      lr = 0.0001
      with open(self.path + 'model_folder/status.txt', 'a') as f:
        print("STATUS:", file=f)
        print("\tKL and Reconstruction loss: {:.4f}, {:.4f}".format(self.kl_loss[-1].data, self.recon_loss[-1].data), file=f)
        print(f'\tEpoch {self.epoch_i}, Beta value {self.model.beta:.4f}, Learning rate {lr}, epoch loss {epoch_loss:.4f}', file=f)
 
    def plotter(self):
      n_plot = str(self.plot_counter)
      self.model.frame_predictor.eval()
      self.model.posterior.eval() 
      self.model.prior.eval()
      self.model.encoder.eval()
      self.model.decoder.eval()
      self.model.flow.eval()
      
      with torch.no_grad():
        image = next(iter(self.test_loader)).to(device)
        time_steps = self.n_frames - 1
        
        image  = self.preprocess(image, reverse=False)
        predictions = self.model.plot_rec(image, encoder_sample=False)
        predictions  = self.preprocess(predictions, reverse=True)
        
        # With x
        predictions_x = self.model.plot_rec(image, encoder_sample=True)
        predictions_x  = self.preprocess(predictions_x, reverse=True)
        image  = self.preprocess(image, reverse=True)
        
 
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
      
      fig, ax = plt.subplots(3, time_steps , figsize = (20,8))
      for i in range(0, time_steps):
        ax[0,i].imshow(self.convert_to_numpy(image[0, i, :, :, :]))
        ax[0,i].set_title("True Image")
        ax[1,i].imshow(self.convert_to_numpy(predictions[i, 0, :, :, :]))
        ax[1,i].set_title("Predictions Image")
        ax[2,i].imshow(self.convert_to_numpy(predictions_x[i, 0, :, :, :]))
        ax[2,i].set_title("Predictions_x Image")
        
      if not self.verbose:
        fig.savefig(self.path +'png_folder/samples' + n_plot + '.png', bbox_inches='tight')
        plt.close(fig)

      if self.verbose:
        print("\tKL and Reconstruction loss: {:.4f}, {:.4f}".format(self.kl_loss[-1].data, self.recon_loss[-1].data))
        plt.show()
 
      self.plot_counter += 1
      
      self.model.frame_predictor.train()
      self.model.posterior.train() 
      self.model.prior.train()
      self.model.encoder.train()
      self.model.decoder.train()
      self.model.flow.train()

    def convert_to_numpy(self, x):
        return x.permute(1,2,0).squeeze().detach().cpu().numpy()
    
    

