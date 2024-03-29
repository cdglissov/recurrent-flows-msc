'''Train script file'''
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from data_generators import stochasticMovingMnist
from data_generators import bair_push
import matplotlib.pyplot as plt
from VRNN import VRNN
from utils import set_gpu
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
        self.patience_lr = args.patience_lr
        self.factor_lr = args.factor_lr
        self.min_lr = args.min_lr
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
        
    def build(self):
        self.train_loader, self.test_loader = self.create_loaders()
        
        if not os.path.exists(self.path + 'png_folder'):
          os.makedirs(self.path + 'png_folder')
        if not os.path.exists(self.path + 'model_folder'):
          os.makedirs(self.path + 'model_folder')
        
        model = VRNN(self.params)
        
        if self.multigpu and torch.cuda.device_count() > 1:
            print("Using:", torch.cuda.device_count(), "GPUs")
            self.model = nn.DataParallel(model).to(device)
        else:
            self.model = model.to(device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', 
                                                                    patience=self.patience_lr, 
                                                                    factor=self.factor_lr, 
                                                                    min_lr=self.min_lr)
        self.earlystopping = EarlyStopping(min_delta = 0, patience = self.patience_es, 
                                           verbose = self.verbose)
        
        
    def create_loaders(self):
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
            	trainset = stochasticMovingMnist.MovingMNIST(True, 'Mnist', 
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
            	trainset = bair_push.PushDataset(split='train',
                                              dataset_dir=string+'/bair_robot_data/processed_data/',
                                              seq_len=self.n_frames)
            	testset = bair_push.PushDataset(split='test',
                                             dataset_dir=string+'/bair_robot_data/processed_data/',
                                             seq_len=self.n_frames)

        train_loader = DataLoader(trainset,batch_size=self.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(testset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        return train_loader, test_loader
 
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
        elif preprocess_range == "minmax":
            if reverse == False:
                x = (x - x.min()) / (x.max() - x.min())
            else:
                x = x * (x.max()-x.min())+x.min()
        else:
            print("Invalid preprocess choice")
        return x
 
    def train(self):
      counter = 0
      max_value = self.beta_max
      min_value = self.beta_min
      num_steps = self.beta_steps

      for epoch_i in range(self.n_epochs):
          self.model.train()
          self.epoch_i += 1
          self.batch_loss_history = []
          for batch_i, image in enumerate(tqdm(self.train_loader, desc="Epoch " + str(self.epoch_i), position=0, leave=False)):
            batch_i += 1
            if self.choose_data=='bair':
                image = image[0].to(device)
            else:
                image = image.to(device)
            image = self.preprocess(image)
            
            #TODO ADD TO ARGUMENTS
            if self.variable_beta: 
                self.model.beta = min(max_value, min_value + counter*(max_value - min_value) / num_steps)
            else:
                self.model.beta = max_value
            if self.multigpu and torch.cuda.device_count() > 1:
                kl, nll = self.model.module.loss(image)
            else:
                kl, nll = self.model.loss(image)
            loss = nll + kl
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            store_loss = float(loss.data)
            self.losses.append(store_loss)
            self.kl_loss.append(kl.data)
            self.recon_loss.append(nll.data)
            counter += 1
            
          self.plotter()   
          epoch_loss = np.mean(self.losses)
 
          if self.epoch_i % 1 == 0:
            # Save model after each 25 epochs
            self.checkpoint('vrnn.pt', self.epoch_i, epoch_loss) 
 
          # Do early stopping, if above patience stop training, if better loss, save model
          stop = self.earlystopping.step(self.epoch_i, epoch_loss)
          if stop:
            break
          if (self.earlystopping.best_loss < self.best_loss) and (self.epoch_i > 50):
            self.best_loss = self.earlystopping.best_loss
            self.checkpoint('vrnn_best_model.pt', self.epoch_i, epoch_loss) 
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
          'kl_loss': self.kl_loss,
          'recon_loss': self.recon_loss}, self.path + 'model_folder/' + model_name)
      
    def load(self, path):
      load_model = torch.load(path)
      self.model.load_state_dict(load_model['model_state_dict'])
      self.optimizer.load_state_dict(load_model['optimizer_state_dict'])
      self.epoch_i += load_model['epoch']
      loss = load_model['loss']
      self.best_loss = loss
      return (self.epoch_i, loss)
 
    def status(self):
      # Only works for python 3.x
      lr = self.optimizer.param_groups[0]['lr']
      with open(self.path + 'model_folder/status.txt', 'a') as f:
        print("STATUS:", file=f)
        print("\tKL and Reconstruction loss: {:.4f}, {:.4f}".format(self.kl_loss[-1].data, self.recon_loss[-1].data), file=f)
        print(f'\tEpoch {self.epoch_i}, Beta value {self.model.beta:.4f}, Learning rate {lr}', file=f)
 
    def plotter(self):
      n_plot = str(self.plot_counter)
      with torch.no_grad():
        self.model.eval()
        image = next(iter(self.test_loader)).to(device)
        time_steps = self.n_frames - 1
        n_predictions = time_steps
        image  = self.preprocess(image, reverse=False)
        
        #without encoder
        one_step_predictions, predictions = self.model.predict(image, n_predictions = n_predictions, encoder_sample = False)
        recons = self.model.reconstruct(image)
        samples = self.model.sample(n_samples = time_steps)
        samples  = self.preprocess(samples, reverse=True)
        recons  = self.preprocess(recons, reverse=True)
        predictions  = self.preprocess(predictions, reverse=True)
        one_step_predictions  = self.preprocess(one_step_predictions, reverse=True)
        
        # With encoder
        one_step_predictions_x, predictions_x = self.model.predict(image, n_predictions = n_predictions, encoder_sample = True)
        recons_x = self.model.reconstruct(image)
        one_step_predictions_x  = self.preprocess(one_step_predictions_x, reverse=True)
        recons_x  = self.preprocess(recons_x, reverse=True)
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
      
      fig, ax = plt.subplots(6, time_steps , figsize = (20,5*5))
      for i in range(0, time_steps):
        ax[0,i].imshow(self.convert_to_numpy(one_step_predictions[0, i, :, :, :]))
        ax[0,i].set_title("Random one step predictions")
        ax[1,i].imshow(self.convert_to_numpy(one_step_predictions[i, 0, :, :, :]))
        ax[1,i].set_title("One step predictions at timestep t")
        ax[2,i].imshow(self.convert_to_numpy(image[0, i, :, :, :]))
        ax[2,i].set_title("True Image")
        ax[3,i].imshow(self.convert_to_numpy(predictions[i, 0, :, :, :]))
        ax[3,i].set_title("Prediction")
        ax[4,i].imshow(self.convert_to_numpy(samples[i, 0, :, :, :]))
        ax[4,i].set_title("Sample at timestep t")
        ax[5,i].imshow(self.convert_to_numpy(samples[0, i, :, :, :]))
        ax[5,i].set_title("Random Samples at timestep 1")
      if not self.verbose:
        fig.savefig(self.path +'png_folder/samples' + n_plot + '.png', bbox_inches='tight')
        plt.close(fig)

      fig, ax = plt.subplots(5, time_steps , figsize = (20,5*5))
      for i in range(0, time_steps):
        ax[0,i].imshow(self.convert_to_numpy(one_step_predictions_x[0, i, :, :, :]))
        ax[0,i].set_title("Random one step predictions")
        ax[1,i].imshow(self.convert_to_numpy(one_step_predictions_x[i, 0, :, :, :]))
        ax[1,i].set_title("Prediction at timestep t")
        ax[2,i].imshow(self.convert_to_numpy(image[0, i, :, :, :]))
        ax[2,i].set_title("True Image")
        ax[3,i].imshow(self.convert_to_numpy(recons_x[i, 0, :, :, :]))
        ax[3,i].set_title("Reconstructed Image")
        ax[4,i].imshow(self.convert_to_numpy(predictions_x[i, 0, :, :, :]))
        ax[4,i].set_title("Prediction")
      if not self.verbose:
        fig.savefig(self.path +'png_folder/samples_with_x'+ n_plot + '.png', bbox_inches='tight')
        plt.close(fig)
      if self.verbose:
        print("\tKL and Reconstruction loss: {:.4f}, {:.4f}".format(self.kl_loss[-1].data, self.recon_loss[-1].data))
        plt.show()
 
      self.plot_counter += 1
      self.model.train()

    def convert_to_numpy(self, x):
        return x.permute(1,2,0).squeeze().detach().cpu().numpy()