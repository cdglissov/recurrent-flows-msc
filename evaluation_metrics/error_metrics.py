import sys
# Adding deepflows to system path
sys.path.insert(1, './deepflows/')
import torch 
import torch.utils.data
from torch.utils.data import DataLoader
import os
import lpips
from Utils import set_gpu
import matplotlib.pyplot as plt
from data_generators import MovingMNIST
from data_generators import PushDataset
device = set_gpu(True)
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from tqdm import tqdm 
import math

class Evaluator(object):
    def __init__(self, solver, args, settings):
        self.args = args
        self.n_bits = args.n_bits
        self.solver = solver
        self.n_frames = settings.n_frames
        self.start_predictions = settings.start_predictions
        self.verbose = args.verbose
        self.path = str(os.path.abspath(os.getcwd())) + args.path
        self.batch_size = args.batch_size
        self.choose_data = args.choose_data
        self.digit_size = args.digit_size
        self.step_length = args.step_length
        self.num_digits = args.num_digits
        self.image_size = args.image_size
        self.preprocess_range = args.preprocess_range
        self.preprocess_scale = args.preprocess_scale
        self.num_workers=args.num_workers
        self.multigpu = args.multigpu
        self.num_samples_to_plot = settings.num_samples_to_plot
        self.debug_mnist = settings.debug_mnist
        self.debug_plot = settings.debug_plot
        self.resample = settings.resample
        self.show_elbo_gap = settings.show_elbo_gap
        self.n_conditions = settings.n_conditions
        # Number of frames the model has been trained on
        self.n_trained = args.n_frames
        self.temperatures = settings.temperatures
        
    def build(self):
        self.test_loader = self.create_loaders()
        
        if not os.path.exists(self.path + 'png_folder'):
          os.makedirs(self.path + 'png_folder')
        if not os.path.exists(self.path + 'model_folder'):
          os.makedirs(self.path + 'model_folder')
        if not os.path.exists(self.path + 'eval_folder'):
          os.makedirs(self.path + 'eval_folder')
          
        if self.multigpu and torch.cuda.device_count() > 1:
            print("Using:", torch.cuda.device_count(), "GPUs")
            self.model = self.solver.model.to(device)
        else:
            self.model = self.solver.model.to(device)
            
        self.model.eval()
        # best forward scores
        self.lpipsNet = lpips.LPIPS(net='alex').to(device)
        
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
                
            if self.debug_mnist:
                te_split_len = 200
                testset = torch.utils.data.random_split(testset, 
                                [te_split_len, len(testset)-te_split_len])[0]
                
        if self.choose_data=='bair':
            	string=str(os.path.abspath(os.getcwd()))
            	testset = PushDataset(split='test',
                                             dataset_dir=string+'/bair_robot_data/processed_data/',
                                             seq_len=self.n_frames)

        test_loader = DataLoader(testset, batch_size=self.batch_size, 
                                 num_workers=self.num_workers, shuffle=True, drop_last=True)
        return test_loader    
    
    def convert_to_numpy(self, x):
        return x.permute(1,2,0).squeeze().detach().cpu().numpy()
    
    def plot_samples(self, predictions, true_image):
      num_samples = self.num_samples_to_plot # The number of examples plotted
      time_steps = predictions.shape[1]
      fig, ax = plt.subplots(num_samples*2, time_steps ,figsize = (time_steps, 4*num_samples))
      for k in range(0, num_samples):
          for i in range(0, time_steps):
            ax[2*(k),i].imshow(self.convert_to_numpy(true_image[k, i, :, :, :]))
            if i == 0:
                ax[2*(k),i].set_yticks([])
                ax[2*(k),i].set_xticks([])
                ax[2*(k),i].set_ylabel("Ground Truth")
            else:
                ax[2*(k),i].axis("off")
            ax[2*(k)+1,i].imshow(self.convert_to_numpy(predictions[k, i, :, :, :]))
            if i == 0:
                ax[2*(k)+1,i].set_yticks([])
                ax[2*(k)+1,i].set_xticks([])
                ax[2*(k)+1,i].set_ylabel("Prediction")
            else:
                ax[2*(k)+1,i].axis("off")
      fig.savefig(self.path +'eval_folder/samples' +  '.png', bbox_inches='tight') #dpi=fig.get_dpi()*2)
      plt.close(fig)
      
    # Eval From SVG
    def eval_seq(self, gt, pred):
        # Takes a ground truth (gt) of size [bs, time, c, h, w]
        T = gt.shape[1]
        bs = gt.shape[0]
        ssim = torch.zeros((bs, T))
        psnr = torch.zeros((bs, T))
        mse = torch.zeros((bs, T))
        for i in range(bs):
            for t in range(T):
                for c in range(gt.shape[2]):
                    image_gen = np.uint8(gt[i,t,c,:,:].cpu())
                    image_true = np.uint8(pred[i, t, c,:, :].cpu())
                    ssim[i, t] += structural_similarity(image_gen, image_true)
                    psnr[i, t] += peak_signal_noise_ratio(image_gen, image_true)                    
                ssim[i, t] /= gt.shape[2]
                psnr[i, t] /= gt.shape[2]
                mse[i, t] = torch.mean( (gt[i, t, :, : ,:] - pred[i, t, :, : ,:])**2, dim = [0, 1, 2])
        return mse, ssim, psnr
    
    def get_lpips(self, X, Y):
        T = Y.shape[1]
        bs = X.shape[0]
        lpips = torch.zeros((bs, T))
        for i in range(bs):
            for t in range(T):
                # If range [0,255] needs to be [-1,1]
                img0 = X[i,t,:,:,:].to(device) / 255 * 2 -1
                img1 = Y[i,t,:,:,:].to(device) / 255 * 2 -1
                # 3 image channels is required.
                if img0.shape[0] == 1:
                    img0 = img0.repeat(3,1,1)
                    img1 = img1.repeat(3,1,1)
                lpips[i,t] = self.lpipsNet(img0, img1)
        return lpips
    
    def plot_elbo_gap(self, image):
        recons, recons_flow, averageKLDseq, averageNLLseq = self.model.reconstruct_elbo_gap(image)
        recons  = self.solver.preprocess(recons, reverse=True)
        recons_flow  = self.solver.preprocess(recons_flow, reverse=True)
        image  = self.solver.preprocess(image, reverse=True)
        time_steps = image.shape[1]
        fig, ax = plt.subplots(7, time_steps , figsize = (2*time_steps, 2*6))
        for i in range(0, time_steps):
            ax[0,i].imshow(self.convert_to_numpy(image[0, i, :, :, :]))
            ax[0,i].set_title("True Image")
            ax[0,i].axis('off')
            for z, zname in zip(range(0,2),list(['Prior','Encoder'])): 
                ax[1+z,i].imshow(self.convert_to_numpy(recons[z, i, 0, :, :, :]))
                ax[1+z,i].set_title(str(zname))
                ax[1+z,i].axis('off')
                ax[3+z,i].imshow(self.convert_to_numpy(recons_flow[z,i, 0, :, :, :]))
                ax[3+z,i].set_title(str(zname)+' flow')
                ax[3+z,i].axis('off')
        plt.subplot(8, 1, 7)
        plt.bar(np.arange(time_steps), averageKLDseq[:,0], align='center', width=0.3)
        plt.xlim((0-0.5, time_steps-0.5))
        plt.xticks(range(0, time_steps), range(0, time_steps))
        plt.xlabel("Frame number")
        plt.ylabel("Average KLD")
        plt.subplot(8, 1, 8)
        plt.bar(np.arange(time_steps)-0.15, -averageNLLseq[0, :, 0], align='center', width=0.3,label = 'Prior')
        plt.bar(np.arange(time_steps)+0.15, -averageNLLseq[1, :, 0], align='center', width=0.3,label = 'Posterior')
        plt.xlim((0-0.5, time_steps-0.5))
        low = min(min(-averageNLLseq[0, 1:, 0]),min(-averageNLLseq[1, 1:, 0]))
        high = max(max(-averageNLLseq[0, 1:, 0]),max(-averageNLLseq[1, 1:, 0]))
        plt.ylim([math.ceil(low-0.5*(high-low)), math.ceil(high+0.5*(high-low))])
        plt.xticks(range(0, time_steps), range(0, time_steps))
        plt.xlabel("Frame number")
        plt.ylabel("bits dim ll")
        plt.legend()
        fig.savefig(self.path + 'eval_folder/KLDdiagnostic' + '.png', bbox_inches='tight')
        plt.close(fig)
    
    def compute_loss(self, nll, kl, dims, t=10):

        kl_store = kl.data
        nll_store = nll.data
        elbo = -(kl_store+nll_store)

        bits_per_dim_loss = float(-elbo/(np.log(2.)*torch.prod(torch.tensor(dims))*t))
        kl_loss = float(kl_store/t)
        recon_loss = float(nll_store/t)
        
        return bits_per_dim_loss, kl_loss, recon_loss
    
    def get_eval_values(self, model_name):
      start_predictions = self.start_predictions 
      PSNR_max = 0.0
      MSE_min = 1e15
      SSIM_max = 0.0
      LPIPS_min = 1e15
      BPD_min = 1e15
      
      with torch.no_grad():
          self.model.eval()
          for time in range(0, self.resample):
              SSIM_values = []
              PSNR_values = []
              MSE_values = []
              LPIPS_values = []
              BPD = [] 
              DKL=[] 
              RECON = []
              for batch_i, image in enumerate(tqdm(self.test_loader, desc="Running", position=0, leave=True)):
                
                if self.choose_data=='bair':
                    image = image[0].to(device)
                else:
                    image = image.to(device)  
                image = self.solver.preprocess(image)
                
                x_true, predictions = self.model.predict(image, self.n_frames-start_predictions, start_predictions)
                
                # Computes eval loss
                if model_name == "rfn.pt":
                    logdet = 0
                    _, kl, nll = self.model.loss(image, logdet)
                    bits_per_dim_loss, kl_loss, recon_loss = self.compute_loss(nll=nll, 
                                                                               kl=kl, 
                                                                               dims=image.shape[2:], 
                                                                               t=image.shape[1]-1)
                else:
                    kl, nll = self.model.loss(image)
                    bits_per_dim_loss, kl_loss, recon_loss = self.compute_loss(nll=nll, 
                                                                               kl=kl, 
                                                                               dims=image.shape[2:], 
                                                                               t=image.shape[1]-1)
                
                image  = self.solver.preprocess(image, reverse=True)
                predictions  = self.solver.preprocess(predictions, reverse=True)

                ground_truth = image[:, start_predictions:,:,:,:].type(torch.FloatTensor).to(device)
                predictions = predictions.permute(1,0,2,3,4).type(torch.FloatTensor).to(device)
                
                mse, ssim, psnr = self.eval_seq(predictions, ground_truth)
                lpips = self.get_lpips(predictions, ground_truth)
                
                
                SSIM_values.append(ssim)
                PSNR_values.append(psnr)
                MSE_values.append(mse)
                LPIPS_values.append(lpips)
                BPD.append(bits_per_dim_loss)
                DKL.append(kl_loss) 
                RECON.append(recon_loss)
               
              #TODO: Det er lidt mærkeligt at den skal være inde i eval loopet. Så har rykket den ud
              # Måske gem den gennemsnitlige loss værdi for alle de forskellige inputs og så plot?
              if self.show_elbo_gap:
                self.plot_elbo_gap(image)
      
              # Shape: [number of sequences, number of future frames predicted]
              PSNR_values = torch.cat(PSNR_values)
              MSE_values = torch.cat(MSE_values)
              SSIM_values = torch.cat(SSIM_values)
              LPIPS_values = torch.cat(LPIPS_values)
        
              # Shape: [one avg. loss for each minibatch]
              BPD = torch.FloatTensor(BPD)
              DKL = torch.FloatTensor(DKL)
              RECON = torch.FloatTensor(RECON)
              
              if BPD.mean() < BPD_min:
                  BPD_best = BPD
                  DKL_best = DKL
                  RECON_best = RECON
                  BPD_min = BPD.mean()
                  
              if MSE_values.mean() < MSE_min:
                  MSE_best = MSE_values
                  MSE_min = MSE_values.mean()
              
              if PSNR_values.mean() > PSNR_max:
                  PSNR_max = PSNR_values.mean()
                  PSNR_best = PSNR_values
                  
              if SSIM_values.mean() > SSIM_max:
                  SSIM_max = SSIM_values.mean()
                  SSIM_best = SSIM_values
                  
              if LPIPS_values.mean() < LPIPS_min:
                  LPIPS_min = LPIPS_values.mean()
                  LPIPS_best = LPIPS_values
      #maybe give best and worst plots to this, with corresponding ssim score etc
      if self.debug_plot:
          self.plot_samples(predictions.byte(), ground_truth.byte())
      
      return MSE_best, PSNR_best, SSIM_best, LPIPS_best, BPD_best, DKL_best, RECON_best
    
    def test_temp_values(self, path, label_names, experiment_names):
        markersize = 5
        n_train = self.n_trained
        markers = ["o", "v", "x", "*", "^", "s", "H", "P", "X"]
        
        for i in range(0,len(experiment_names)):
            fig, ax = plt.subplots(1, 3 ,figsize = (20,5))
            fig2, ax2 = plt.subplots(1, 3 ,figsize = (20,5))
            for temp in self.temperatures:
                lname = label_names[i] + " "+ str(temp)
                mark = markers[i]
                alpha = 0.05
                temp_id = "/t" + str(temp).replace('.','')
                eval_dict = torch.load(path + experiment_names[i] + '/eval_folder/'+temp_id+'evaluations.pt')
                SSIM  = eval_dict['SSIM_values']
                PSNR  = eval_dict['PSNR_values']
                LPIPS  = eval_dict['LPIPS_values']
                
                print("Temperature is set to " + str(eval_dict['temperature']) + " for experiment " +lname)
                
                y = SSIM.mean(0).numpy()
                xaxis = np.arange(0+self.n_conditions, len(y)+self.n_conditions)
                conf_std = 1.96 * np.std(SSIM.numpy(),0)/np.sqrt(np.shape(SSIM.numpy())[0])
                ax[0].plot(xaxis, y, label = lname, marker=mark, markersize=markersize)
                ax[0].fill_between(xaxis, y-conf_std, y+conf_std, alpha=.1)
                
                y = PSNR.mean(0).numpy()
                twostd = 1.96 * np.std(PSNR.numpy(),0)/np.sqrt(np.shape(PSNR.numpy())[0])
                ax[1].plot(xaxis,y,label = lname, marker=mark, markersize=markersize)
                ax[1].fill_between(xaxis, y-twostd, y+twostd, alpha=.1)
                
                y = LPIPS.mean(0).numpy()
                twostd = 1.96 * np.std(LPIPS.numpy(),0)/np.sqrt(np.shape(LPIPS.numpy())[0])
                ax[2].plot(xaxis,y,label = lname, marker=mark, markersize=markersize)
                ax[2].fill_between(xaxis, y-twostd, y+twostd, alpha=.1)
            
                y = np.median(SSIM.numpy(),0)
                ax2[0].plot(xaxis, y, label = lname, marker=mark,markersize=markersize)
                ax2[0].fill_between(xaxis, 
                  np.quantile(SSIM.numpy(), alpha/2, axis = 0), 
                  np.quantile(SSIM.numpy(), 1-alpha/2, axis = 0), alpha=.1)
                
                y = np.median(PSNR.numpy(),0)
                ax2[1].plot(xaxis, y, label = lname, marker=mark, markersize=markersize)
                ax2[1].fill_between(xaxis, 
                  np.quantile(PSNR.numpy(), alpha/2, axis = 0), 
                  np.quantile(PSNR.numpy(), 1-alpha/2, axis = 0), alpha=.1)
                
                y = np.median(LPIPS.numpy(),0)
                ax2[2].plot(np.arange(0, len(y)), y, label = lname, marker=mark, 
                   markersize=markersize)
                ax2[2].fill_between(np.arange(0, len(y)), 
                  np.quantile(LPIPS.numpy(), alpha/2, axis = 0), 
                  np.quantile(LPIPS.numpy(), 1-alpha/2, axis = 0), alpha=.1)
            
            ax[0].set_ylabel('score')
            ax[0].set_xlabel('t')
            ax[0].set_title('Avg. SSIM with 95% confidence interval')
            ax[0].axvline(x=n_train, color='k', linestyle='--')
            ax[0].legend()
            ax[0].grid()
            
            ax[1].set_ylabel('score')
            ax[1].set_xlabel('t')
            ax[1].set_title('Avg. PSNR with 95% confidence interval')
            ax[1].axvline(x=n_train, color='k', linestyle='--')
            ax[1].legend()
            ax[1].grid()
            
            ax[2].set_ylabel('score')
            ax[2].set_xlabel('t')
            ax[2].set_title('Avg. LPIPS with 95% confidence interval')
            ax[2].axvline(x=n_train, color='k', linestyle='--')
            ax[2].legend()
            ax[2].grid()
            
            ax2[0].set_ylabel('score')
            ax2[0].set_xlabel('t')
            ax2[0].set_title('Avg. SSIM with 95% quantiles')
            ax2[0].axvline(x=n_train, color='k', linestyle='--')
            ax2[0].legend()
            ax2[0].grid()
            
            ax2[1].set_ylabel('score')
            ax2[1].set_xlabel('t')
            ax2[1].set_title('Avg. PSNR with 95% quantiles')
            ax2[1].axvline(x=n_train, color='k', linestyle='--')
            ax2[1].legend()
            ax2[1].grid()
            
            ax2[2].set_ylabel('score')
            ax2[2].set_xlabel('t')
            ax2[2].set_title('Avg. LPIPS with 95% quantiles')
            ax2[2].axvline(x=n_train, color='k', linestyle='--')
            ax2[2].legend()
            ax2[2].grid()
            
            fig.savefig(path + experiment_names[i] + '/eval_folder/temps_eval_plots_mean.png', bbox_inches='tight')
            fig2.savefig(path + experiment_names[i] +  '/eval_folder/temps_eval_plots_median.png', bbox_inches='tight')  
    
    def plot_eval_values(self, path, label_names, experiment_names):
        markersize = 5
        n_train = self.n_trained
        fig, ax = plt.subplots(1, 3 ,figsize = (20,5))
        fig2, ax2 = plt.subplots(1, 3 ,figsize = (20,5))
        markers = ["o", "v", "x", "*", "^", "s", "H", "P", "X"]
        
        for i in range(0,len(experiment_names)):
            lname = label_names[i]
            mark = markers[i]
            alpha = 0.05
            eval_dict = torch.load(path + experiment_names[i] + '/eval_folder/evaluations.pt')
            SSIM  = eval_dict['SSIM_values']
            PSNR  = eval_dict['PSNR_values']
            LPIPS  = eval_dict['LPIPS_values']
            
            print("Temperature is set to " + str(eval_dict['temperature']) + " for experiment " +lname)
            
            y = SSIM.mean(0).numpy()
            xaxis = np.arange(0+self.n_conditions, len(y)+self.n_conditions)
            conf_std = 1.96 * np.std(SSIM.numpy(),0)/np.sqrt(np.shape(SSIM.numpy())[0])
            ax[0].plot(xaxis, y, label = lname, marker=mark, markersize=markersize)
            ax[0].fill_between(xaxis, y-conf_std, y+conf_std, alpha=.1)
            
            y = PSNR.mean(0).numpy()
            twostd = 1.96 * np.std(PSNR.numpy(),0)/np.sqrt(np.shape(PSNR.numpy())[0])
            ax[1].plot(xaxis,y,label = lname, marker=mark, markersize=markersize)
            ax[1].fill_between(xaxis, y-twostd, y+twostd, alpha=.1)
            
            y = LPIPS.mean(0).numpy()
            twostd = 1.96 * np.std(LPIPS.numpy(),0)/np.sqrt(np.shape(LPIPS.numpy())[0])
            ax[2].plot(xaxis,y,label = lname, marker=mark, markersize=markersize)
            ax[2].fill_between(xaxis, y-twostd, y+twostd, alpha=.1)
        
            y = np.median(SSIM.numpy(),0)
            ax2[0].plot(xaxis, y, label = lname, marker=mark,markersize=markersize)
            ax2[0].fill_between(xaxis, 
              np.quantile(SSIM.numpy(), alpha/2, axis = 0), 
              np.quantile(SSIM.numpy(), 1-alpha/2, axis = 0), alpha=.1)
            
            y = np.median(PSNR.numpy(),0)
            ax2[1].plot(xaxis, y, label = lname, marker=mark, markersize=markersize)
            ax2[1].fill_between(xaxis, 
              np.quantile(PSNR.numpy(), alpha/2, axis = 0), 
              np.quantile(PSNR.numpy(), 1-alpha/2, axis = 0), alpha=.1)
            
            y = np.median(LPIPS.numpy(),0)
            ax2[2].plot(np.arange(0, len(y)), y, label = lname, marker=mark, 
               markersize=markersize)
            ax2[2].fill_between(np.arange(0, len(y)), 
              np.quantile(LPIPS.numpy(), alpha/2, axis = 0), 
              np.quantile(LPIPS.numpy(), 1-alpha/2, axis = 0), alpha=.1)
            
        ax[0].set_ylabel('score')
        ax[0].set_xlabel('t')
        ax[0].set_title('Avg. SSIM with 95% confidence interval')
        ax[0].axvline(x=n_train, color='k', linestyle='--')
        ax[0].legend()
        ax[0].grid()
        
        ax[1].set_ylabel('score')
        ax[1].set_xlabel('t')
        ax[1].set_title('Avg. PSNR with 95% confidence interval')
        ax[1].axvline(x=n_train, color='k', linestyle='--')
        ax[1].legend()
        ax[1].grid()
        
        ax[2].set_ylabel('score')
        ax[2].set_xlabel('t')
        ax[2].set_title('Avg. LPIPS with 95% confidence interval')
        ax[2].axvline(x=n_train, color='k', linestyle='--')
        ax[2].legend()
        ax[2].grid()
        
        ax2[0].set_ylabel('score')
        ax2[0].set_xlabel('t')
        ax2[0].set_title('Avg. SSIM with 95% quantiles')
        ax2[0].axvline(x=n_train, color='k', linestyle='--')
        ax2[0].legend()
        ax2[0].grid()
        
        ax2[1].set_ylabel('score')
        ax2[1].set_xlabel('t')
        ax2[1].set_title('Avg. PSNR with 95% quantiles')
        ax2[1].axvline(x=n_train, color='k', linestyle='--')
        ax2[1].legend()
        ax2[1].grid()
        
        ax2[2].set_ylabel('score')
        ax2[2].set_xlabel('t')
        ax2[2].set_title('Avg. LPIPS with 95% quantiles')
        ax2[2].axvline(x=n_train, color='k', linestyle='--')
        ax2[2].legend()
        ax2[2].grid()
        
        fig.savefig(path + experiment_names[i] + '/eval_folder/eval_plots_mean.png', bbox_inches='tight')
        fig2.savefig(path + experiment_names[i] +  '/eval_folder/eval_plots_median.png', bbox_inches='tight') 
    
