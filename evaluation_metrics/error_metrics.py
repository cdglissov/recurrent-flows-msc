import sys
# Adding deepflows to system path
sys.path.insert(1, './deepflows_git_gut/')
import torch
import torch.utils.data
from torch.utils.data import DataLoader
import os
import lpips
from Utils import set_gpu
import matplotlib.pyplot as plt
from data_generators import MovingMNIST
from data_generators import MovingMNIST_synchronized
from data_generators import PushDataset
from data_generators import KTH
device = set_gpu(True)
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from tqdm import tqdm
from evaluation_metrics.FVD_score import fvd
plt.rcParams.update({'text.usetex': True})

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
        self.extra_plots = settings.extra_plots
        self.n_conditions = settings.n_conditions
        # Number of frames the model has been trained on
        self.n_trained = args.n_frames
        self.temperatures = settings.temperatures
        self.use_validation_set = settings.use_validation_set

    def build(self):
        self.test_loader = self.create_loaders()

        if not os.path.exists(self.path + 'png_folder'):
          os.makedirs(self.path + 'png_folder')
        if not os.path.exists(self.path + 'model_folder'):
          os.makedirs(self.path + 'model_folder')
        if not os.path.exists(self.path + 'eval_folder'):
          os.makedirs(self.path + 'eval_folder')
        print(self.path + 'eval_folder')
        
        if self.multigpu and torch.cuda.device_count() > 1:
            print("Using:", torch.cuda.device_count(), "GPUs")
            self.model = self.solver.model.to(device)
        else:
            self.model = self.solver.model.to(device)

        self.model.eval()
        # best forward scores
        self.lpipsNet = lpips.LPIPS(net='alex').to(device)

    def create_loaders(self, drop_last=True):
        #if self.use_validation_set:
        #    train_boolean=True
        #    train_bair='train'
        #else:
        train_boolean=False
        train_bair='test'

        if self.choose_data=='mnist':
            plt.rcParams['image.cmap']='gray'
            testset = MovingMNIST(train_boolean, 'Mnist',
                                 seq_len=self.n_frames,
                                 image_size=self.image_size,
                                 digit_size=self.digit_size,
                                 num_digits=self.num_digits,
												deterministic=False,
                                 three_channels=False,
                                 step_length=self.step_length,
                                 normalize=False)

            if self.debug_mnist:
                te_split_len = 1000
                testset = torch.utils.data.random_split(testset,
                                [te_split_len, len(testset)-te_split_len])[0]

        if self.choose_data=='bair':
            plt.rcParams['image.cmap']='viridis'
            string=str(os.path.abspath(os.getcwd()))
            testset = PushDataset(split=train_bair,
                                             dataset_dir=string+'/bair_robot_data/processed_data/',
                                             seq_len=self.n_frames)
        
        if self.choose_data =='kth':
            plt.rcParams['image.cmap']='gray'
            string=str(os.path.abspath(os.getcwd()))
            testset = KTH(
                        train=train_boolean, 
                        data_root=string+"/kth_data",
                        seq_len=self.n_frames, 
                        image_size=self.image_size)

        if self.use_validation_set:
            testset_sub = torch.utils.data.Subset(testset, list(range(0, 1000, 1)))
            test_loader = DataLoader(testset_sub, batch_size=self.batch_size,
                                     num_workers=self.num_workers, shuffle=False, drop_last=drop_last)
        else:
            test_loader = DataLoader(testset, batch_size=self.batch_size,
                                     num_workers=self.num_workers, shuffle=False, drop_last=drop_last)
        return test_loader

    def convert_to_numpy(self, x):
        return x.permute(1,2,0).squeeze().detach().cpu().numpy()


    def plot_samples(self, predictions, true_image, name="samples", eval_score = None, set_top=0.9):
      num_samples = self.num_samples_to_plot # The number of examples plotted
      time_steps = predictions.shape[1]
      fig, ax = plt.subplots(num_samples*2, time_steps, gridspec_kw = {'wspace':0.0001, 'hspace':0.0001, 'top':set_top} ,figsize = (time_steps, num_samples*2))
      plt.subplots_adjust(wspace=0.0001,hspace= 0.0001, top=set_top)
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
            if eval_score != None:
                ax[2*(k)+1,i].set_title(f"{float(eval_score[k,i]):.3f}")
            if i == 0:
                ax[2*(k)+1,i].set_yticks([])
                ax[2*(k)+1,i].set_xticks([])
                ax[2*(k)+1,i].set_ylabel("Prediction")
            else:
                ax[2*(k)+1,i].axis("off")
      fig.savefig(self.path +'eval_folder/' + name +  '.pdf', bbox_inches="tight")
      plt.close(fig)


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
                # 3 image channels are required.
                if img0.shape[0] == 1:
                    img0 = img0.repeat(3,1,1)
                    img1 = img1.repeat(3,1,1)
                lpips[i,t] = self.lpipsNet(img0, img1)
        return lpips

    def plot_elbo_gap(self, image):
        # Restrict the length of the image.
        t = 10
        image = image[:,0:t,:,:,:]
        recons, recons_flow, averageKLDseq, averageNLLseq = self.model.reconstruct_elbo_gap(image)
        dims = image.shape[2:]
        averageNLLseq = averageNLLseq/(np.log(2.)*torch.prod(torch.tensor(dims)))
        recons  = self.solver.preprocess(recons, reverse=True)
        recons_flow  = self.solver.preprocess(recons_flow, reverse=True)
        image  = self.solver.preprocess(image, reverse=True)
        time_steps = image.shape[1]
        fig, ax = plt.subplots(5, time_steps , figsize = (2*time_steps, 1.62*6))
        for i in range(0, time_steps):
            ax[0,i].imshow(self.convert_to_numpy(image[0, i, :, :, :]))
            ax[0,i].set_xticks([])
            ax[0,i].set_yticks([])
            for z, zname in zip(range(0,2),list(['Prior','Encoder'])):
                if i == 0:
                    ax[1+z,i].axis('off')
                else:
                    ax[1+z,i].imshow(self.convert_to_numpy(recons[z, i, 0, :, :, :]))
                #ax[1+z,i].set_title(str(zname))
                ax[1+z,i].set_xticks([])
                ax[1+z,i].set_yticks([])
        
        fontsize = 30
        rotation = 0
        labelpad = 60
        ax[0,0].set_ylabel(r'GT:',fontsize = fontsize, rotation = rotation, labelpad = labelpad)
        ax[1,1].set_ylabel(r'Prior:',fontsize = fontsize, rotation = rotation, labelpad = labelpad)
        ax[2,1].set_ylabel(r'Posterior:',fontsize = fontsize, rotation = rotation, labelpad = labelpad)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        ax1 = plt.subplot(8, 1, 6)
        plt.bar(np.arange(time_steps), averageKLDseq[:,0], align='center', width=0.3)
        plt.xlim((0-0.5, time_steps-0.5))
        plt.xticks(range(0, time_steps), range(0, time_steps))
        #plt.xlabel(r"$t$")
        plt.ylabel(r"Avg. KLD",fontsize = 20)
        plt.yticks(fontsize=20)
        ax2 = plt.subplot(8, 1, 7)
        plt.bar(np.arange(time_steps)-0.15, averageNLLseq[0, :, 0], align='center', width=0.3,label = 'Prior')
        plt.bar(np.arange(time_steps)+0.15, averageNLLseq[1, :, 0], align='center', width=0.3,label = 'Posterior')
        plt.xlim((0-0.5, time_steps-0.5))
        low = min(min(averageNLLseq[0, 1:, 0]),min(averageNLLseq[1, 1:, 0]))
        high = max(max(averageNLLseq[0, 1:, 0]),max(averageNLLseq[1, 1:, 0]))
        
        plt.ylim([low-0.5*(high-low), high+0.5*(high-low)])
        plt.xticks(range(0, time_steps), range(0, time_steps),fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel(r"$t$",fontsize=fontsize)
        plt.ylabel(r"BPP",fontsize=20)
        ax1.get_shared_x_axes().join(ax1, ax2)
        ax1.set_xticklabels([])
        plt.legend(fontsize=20)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.05, hspace=0.1)
        fig.savefig(self.path + 'eval_folder/KLDdiagnostic' + '.pdf', bbox_inches='tight')
        plt.close(fig)


    def plot_prob_of_t(self, probT):
        plt.figure()
        xaxis = np.arange(self.n_conditions, probT.shape[2] + self.n_conditions)
        y = probT[:,0,:].mean(0)
        plt.plot(xaxis, y, label = 'Prior')
        # Almost no difference between prior and posterior, so only plot prior
        #plt.plot(xaxis, probT[:,1,:].mean(0), label = 'Posterior')
        #plt.legend()
        conf_std = 1.96 * np.std(probT[:,0,:].numpy(),0)/np.sqrt(np.shape(probT[:,0,:].numpy())[0])
        plt.fill_between(xaxis, y-conf_std, y+conf_std, alpha=.1)

        plt.ylabel(r"Bits per pixel", fontsize = 20)
        plt.xlabel(r"Frame number:$X_{t}$", fontsize = 20)
        plt.title(r'$P(X_{'+str(self.n_conditions)+'}= X_t \mid X_{<'+str(self.n_conditions)+'})$', fontsize = 20)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.grid()

        plt.savefig(self.path + 'eval_folder/bpp_sequence' + '.pdf', bbox_inches='tight')

        plt.close()

    def get_interpolations(self):
        ## Only works for RFN when trained on one digit.
        
        interpolation_set = MovingMNIST(False, 'Mnist',
                                 seq_len=self.n_frames,
                                 image_size=self.image_size,
                                 digit_size=self.digit_size,
                                 num_digits=self.num_digits,
												 deterministic=False,
                                 three_channels=False,
                                 step_length=self.step_length,
                                 normalize=False, set_starting_position = True, seed = 3)
        interpolation_loader = DataLoader(interpolation_set, batch_size=self.batch_size,
                                 num_workers=self.num_workers, shuffle=False, drop_last=True)
        interpolation_set_2 = MovingMNIST(False, 'Mnist',
                                 seq_len=self.n_frames,
                                 image_size=self.image_size,
                                 digit_size=self.digit_size,
                                 num_digits=self.num_digits,
												 deterministic=False,
                                 three_channels=False,
                                 step_length=self.step_length,
                                 normalize=False, set_starting_position = True, seed = 10)
        interpolation_loader_2 = DataLoader(interpolation_set_2, batch_size=self.batch_size,
                                 num_workers=self.num_workers, shuffle=False, drop_last=True)
        # So two dataset with the same loader
        set1 = next(iter(interpolation_loader)).to(device)
        set2 = next(iter(interpolation_loader_2)).to(device)

        num_timestep = 7
        num_interpolations = 5
        num_batch = 10 # Chooses the batch so different numbers

        set1 = self.solver.preprocess(set1)
        set2 = self.solver.preprocess(set2)

        zts_1, hts_1 = self.model.get_zt_ht_from_seq(set1,num_timestep)
        zts_2, hts_2 = self.model.get_zt_ht_from_seq(set2,num_timestep)

        zts_1 = zts_1.to(device)
        hts_1 = hts_1.to(device)
        zts_2 = zts_2.to(device)
        hts_2 = hts_2.to(device)

        num = np.arange(0,num_interpolations+2)/(num_interpolations + 1)

        # When t = 1 ,hts = hts_1
        # When t = 0, hts = hts_2
        interpolations = []
        for i in range(0,len(num)):
            t = num[i]
            zts = t * (zts_1 - zts_2) + zts_2
            hts = t * (hts_1 - hts_2) + hts_2
            prediction = self.model.predicts_from_zt_ht(set1, zts, hts)
            prediction = self.solver.preprocess(prediction, reverse=True)
            prediction = prediction.permute(1,0,2,3,4).type(torch.FloatTensor).to(device)
            interpolations.append(prediction)
        set1  = self.solver.preprocess(set1, reverse=True)
        set2  = self.solver.preprocess(set2, reverse=True)
        weight = torch.range(1,num_timestep,1).unsqueeze(1).unsqueeze(2).unsqueeze(3).to(device)
        fig, ax = plt.subplots(1, 4 + num_interpolations, figsize = (20,180))
        truevals = (weight*set2[num_batch, 0:num_timestep, :, :, :]).sum(0)/num_timestep
        ax[0].imshow(self.convert_to_numpy(truevals))
        ax[0].axis('off')
        ax[0].set_title(r"True seq")
        interpolation = (weight*interpolations[0][num_batch, 0:num_timestep, :, :, :]).sum(0)/num_timestep
        ax[1].imshow(self.convert_to_numpy(interpolation))
        ax[1].axis('off')
        ax[1].set_title(r"Reconstruction")
        for i in range(1,num_interpolations+1):
            interpolation = (weight*interpolations[i][num_batch, 0:num_timestep, :, :, :]).sum(0)/num_timestep
            ax[1+i].imshow(self.convert_to_numpy(interpolation))
            ax[1+i].axis('off')
            ax[1+i].set_title(r"Interpolation")


        interpolation = (weight*interpolations[num_interpolations+1][num_batch, 0:num_timestep, :, :, :]).sum(0)/num_timestep
        ax[num_interpolations+2].imshow(self.convert_to_numpy(interpolation))
        ax[num_interpolations+2].axis('off')
        ax[num_interpolations+2].set_title(r"Reconstruction")
        truevals = (weight*set1[num_batch, 0:num_timestep, :, :, :]).sum(0)/num_timestep
        ax[num_interpolations+3].imshow(self.convert_to_numpy(truevals))
        ax[num_interpolations+3].axis('off')
        ax[num_interpolations+3].set_title(r"True seq")
        fig.savefig(self.path +'eval_folder/' + 'interpolations' +  '.pdf', bbox_inches='tight')

    def compute_loss(self, nll, kl, dims, t=10):

        kl_store = kl.data
        nll_store = nll.data
        elbo = -(kl_store+nll_store)

        bits_per_dim_loss = float(-elbo/(np.log(2.)*torch.prod(torch.tensor(dims))*t))
        kl_loss = float(kl_store/t)
        recon_loss = float(nll_store/t)

        return bits_per_dim_loss, kl_loss, recon_loss

    def get_loss(self, model_name, loss_resamples):
        BPD = []
        DKL=[]
        RECON = []

        with torch.no_grad():
          self.model.eval()
          BPD_means = []
          DKL_means = []
          RECON_means = []
          for resample in range(0,loss_resamples):
              BPD = []
              DKL=[]
              RECON = []
              for batch_i, true_image in enumerate(tqdm(self.test_loader, desc="Running", position=0, leave=True)):
                  if self.choose_data=='bair':
                      image = true_image[0].to(device)
                  else:
                      image = true_image.to(device)
                  
                  image = self.solver.preprocess(image)
                  # It doesnt make sense to get loss for longer seqs than trained on, atleast not if need to be compared to the trained loss.
                  imageloss = image[:,:self.n_trained,:,:,:]
                  # Computes eval loss
                  if model_name == "rfn.pt":
                      logdet = 0
                      _, kl, nll = self.model.loss(imageloss, logdet)
                      bits_per_dim_loss, kl_loss, recon_loss = self.compute_loss(nll=nll,
                                                                               kl=kl,
                                                                               dims=imageloss.shape[2:],
                                                                               t=imageloss.shape[1]-1)
                  else:
                      kl, nll = self.model.loss(imageloss)
                      bits_per_dim_loss, kl_loss, recon_loss = self.compute_loss(nll=nll,
                                                                               kl=kl,
                                                                               dims=image.shape[2:],
                                                                               t=image.shape[1]-1)
                  
                    
                  BPD.append(bits_per_dim_loss)
                  DKL.append(kl_loss)
                  RECON.append(recon_loss)
              
              BPD = torch.FloatTensor(BPD)
              DKL = torch.FloatTensor(DKL)
              RECON = torch.FloatTensor(RECON)
              # Find the mean of the bits per dim of one whole data set
              BPD_means.append(BPD)
              DKL_means.append(DKL) 
              RECON_means.append(RECON)
          # Shape  [loss_resamples*len(test_set)]
          BPD_means = torch.stack(BPD_means)
          DKL_means = torch.stack(DKL_means)
          RECON_means = torch.stack(RECON_means)
          print(BPD_means.shape)
          
          BPD_means_mean = BPD_means.mean()
          BPD_means_std = BPD_means.std()
          #CI = BPD_means_std/(loss_resamples**(1/2))
          # Min so far on BAir is 2.66
          print('Mean Loss: '+str(BPD_means_mean.numpy()) + ' Std: '+str(BPD_means_std))
	
    def get_eval_values(self, model_name):
      start_predictions = self.start_predictions

      SSIM_values = []
      PSNR_values = []
      MSE_values = []
      LPIPS_values = []
      BPD = []
      DKL=[]
      RECON = []
      preds = []
      gt = []
      NLL_PROB = []
      NLL_PRI = []
      NLL_PO = []
      AG = [] # Amotization gap.
      SSIM_std_values=[]
      PSNR_std_values=[]
      LPIPS_std_values=[]

      with torch.no_grad():
          self.model.eval()
          for batch_i, true_image in enumerate(tqdm(self.test_loader, desc="Running", position=0, leave=True)):

              SSIM_mean = []
              PSNR_mean = []
              LPIPS_mean = []
              for time in range(0, self.resample):
                  if self.choose_data=='bair':
                      image = true_image[0].to(device)
                  else:
                      image = true_image.to(device)
                  
                  image = self.solver.preprocess(image)
                  image_notchanged = image

                  x_true, predictions = self.model.predict(image, self.n_frames-start_predictions, start_predictions)
                  # Computes eval loss
                  if model_name == "rfn.pt":
                      logdet = 0
                      # It doesnt make sense to get loss for longer seqs than trained on, atleast not if need to be compared to the trained loss.
                      imageloss = image[:,:self.n_trained,:,:,:]
                      _, kl, nll = self.model.loss(imageloss, logdet)
                      bits_per_dim_loss, kl_loss, recon_loss = self.compute_loss(nll=nll,
                                                                               kl=kl,
                                                                               dims=imageloss.shape[2:],
                                                                               t=imageloss.shape[1]-1)


                      if time == 0 and self.extra_plots:# Doesnt makes sense to get these for more then one time. Or i gueess you could if you wanted to.
                          #zts, hts, cts = self.model.get_zt_ht_from_seq(imageloss,3)
                          dims = imageloss.shape[2:]
                          nll_prob = self.model.probability_future(image,start_predictions)/(np.log(2.)*torch.prod(torch.tensor(dims)))
                          _, _, _, averageNLLseq = self.model.reconstruct_elbo_gap(imageloss, sample = False)
                          ## These needs to be saved
                          nllprior = averageNLLseq[0, 1:, :]/(np.log(2.)*torch.prod(torch.tensor(dims)))
                          nllposterior = averageNLLseq[1, 1:, :]/(np.log(2.)*torch.prod(torch.tensor(dims)))
                          amortization_gap = (nllprior - nllposterior)

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

                  #[seq_id, n_frames] = 1 batch
                  if time == 0:
                      psnr_best = psnr
                      ssim_best = ssim
                      mse_best = mse
                      lpips_best = lpips
                      best_preds_ssim = predictions
                  else:
                      psnr_better_id = psnr_best.mean(-1) < psnr.mean(-1)
                      psnr_best[psnr_better_id, :] = psnr[psnr_better_id, :]

                      ssim_better_id = ssim_best.mean(-1) < ssim.mean(-1)
                      ssim_best[ssim_better_id, :] = ssim[ssim_better_id, :]

                      mse_better_id = mse_best.mean(-1) > mse.mean(-1)
                      mse_best[mse_better_id, :] = mse[mse_better_id, :]

                      lpips_better_id = lpips_best.mean(-1) > lpips.mean(-1)
                      lpips_best[lpips_better_id, :] = lpips[lpips_better_id, :]

                      #Get better preds based on SSIM
                      best_preds_ssim[ssim_better_id,:,:,:,:] = predictions[ssim_better_id,:,:,:,:]

                  #save sequence for each sample
                  SSIM_mean.append(ssim)
                  PSNR_mean.append(psnr)
                  LPIPS_mean.append(lpips)

             # [resample, batch, time] Approximate uncertainty of the estimate for each sequence sample
              SSIM_mean=torch.stack(SSIM_mean)
              # To [batch, time]
              SSIM_mean = SSIM_mean.mean(0)
              #SSIM_std_mean = SSIM_std.std(0)/np.sqrt(SSIM_std.shape[0])
              SSIM_std_values.append(SSIM_mean)

              PSNR_mean = torch.stack(PSNR_mean).mean(0)
              #PSNR_std_mean = PSNR_std.std(0)/np.sqrt(PSNR_std.shape[0])
              PSNR_std_values.append(PSNR_mean)

              LPIPS_mean = torch.stack(LPIPS_mean).mean(0)
              #LPIPS_std_mean = LPIPS_std.std(0)/np.sqrt(LPIPS_std.shape[0])
              LPIPS_std_values.append(LPIPS_mean)

              #Store values
              preds.append(best_preds_ssim)
              gt.append(ground_truth)
              SSIM_values.append(ssim_best)
              PSNR_values.append(psnr_best)
              MSE_values.append(mse_best)
              LPIPS_values.append(lpips_best)
              BPD.append(bits_per_dim_loss)
              DKL.append(kl_loss)
              RECON.append(recon_loss)

              if model_name == "rfn.pt" and self.extra_plots:
                  NLL_PROB.append(nll_prob)
                  NLL_PRI.append(nllprior)
                  NLL_PO.append(nllposterior)
                  AG.append(amortization_gap)


          if model_name == "rfn.pt" and self.extra_plots:
              NLL_PROB = torch.cat(NLL_PROB,dim = 0)
              NLL_PRI = torch.cat(NLL_PRI, dim = 1)
              NLL_PO = torch.cat(NLL_PO, dim = 1)
              AG  = torch.cat(AG, dim = 1)
              print("BPP Prior: ", NLL_PRI.mean())
              print("BPP Posterior: ", NLL_PO.mean())
              print('Amortization gap: '+str(AG.mean()))

          # Shape: [seq_id, n_frames]
          PSNR_values = torch.cat(PSNR_values)
          MSE_values = torch.cat(MSE_values)
          SSIM_values = torch.cat(SSIM_values)
          LPIPS_values = torch.cat(LPIPS_values)

          #get mean uncertainty over batches
          SSIM_std_values = torch.cat(SSIM_std_values, 0)
          PSNR_std_values = torch.cat(PSNR_std_values, 0)
          LPIPS_std_values = torch.cat(LPIPS_std_values, 0)

          # Sort gt and preds based on highest to lowest SSIM values
          ordered = torch.argsort(SSIM_values.mean(-1), descending=True)
          preds = torch.cat(preds)[ordered,...]
          gt = torch.cat(gt)[ordered,...]

          # Shape: [one avg. loss for each minibatch]
          BPD = torch.FloatTensor(BPD)
          DKL = torch.FloatTensor(DKL)
          RECON = torch.FloatTensor(RECON)

          if model_name == "rfn.pt" and self.extra_plots:
            self.plot_elbo_gap(image_notchanged)
            self.plot_prob_of_t(NLL_PROB)


      if self.debug_plot:
          ns = self.num_samples_to_plot
          nf = 10 #time rollouts
          self.plot_samples(predictions[:,0:nf,...].byte(), ground_truth[:,0:nf,...].byte(), name="random_samples_ssim")
          self.plot_samples(preds[:ns,0:nf,...].byte(), gt[:ns,0:nf,...].byte(),
                            name="best_samples", eval_score = SSIM_values[ordered,...][:ns,0:nf,...], set_top=1.1)
          self.plot_samples(preds[-ns:,0:nf,...].byte(), gt[-ns:,0:nf,...].byte(),
                            name="worst_samples", eval_score = SSIM_values[ordered,...][-ns:,0:nf,...], set_top=1.1)
      return MSE_values, PSNR_values, SSIM_values, LPIPS_values, BPD, DKL, RECON, SSIM_std_values, PSNR_std_values, LPIPS_std_values

    def test_temp_values(self, path, label_names, experiment_names):
        n_train = self.n_trained
        markersize = 5
        fig, ax = plt.subplots(1, 3 ,figsize = (19,5))# figsize = (19,7)
        fig2, ax2 = plt.subplots(1, 3,figsize = (19,5))
        fig3, ax3 = plt.subplots(1, 3,figsize = (19,5))
        fig.subplots_adjust(hspace=1, wspace=0.17)
        fig2.subplots_adjust(hspace=1, wspace=0.17)
        fig3.subplots_adjust(hspace=1, wspace=0.17)
        markers = ["o", "v", "x", "*", "^", "s", "H", "P", "X","1","2","3"]

        for temp,i in zip(self.temperatures,range(0,len(self.temperatures))):
            lname = '$T=' + " "+ str(temp)+'$'
            temp_id = "/t" + str(temp).replace('.','')
            eval_dict = torch.load(path + experiment_names[0] + '/eval_folder/'+temp_id+'evaluations.pt')

            mark = markers[i]
            alpha = 0.05 # For quantiles
            alpha_CI = 0.2 # For the transparentcy of the plotted CI

            SSIM  = eval_dict['SSIM_values']  # This is max values
            PSNR  = eval_dict['PSNR_values']    # This is max values
            LPIPS  = eval_dict['LPIPS_values']  # This is max values
            SSIM_std_mean = eval_dict['SSIM_std_mean'] # This is this is mean of resample. So the mean of [resample, batch, time].mean(0) to [batch,time] 
            PSNR_std_mean = eval_dict['PSNR_std_mean']
            LPIPS_std_mean = eval_dict['LPIPS_std_mean']
            print("Temperature is set to " + str(eval_dict['temperature']) + " for experiment " +lname)

            y = SSIM.mean(0).numpy()
            xaxis = np.arange(0+self.n_conditions, len(y)+self.n_conditions)
            conf_std = 1.96 * np.std(SSIM.numpy(),0)/np.sqrt(np.shape(SSIM.numpy())[0])
            ax[0].plot(xaxis, y, label = lname, marker=mark, markersize=markersize)
            ax[0].fill_between(xaxis, y-conf_std, y+conf_std, alpha=alpha_CI)

            y = PSNR.mean(0).numpy()
            twostd = 1.96 * np.std(PSNR.numpy(),0)/np.sqrt(np.shape(PSNR.numpy())[0])
            ax[1].plot(xaxis,y,label = lname, marker=mark, markersize=markersize)
            ax[1].fill_between(xaxis, y-twostd, y+twostd, alpha=alpha_CI)

            y = LPIPS.mean(0).numpy()
            twostd = 1.96 * np.std(LPIPS.numpy(),0)/np.sqrt(np.shape(LPIPS.numpy())[0])
            ax[2].plot(xaxis,y,label = lname, marker=mark, markersize=markersize)
            ax[2].fill_between(xaxis, y-twostd, y+twostd, alpha=alpha_CI)

            y = np.median(SSIM.numpy(),0)
            ax2[0].plot(xaxis, y, label = lname, marker=mark,markersize=markersize)
            ax2[0].fill_between(xaxis,
              np.quantile(SSIM.numpy(), alpha/2, axis = 0),
              np.quantile(SSIM.numpy(), 1-alpha/2, axis = 0), alpha=alpha_CI)

            y = np.median(PSNR.numpy(),0)
            ax2[1].plot(xaxis, y, label = lname, marker=mark, markersize=markersize)
            ax2[1].fill_between(xaxis,
              np.quantile(PSNR.numpy(), alpha/2, axis = 0),
              np.quantile(PSNR.numpy(), 1-alpha/2, axis = 0), alpha=alpha_CI)

            y = np.median(LPIPS.numpy(),0)
            ax2[2].plot(np.arange(0, len(y)), y, label = lname, marker=mark,
               markersize=markersize)
            ax2[2].fill_between(np.arange(0, len(y)),
              np.quantile(LPIPS.numpy(), alpha/2, axis = 0),
              np.quantile(LPIPS.numpy(), 1-alpha/2, axis = 0), alpha=alpha_CI)
            
            y = SSIM_std_mean.mean(0).numpy()
            conf_std = 1.96 * np.std(SSIM_std_mean.numpy(),0)/np.sqrt(np.shape(SSIM_std_mean.numpy())[0])
            ax3[0].errorbar(xaxis, y, yerr=conf_std,label = lname)

            y = PSNR_std_mean.mean(0).numpy()
            conf_std = 1.96 * np.std(PSNR_std_mean.numpy(),0)/np.sqrt(np.shape(PSNR_std_mean.numpy())[0])
            ax3[1].errorbar(xaxis, y, yerr=conf_std, label = lname)

            #y = LPIPS.mean(0).numpy()
            y = LPIPS_std_mean.mean(0).numpy()
            conf_std = 1.96 * np.std(LPIPS_std_mean.numpy(),0)/np.sqrt(np.shape(LPIPS_std_mean.numpy())[0])
            ax3[2].errorbar(xaxis, y, yerr=conf_std, label = lname)
        fontsizeaxislabel = 25 
        fontsizelegend = 17
        fontsizetitle = 25
        fontsizeticks = 23 
        labelpad = 5
        
        ax[0].set_ylabel(r'score', fontsize = fontsizeaxislabel, labelpad = labelpad)
        ax[0].set_xlabel(r'$t$', fontsize = fontsizeaxislabel)
        ax[0].set_title(r'Max. SSIM with 95$\%$ CI',fontsize = fontsizetitle)
        ax[0].axvline(x=n_train, color='k', linestyle='--')
        #ax[0].legend(fontsize = fontsizelegend)

        
        axnow = ax[0]
        ax[0].grid()
        for tick in axnow.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsizeticks) 
        for tick in axnow.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsizeticks) 
            
        

        ax[1].set_xlabel(r'$t$', fontsize = fontsizeaxislabel)
        ax[1].set_title(r'Max. PSNR with 95$\%$ CI', fontsize = fontsizetitle)
        ax[1].axvline(x=n_train, color='k', linestyle='--')
        #ax[1].legend(fontsize = fontsizelegend)

        axnow = ax[1]
        ax[1].grid()
        for tick in axnow.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsizeticks) 
        for tick in axnow.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsizeticks) 
        

        ax[2].set_xlabel(r'$t$', fontsize = fontsizeaxislabel)
        ax[2].set_title(r'Min. LPIPS with 95$\%$ CI', fontsize = fontsizetitle)
        ax[2].axvline(x=n_train, color='k', linestyle='--')
        #ax[2].legend(fontsize = fontsizelegend)
        ax[2].legend(bbox_to_anchor=(0.1,-0.23), loc="lower left", 
                bbox_transform=fig.transFigure, ncol=6, fontsize = fontsizelegend)
        
        axnow = ax[2]
        ax[2].grid()
        for tick in axnow.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsizeticks) 
        for tick in axnow.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsizeticks) 
        

        ax2[0].set_ylabel(r'score',fontsize = fontsizeaxislabel,labelpad = labelpad)
        ax2[0].set_xlabel(r'$t$', fontsize = fontsizeaxislabel)
        ax2[0].set_title(r'Max. SSIM with 95$\%$ quantiles',fontsize = fontsizetitle)
        ax2[0].axvline(x=n_train, color='k', linestyle='--')
        #ax2[0].legend(fontsize = fontsizelegend)

        ax = ax2[0]
        ax2[0].grid()
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsizeticks) 
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsizeticks) 
        

        ax2[1].set_xlabel(r'$t$',fontsize = fontsizeaxislabel)
        ax2[1].set_title(r'Max. PSNR with 95$\%$ quantiles',fontsize = fontsizetitle)
        ax2[1].axvline(x=n_train, color='k', linestyle='--')
        #ax2[1].legend(fontsize = fontsizelegend)

        ax = ax2[1]
        ax2[1].grid()
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsizeticks) 
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsizeticks) 
        

        ax2[2].set_xlabel(r'$t$',fontsize = fontsizeaxislabel)
        ax2[2].set_title(r'Min. LPIPS with 95$\%$ quantiles',fontsize = fontsizetitle)
        ax2[2].axvline(x=n_train, color='k', linestyle='--')
        ax2[2].legend(bbox_to_anchor=(0.1,-0.23), loc="lower left", 
                bbox_transform=fig2.transFigure, ncol=6, fontsize = fontsizelegend)
        #ax2[2].legend(fontsize = fontsizelegend)

        ax = ax2[2]
        ax2[2].grid()
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsizeticks) 
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsizeticks) 
        

        ax3[0].set_ylabel(r'score',fontsize = fontsizeaxislabel,labelpad = labelpad)
        ax3[0].set_xlabel(r'$t$',fontsize = fontsizeaxislabel)
        ax3[0].set_title(r'Avg. SSIM',fontsize = fontsizetitle)
        ax3[0].axvline(x=n_train, color='k', linestyle='--')
        #ax3[0].legend(fontsize = fontsizelegend)

        ax = ax3[0]
        ax3[0].grid()
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsizeticks) 
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsizeticks) 
        

        ax3[1].set_xlabel(r'$t$',fontsize = fontsizeaxislabel)
        ax3[1].set_title(r'Avg. PSNR',fontsize = fontsizetitle)
        ax3[1].axvline(x=n_train, color='k', linestyle='--')
        #ax3[1].legend(fontsize = fontsizelegend)

        ax = ax3[1]
        ax3[1].grid()
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsizeticks) 
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsizeticks) 
        

        ax3[2].set_xlabel(r'$t$',fontsize = fontsizeaxislabel)
        ax3[2].set_title(r'Avg. LPIPS',fontsize = fontsizetitle)
        ax3[2].axvline(x=n_train, color='k', linestyle='--')
        ax3[2].legend(bbox_to_anchor=(0.1,-0.23), loc="lower left", 
                bbox_transform=fig3.transFigure, ncol=6, fontsize = fontsizelegend)
        ax = ax3[2]
        ax3[2].grid()
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsizeticks) 
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsizeticks) 
        
        
        fig.savefig(path + experiment_names[0] + '/eval_folder/eval_plots_max_temp.pdf', bbox_inches='tight')
        fig2.savefig(path + experiment_names[0] +  '/eval_folder/eval_plots_max_median_temp.pdf', bbox_inches='tight')
        fig3.savefig(path + experiment_names[0] +  '/eval_folder/eval_plots_mean_mean_temp.pdf', bbox_inches='tight')


    def get_fvd_values(self, model_name, n_predicts):
      start_predictions = self.start_predictions
      #self.create_loaders(self, drop_last=True):
      FVD_values = []
      test_loader=self.create_loaders(drop_last=False)
      with torch.no_grad():
          self.model.eval()
          for _ in range(0,2):
            preds = []
            gts = []
          
            for batch_i, true_image in enumerate(tqdm(test_loader, desc="Running", position=0, leave=True)):
  
                if self.choose_data=='bair':
                    image = true_image[0].to(device)
                else:
                    image = true_image.to(device)
                image = self.solver.preprocess(image)
                
                cur_bs = image.shape[0]
                if cur_bs < self.batch_size:
                  pad=torch.zeros(self.batch_size-cur_bs, *image.shape[1:]).to(device)
                  full_image = torch.cat((image, pad),0)
                  _, predictions = self.model.predict(full_image, n_predicts, start_predictions)
                  predictions = predictions[:,:cur_bs,...]
                else:
                  _, predictions = self.model.predict(image, n_predicts, start_predictions)
                  
                image  = self.solver.preprocess(image, reverse=True)
                predictions  = self.solver.preprocess(predictions, reverse=True)
  
                # should be [t, b, c, h, w]
                ground_truth = image[:, (start_predictions):(n_predicts+start_predictions),:,:,:].cpu()
                predictions = predictions.permute(1,0,2,3,4).cpu()

                gts.append(ground_truth)
                preds.append(predictions)
              
          
            ground_truths_fvd=torch.cat((gts),0).permute(1,0,2,3,4)
            predictions_fvd=torch.cat((preds),0).permute(1,0,2,3,4)


            FVD = fvd(ground_truths_fvd, predictions_fvd)
            FVD_values.append(FVD)

      
      if len(FVD_values)>1:
        FVD_mean=np.mean(FVD_values)
        FVD_std=np.std(FVD_values)
      else:
        FVD_mean = FVD_values[0]
        FVD_std = 0
      
      print(FVD_mean)
      print(FVD_std)
      
      return FVD_mean, FVD_std
    def minmax_scale(self,x):
      x = (x - x.min()) / (x.max() - x.min())
      return x
    
    def param_plots(self, path, n_conditions):

        print("Init parameter analysis")
        seq_len=30
        plt.rcParams['image.cmap']='gray'
        param_test_set = MovingMNIST_synchronized(False, 'Mnist', seq_len=seq_len,
                                                  num_digits=2,
                                                  image_size=self.image_size, digit_size=self.digit_size,
                                                  deterministic=False, three_channels = False,
                                                  step_length=self.step_length, normalize = False,
                                                  make_target = False, seed = None)

        te_split_len = 400
        param_test_set = torch.utils.data.random_split(param_test_set,
                                [te_split_len, len(param_test_set)-te_split_len])[0]
        
        
        
        test_loader = DataLoader(param_test_set, batch_size=self.batch_size,
                                 num_workers=self.num_workers, shuffle=True, drop_last=True)
        mu_p_params=[]
        std_p_params=[]
        mu_q_params=[]
        std_q_params=[]
        mu_flow_params=[]
        std_flow_params=[]
        # DO NOT DELETE TEMP, OTHERWISE PARAM_TEST_SET WONT UPDATE BOUNDARIES
        temp = next(iter(param_test_set)) #hack
        ##########################################################
        digit_one = list(np.where(param_test_set.dataset.hit_boundary==1)[0])
        digit_two = list(np.where(param_test_set.dataset.hit_boundary==2)[0])
        
        with torch.no_grad():
          self.model.eval()
          for batch_i, true_image in enumerate(tqdm(test_loader, desc="Running", position=0, leave=True)):
              
              #fig, ax = plt.subplots(2,true_image.shape[1], figsize=(true_image.shape[1],2))
              #for i in range(0,true_image.shape[1]):
              #  ax[0,i].imshow(true_image[0,i,...].permute(1,2,0))
              #  ax[1,i].imshow(true_image[2,i,...].permute(1,2,0))
              #fig.savefig("/work1/s146996/work1/test_if_works"+str(batch_i)+".pdf", bbox_inches='tight')
            
              if self.choose_data=='bair':
                image = true_image[0].to(device)
              else:
                image = true_image.to(device)
              image = self.solver.preprocess(image)
              
              mu_p, std_p, mu_q, std_q, mu_flow, std_flow, prediction = self.model.param_analysis(x=image,
                                                                                      n_conditions=n_conditions,
                                                                                      n_predictions=seq_len-n_conditions)
              
              t,b,c,h,w = mu_p.shape
              tf,bf,cf,hf,wf = mu_flow.shape
        
              
              mu_p_params.append(mu_p.sum([2,3,4]))
              std_p_params.append(std_p.sum([2,3,4]))
              mu_q_params.append(mu_q.sum([2,3,4]))
              std_q_params.append(std_q.sum([2,3,4]))
              mu_flow_params.append(mu_flow.sum([2,3,4]))
              std_flow_params.append(std_flow.sum([2,3,4]))
        
        mu_p_params=torch.stack(mu_p_params).mean([0,2]).cpu().numpy()
        std_p_params=torch.stack(std_p_params).mean([0,2]).cpu().numpy()
        mu_q_params=torch.stack(mu_q_params).mean([0,2]).cpu().numpy()
        std_q_params=torch.stack(std_q_params).mean([0,2]).cpu().numpy()
        mu_flow_params=torch.stack(mu_flow_params).mean([0,2]).cpu().numpy()
        std_flow_params=torch.stack(std_flow_params).mean([0,2]).cpu().numpy()
        
        mu_p_params=self.minmax_scale(mu_p_params)
        std_p_params=self.minmax_scale(std_p_params)
        mu_q_params=self.minmax_scale(mu_q_params)
        std_q_params=self.minmax_scale(std_q_params)
        mu_flow_params=self.minmax_scale(mu_flow_params)
        std_flow_params=self.minmax_scale(std_flow_params)
        
        b,t,c,h,w = true_image.shape
        
        
        test = self.solver.preprocess(image[:,1:11,...].permute(0,1,2,4,3), reverse=True)[0].reshape(-1,w).transpose(0, 1).cpu()
        test_pred = self.solver.preprocess(prediction[:,1:11,...].permute(0,1,2,4,3), reverse=True)[0].reshape(-1,w).transpose(0, 1).cpu()
        test1 = self.solver.preprocess(image[:,11:21,...].permute(0,1,2,4,3), reverse=True)[0].reshape(-1,w).transpose(0, 1).cpu()
        test_pred1 = self.solver.preprocess(prediction[:,11:21,...].permute(0,1,2,4,3), reverse=True)[0].reshape(-1,w).transpose(0, 1).cpu()
        test2 = self.solver.preprocess(image[:,21:29,...].permute(0,1,2,4,3), reverse=True)[0].reshape(-1,w).transpose(0, 1).cpu()
        test_pred2 = self.solver.preprocess(prediction[:,21:29,...].permute(0,1,2,4,3), reverse=True)[0].reshape(-1,w).transpose(0, 1).cpu()
        test=torch.cat((test, test_pred),0)
        test1=torch.cat((test1, test_pred1),0)
        test2=torch.cat((test2, test_pred2),0)
        fig, ax = plt.subplots(2, 1 ,figsize = (1*10, 2*4))
        #fig2, ax2 = plt.subplots(figsize = (10,10), gridspec_kw={"hspace":0.0001, "wspace":0.0001})
        fig2, ax2 = plt.subplots(3,1,figsize = (1*5,3*5),gridspec_kw={"hspace":0.01, "wspace":0.001, "top":0.2})
        
        names = [r"$\mu_{prior}$",r"$\sigma_{prior}$",
         r"$\mu_{posterior}$", r"$\sigma_{posterior}$",
         r"$\mu_{base dist}$", r"$\sigma_{base dist}$" ]

        #alpha = 0.05
        xaxis = np.arange(1, seq_len, 1)
        seq_ax = [1,seq_len-1]
        ax[0].plot(xaxis,mu_p_params, label=names[0])
        ax[0].plot(xaxis,mu_q_params, label=names[2])
        ax[0].plot(xaxis,mu_flow_params, label=names[4])
        ax[0].set_xlim([seq_ax[0], seq_ax[1]])

        
        ax[1].plot(xaxis,std_p_params, label=names[1])
        ax[1].plot(xaxis,std_q_params, label=names[3])
        ax[1].plot(xaxis,std_flow_params, label=names[5])
        ax[1].set_xlim([seq_ax[0], seq_ax[1]])
        
        ax2[0].imshow(test.cpu().numpy())
        ax2[1].imshow(test1.cpu().numpy())
        ax2[2].imshow(test2.cpu().numpy())
        ax2[0].axis("off")
        ax2[1].axis("off")
        ax2[2].axis("off")
        
        
        ax[0].set_ylabel(r'Average', fontsize=15)
        ax[1].set_ylabel(r'Average', fontsize=15)
        
        ax[0].set_xlabel(r'$t$', fontsize=15)
        ax[1].set_xlabel(r'$t$', fontsize=15)
        for i in range(0,2):
            for k1 in digit_one:
                ax[i].axvline(x=(k1+1), color='r', linestyle='--', linewidth=1)
            for k2 in digit_two:
                ax[i].axvline(x=(k2+1), color='b', linestyle='--', linewidth=1)
            ax[i].legend(fontsize=15)
            #ax[i].grid()
            plt.xticks(fontsize=10)

        fig2.savefig(path + '/parameter_analysis_mnist_plots2.png', bbox_inches='tight')
        fig.savefig(path + '/parameter_analysis2.png', bbox_inches='tight')
        print("Parameter analysis has finished")

    def plot_long_t(self,  model_name):
      if model_name == 'rfn.pt':
        self.model.eval()
        true_image = next(iter(self.test_loader))
        if self.choose_data=='bair':
             image = true_image[0].to(device)
        else:
             image = true_image.to(device)    
        image = self.solver.preprocess(image, reverse=False)
        
        #Long predictions
        conditions, predictions = self.model.predict(image, 80, 5)
        conditions  = self.solver.preprocess(conditions, reverse=True)
        predictions  = self.solver.preprocess(predictions, reverse=True)
        t_list = [3,4,9,19,29,39,49,59,69]
        t_length = len(t_list)
        t_seq = torch.cat([conditions,predictions],0)
        fig, ax = plt.subplots(5, t_length, gridspec_kw = {'wspace':0.06, 'hspace':0}, figsize=(t_length, 5))
        plt.subplots_adjust(wspace=0.06, hspace=0)
        f_size = 20
        
        for k in range(0, 5):
          for i in range(0, t_length): 
            ax[k,i].imshow(self.convert_to_numpy(t_seq[t_list[i], k, :, :, :]))
            if i <2:
              ax[k,i].patch.set_edgecolor('red')  
            else:
              ax[k,i].patch.set_edgecolor('green')  
            #ax[k,i].axis("off")
            ax[k,i].set_yticks([])
            ax[k,i].set_xticks([])
            ax[k,i].patch.set_linewidth('3')
            if k == 0:
              ax[k, i].set_title(r"$t={}$".format(int(t_list[i]+1)), fontsize=f_size)
        fig.savefig(self.path +'eval_folder/' + "plot_long_t" +  '.pdf', bbox_inches="tight")
        plt.close(fig)
      else:
        print("needs to be a RFN.pt model")
        
    def plot_temp(self,  model_name):
      if model_name == 'rfn.pt':
        self.model.eval()
        
        true_image = next(iter(self.test_loader))
        if self.choose_data=='bair':
             image = true_image[0].to(device)
        else:
             image = true_image.to(device)    
        image = self.solver.preprocess(image, reverse=False)
        
        pred_list = []
        #9
        temperatures = [0.025, 0.3, 0.5, 0.6, 0.7, 1]
        n_temps = len(temperatures)
        n_preds = 8
        for i in range(0, n_temps):
          self.model.temperature = temperatures[i]
          conditions, predictions = self.model.predict(image, n_preds, 5)
          conditions  = self.solver.preprocess(conditions, reverse=True)
          predictions  = self.solver.preprocess(predictions, reverse=True)
          pred_list.append(predictions[:,0,:,:,:])
        pred_list = torch.stack(pred_list, 1)
        f_size = 17
        
        fig, ax = plt.subplots(n_temps, n_preds, gridspec_kw = {'wspace':0, 'hspace':0}, figsize=(n_preds, n_temps))
        for k in range(0, n_temps):
          for i in range(0, n_preds): 
            ax[k,i].imshow(self.convert_to_numpy(pred_list[i, k, :, :, :])) 
            ax[k,i].set_yticks([])
            ax[k,i].set_xticks([])
            plt.subplots_adjust(wspace=0, hspace=0)
            if i == 0:
              ax[k, i].set_ylabel(r"$T={}$".format(float(temperatures[k])), fontsize=f_size)
        fig.tight_layout()
        fig.savefig(self.path +'eval_folder/' + "plot_temp_samples" +  '.pdf', )
        plt.close(fig)
      else:
        print("needs to be a RFN.pt model")
        
    def plot_temp_kl(self,  model_name):
      if model_name == 'rfn.pt':
        self.model.eval()
        
        true_image = next(iter(self.test_loader))
        if self.choose_data=='bair':
             image = true_image[0].to(device)
        else:
             image = true_image.to(device)    
        image = self.solver.preprocess(image, reverse=False)
        
        pred_list = []
        #9
        temperatures = [0.025, 0.3, 0.5, 0.6, 0.7, 1]
        n_temps = len(temperatures)
        n_preds = 8
        for i in range(0, n_temps):
          
          conditions, predictions = self.model.predict(image, n_preds, 5)
          conditions  = self.solver.preprocess(conditions, reverse=True)
          predictions  = self.solver.preprocess(predictions, reverse=True)
          pred_list.append(predictions[:,0,:,:,:])
        pred_list = torch.stack(pred_list, 1)
        f_size = 17
        
        fig, ax = plt.subplots(n_temps, n_preds, gridspec_kw = {'wspace':0, 'hspace':0}, figsize=(n_preds, n_temps))
        for k in range(0, n_temps):
          for i in range(0, n_preds): 
            ax[k,i].imshow(self.convert_to_numpy(pred_list[i, k, :, :, :])) 
            ax[k,i].set_yticks([])
            ax[k,i].set_xticks([])
            plt.subplots_adjust(wspace=0, hspace=0)
            if i == 0:
              ax[k, i].set_ylabel(r"$T={}$".format(float(temperatures[k])), fontsize=f_size)
        fig.tight_layout()
        fig.savefig(self.path +'eval_folder/' + "plot_temp_samples" +  '.pdf', )
        plt.close(fig)
      else:
        print("needs to be a RFN.pt model")
   
    def plot_diversity(self,  model_name):
      if model_name == 'rfn.pt':
        #suggestion to improvement, add ground truth as well.
        self.model.eval()
        
        true_image = next(iter(self.test_loader))
        if self.choose_data=='bair':
             image = true_image[0].to(device)
        else:
             image = true_image.to(device)    
        image = self.solver.preprocess(image, reverse=False)
        
        pred_list = []
        #9
        n_resamples = 3
        n_preds = 4
        
        b,t,c,h,w = image.shape
        for i in range(0, n_resamples):
          conditions, predictions = self.model.predict(image, 25, 5)
          conditions  = self.solver.preprocess(conditions, reverse=True)
          predictions  = self.solver.preprocess(predictions, reverse=True)
          pred_list.append(predictions[:,0:2,:,:,:])
        pred_list = torch.stack(pred_list, 1).view(25,-1,c,h,w)
        
        get_pred_list = [3,7,12,20]
        f_size=17
        fig, ax = plt.subplots(n_resamples, n_preds, gridspec_kw = {'wspace':0.0001, 'hspace':0}, figsize=(n_preds, n_resamples))
        plt.subplots_adjust(wspace=0.0001, hspace=0)
        fig2, ax2 = plt.subplots(n_resamples, n_preds, gridspec_kw = {'wspace':0.0001, 'hspace':0}, figsize=(n_preds, n_resamples))
        plt.subplots_adjust(wspace=0.0001, hspace=0)
        for k in range(0, n_resamples):
          for i in range(0,n_preds):
              ax[k,i].imshow(self.convert_to_numpy(pred_list[get_pred_list[i], (k+1)*2-1, :, :, :])) 
              ax[k,i].set_yticks([])
              ax[k,i].set_xticks([])
              ax2[k,i].imshow(self.convert_to_numpy(pred_list[get_pred_list[i], (k)*2, :, :, :])) 
              ax2[k,i].set_yticks([])
              ax2[k,i].set_xticks([])
              if k == 0:
                ax[k, i].set_title(r"$t={}$".format(get_pred_list[i]+1+5), fontsize=f_size)
                ax2[k, i].set_title(r"$t={}$".format(get_pred_list[i]+1+5), fontsize=f_size)
        fig.savefig(self.path +'eval_folder/' + "plot_diversity_1" +  '.pdf', bbox_inches="tight")
        plt.close(fig)

        fig2.savefig(self.path +'eval_folder/' + "plot_diversity_2" +  '.pdf', bbox_inches="tight")
        plt.close(fig2)
      else:
        print("needs to be a RFN.pt model")
        
    def plot_random_samples(self,  model_name):
      if model_name == 'rfn.pt':
        self.model.eval()
        true_image = next(iter(self.test_loader))
        if self.choose_data=='bair':
             image = true_image[0].to(device)
        else:
             image = true_image.to(device)    
        image = self.solver.preprocess(image, reverse=False)
        
        #Long predictions
        conditions, predictions = self.model.predict(image, 10, 3)
        conditions  = self.solver.preprocess(conditions, reverse=True)
        predictions  = self.solver.preprocess(predictions, reverse=True)
        t_list = [0,1,2,3,4,5,6,7,8,9,10,11,12]
        t_length = len(t_list)
        t_seq = torch.cat([conditions,predictions],0)
        fig, ax = plt.subplots(5, t_length, gridspec_kw = {'wspace':0.06, 'hspace':0}, figsize=(t_length, 5))
        plt.subplots_adjust(wspace=0.06, hspace=0)
        f_size = 17
        
        for k in range(0, 5):
          for i in range(0, t_length): 
            ax[k,i].imshow(self.convert_to_numpy(t_seq[t_list[i], k, :, :, :]))
            if i <3:
              ax[k,i].patch.set_edgecolor('red')  
            else:
              ax[k,i].patch.set_edgecolor('green')  
            ax[k,i].set_yticks([])
            ax[k,i].set_xticks([])
            ax[k,i].patch.set_linewidth('3')
            if k == 0:
              ax[k, i].set_title(r"$t={}$".format(int(t_list[i]+1)), fontsize=f_size)
        fig.savefig(self.path +'eval_folder/' + "plot_rollouts" +  '.pdf', bbox_inches="tight")
        plt.close(fig)
      else:
        print("needs to be a RFN.pt model")
