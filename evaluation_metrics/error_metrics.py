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
plt.rcParams.update({'text.usetex': True})
from data_generators import MovingMNIST
from data_generators import PushDataset
device = set_gpu(True)
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from tqdm import tqdm
from evaluation_metrics.FVD_score import fvd

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

    def plot_samples(self, predictions, true_image, name="samples", eval_score = None):
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
            if eval_score != None:
                ax[2*(k)+1,i].set_title(f"{float(eval_score[k,i]):.3f}")
            if i == 0:
                ax[2*(k)+1,i].set_yticks([])
                ax[2*(k)+1,i].set_xticks([])
                ax[2*(k)+1,i].set_ylabel("Prediction")
            else:
                ax[2*(k)+1,i].axis("off")
      fig.savefig(self.path +'eval_folder/' + name +  '.png', bbox_inches='tight') #dpi=fig.get_dpi()*2)
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
        fig, ax = plt.subplots(7, time_steps , figsize = (2*time_steps, 2*6))
        for i in range(0, time_steps):
            ax[0,i].imshow(self.convert_to_numpy(image[0, i, :, :, :]))
            ax[0,i].set_title(r"True Image")
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
        plt.xlabel(r"Frame number")
        plt.ylabel(r"Sum of KLD")
        plt.subplot(8, 1, 8)
        plt.bar(np.arange(time_steps)-0.15, averageNLLseq[0, :, 0], align='center', width=0.3,label = 'Prior')
        plt.bar(np.arange(time_steps)+0.15, averageNLLseq[1, :, 0], align='center', width=0.3,label = 'Posterior')
        plt.xlim((0-0.5, time_steps-0.5))
        low = min(min(averageNLLseq[0, 1:, 0]),min(averageNLLseq[1, 1:, 0]))
        high = max(max(averageNLLseq[0, 1:, 0]),max(averageNLLseq[1, 1:, 0]))
        plt.ylim([low-0.5*(high-low), high+0.5*(high-low)])
        plt.xticks(range(0, time_steps), range(0, time_steps))
        plt.xlabel(r"Frame number")
        plt.ylabel(r"bits dim nll")
        plt.legend()
        fig.savefig(self.path + 'eval_folder/KLDdiagnostic' + '.png', bbox_inches='tight')
        plt.close(fig)

    def plot_prob_of_t(self, probT):
        plt.figure()
        xaxis = np.arange(self.n_conditions, probT.shape[2] + self.n_conditions)
        y = probT[:,0,:].mean(0)
        plt.plot(xaxis, probT[:,0,:].mean(0), label = 'Prior')
        # Almost no difference between prior and posterior, so only plot prior
        #plt.plot(xaxis, probT[:,1,:].mean(0), label = 'Posterior')
        #plt.legend()
        conf_std = 1.96 * np.std(probT[:,0,:].numpy(),0)/np.sqrt(np.shape(probT[:,0,:].numpy())[0])
        plt.fill_between(xaxis, y-conf_std, y+conf_std, alpha=.1)

        plt.ylabel(r"Bits per pixel")
        plt.xlabel(r"Frame number:$X_{t}$")
        plt.title(r'$P(X_{'+str(self.n_conditions)+'}= X_t \mid X_{<'+str(self.n_conditions)+'})$')
        plt.grid()

        plt.savefig(self.path + 'eval_folder/bpp_sequence' + '.png', bbox_inches='tight')

        plt.close()

    def get_interpolations(self):
        ## Only works for RFN when trained on one digit.
        ## Two dataset with two different seeds 
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
              SSIM_std = []
              PSNR_std = []
              LPIPS_std = []
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


                      if time == 0:# Doesnt makes sense to get these for more then one time. Or i gueess you could if you wanted to.
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

                  #save sequence mean for each sample
                  SSIM_std.append(ssim.mean(0))
                  PSNR_std.append(psnr.mean(0))
                  LPIPS_std.append(lpips.mean(0))

             # [resample, 24 means] Approximate uncertainty of the estimate for each sequence sample
              SSIM_std=torch.stack(SSIM_std)
              SSIM_std_mean = SSIM_std.std(0)/np.sqrt(SSIM_std.shape[0])
              SSIM_std_values.append(SSIM_std_mean)

              PSNR_std=torch.stack(PSNR_std)
              PSNR_std_mean = PSNR_std.std(0)/np.sqrt(PSNR_std.shape[0])
              PSNR_std_values.append(PSNR_std_mean)

              LPIPS_std=torch.stack(LPIPS_std)
              LPIPS_std_mean = LPIPS_std.std(0)/np.sqrt(LPIPS_std.shape[0])
              LPIPS_std_values.append(LPIPS_std_mean)

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
          SSIM_std_values = torch.stack(SSIM_std_values).mean(0)
          PSNR_std_values = torch.stack(PSNR_std_values).mean(0)
          LPIPS_std_values = torch.stack(LPIPS_std_values).mean(0)

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
          self.plot_samples(predictions.byte(), ground_truth.byte(), name="random_samples")
          self.plot_samples(preds[:ns,...].byte(), gt[:ns,...].byte(),
                            name="best_samples", eval_score = SSIM_values[ordered,...][:ns,...])
          self.plot_samples(preds[-ns:,...].byte(), gt[-ns:,...].byte(),
                            name="worst_samples", eval_score = SSIM_values[ordered,...][-ns:,...])
      return MSE_values, PSNR_values, SSIM_values, LPIPS_values, BPD, DKL, RECON, SSIM_std_values, PSNR_std_values, LPIPS_std_values

    def test_temp_values(self, path, label_names, experiment_names):
        markersize = 5
        n_train = self.n_trained

        for i in range(0,len(experiment_names)):
            fig, ax = plt.subplots(1, 3 ,figsize = (20,5))
            fig2, ax2 = plt.subplots(1, 3 ,figsize = (20,5))
            for temp in self.temperatures:
                lname = label_names[i] + " "+ str(temp)
                mark = "o"
                temp_id = "/t" + str(temp).replace('.','')
                eval_dict = torch.load(path + experiment_names[i] + '/eval_folder/'+temp_id+'evaluations.pt')
                SSIM  = eval_dict['SSIM_values']
                PSNR  = eval_dict['PSNR_values']
                LPIPS  = eval_dict['LPIPS_values']

                print("Temperature is set to " + str(eval_dict['temperature']) + " for experiment " +lname)

                y = SSIM.mean(0).numpy()
                xaxis = np.arange(0+self.n_conditions, len(y)+self.n_conditions)
                ax[0].plot(xaxis, y, label = lname, marker=mark, markersize=markersize)

                y = PSNR.mean(0).numpy()
                ax[1].plot(xaxis,y,label = lname, marker=mark, markersize=markersize)

                y = LPIPS.mean(0).numpy()
                ax[2].plot(xaxis,y,label = lname, marker=mark, markersize=markersize)


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

            fig.savefig(path + experiment_names[i] + '/eval_folder/temps_eval_plots_mean.png', bbox_inches='tight')

    def plot_eval_values(self, path, label_names, experiment_names):
        markersize = 5
        n_train = self.n_trained
        fig, ax = plt.subplots(1, 3 ,figsize = (20,5))
        fig2, ax2 = plt.subplots(1, 3 ,figsize = (20,5))
        fig3, ax3 = plt.subplots(1, 3 ,figsize = (20,5))
        markers = ["o", "v", "x", "*", "^", "s", "H", "P", "X"]

        for i in range(0,len(experiment_names)):
            lname = label_names[i]
            mark = markers[i]
            alpha = 0.05
            eval_dict = torch.load(path + experiment_names[i] + '/eval_folder/evaluations.pt')
            SSIM  = eval_dict['SSIM_values']
            PSNR  = eval_dict['PSNR_values']
            LPIPS  = eval_dict['LPIPS_values']
            SSIM_std_mean = eval_dict['SSIM_std_mean']
            PSNR_std_mean = eval_dict['PSNR_std_mean']
            LPIPS_std_mean = eval_dict['LPIPS_std_mean']
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

            y = SSIM.mean(0).numpy()
            ax3[0].errorbar(xaxis, y, yerr=1.96*SSIM_std_mean,label = lname)

            y = PSNR.mean(0).numpy()
            ax3[1].errorbar(xaxis,y, yerr=1.96*PSNR_std_mean, label = lname)

            y = LPIPS.mean(0).numpy()
            ax3[2].errorbar(xaxis,y, yerr=1.96*LPIPS_std_mean, label = lname)

        ax[0].set_ylabel(r'score')
        ax[0].set_xlabel(r'$t$')
        ax[0].set_title(r'Avg. SSIM with 95% confidence interval')
        ax[0].axvline(x=n_train, color='k', linestyle='--')
        ax[0].legend()
        ax[0].grid()

        ax[1].set_ylabel(r'score')
        ax[1].set_xlabel(r'$t$')
        ax[1].set_title(r'Avg. PSNR with 95% confidence interval')
        ax[1].axvline(x=n_train, color='k', linestyle='--')
        ax[1].legend()
        ax[1].grid()

        ax[2].set_ylabel(r'score')
        ax[2].set_xlabel(r'$t$')
        ax[2].set_title(r'Avg. LPIPS with 95% confidence interval')
        ax[2].axvline(x=n_train, color='k', linestyle='--')
        ax[2].legend()
        ax[2].grid()

        ax2[0].set_ylabel(r'score')
        ax2[0].set_xlabel(r'$t$')
        ax2[0].set_title(r'Avg. SSIM with 95% quantiles')
        ax2[0].axvline(x=n_train, color='k', linestyle='--')
        ax2[0].legend()
        ax2[0].grid()

        ax2[1].set_ylabel(r'score')
        ax2[1].set_xlabel(r'$t$')
        ax2[1].set_title(r'Avg. PSNR with 95% quantiles')
        ax2[1].axvline(x=n_train, color='k', linestyle='--')
        ax2[1].legend()
        ax2[1].grid()

        ax2[2].set_ylabel(r'score')
        ax2[2].set_xlabel(r'$t$')
        ax2[2].set_title(r'Avg. LPIPS with 95% quantiles')
        ax2[2].axvline(x=n_train, color='k', linestyle='--')
        ax2[2].legend()
        ax2[2].grid()

        ax3[0].set_ylabel(r'score')
        ax3[0].set_xlabel(r'$t$')
        ax3[0].set_title(r'Avg. SSIM with uncertainty')
        ax3[0].axvline(x=n_train, color='k', linestyle='--')
        ax3[0].legend()
        ax3[0].grid()

        ax3[1].set_ylabel(r'score')
        ax3[1].set_xlabel(r'$t$')
        ax3[1].set_title(r'Avg. PSNR with uncertainty')
        ax3[1].axvline(x=n_train, color='k', linestyle='--')
        ax3[1].legend()
        ax3[1].grid()

        ax3[2].set_ylabel(r'score')
        ax3[2].set_xlabel(r'$t$')
        ax3[2].set_title(r'Avg. LPIPS with uncertainty')
        ax3[2].axvline(x=n_train, color='k', linestyle='--')
        ax3[2].legend()
        ax3[2].grid()

        fig.savefig(path + experiment_names[i] + '/eval_folder/eval_plots_mean.png', bbox_inches='tight')
        fig2.savefig(path + experiment_names[i] +  '/eval_folder/eval_plots_median.png', bbox_inches='tight')
        fig3.savefig(path + experiment_names[i] +  '/eval_folder/eval_plots_errorbars.png', bbox_inches='tight')

    def get_fvd_values(self, model_name, n_predicts):
      start_predictions = self.start_predictions

      FVD_values = []

      with torch.no_grad():
          self.model.eval()
          for batch_i, true_image in enumerate(tqdm(self.test_loader, desc="Running", position=0, leave=True)):

              if self.choose_data=='bair':
                  image = true_image[0].to(device)
              else:
                  image = true_image.to(device)
              image = self.solver.preprocess(image)
              

              x_true, predictions = self.model.predict(image, n_predicts, start_predictions)

              image  = self.solver.preprocess(image, reverse=True)
              predictions  = self.solver.preprocess(predictions, reverse=True)
              
              # should be [t, b, c, h, w]
              ground_truth = image[:, start_predictions:,:,:,:].permute(1,0,2,3,4).cpu()
              predictions = predictions.cpu()
              
              FVD = fvd(ground_truth, predictions)
              
              #Store values
              FVD_values.append(FVD/n_predicts)

      return FVD_values
