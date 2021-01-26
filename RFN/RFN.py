from Flow import ListGlow
import torch
import torch.nn as nn
from Utils import VGG_upscaler, VGG_downscaler, SimpleParamNet, ConvLSTMOld, free_bits_kl, batch_reduce
import torch.distributions as td

class RFN(nn.Module):
    def __init__(self, args):
      super(RFN, self).__init__()
      self.params = args
      batch_size = args.batch_size
      self.u_dim = args.x_dim
      self.x_dim = args.condition_dim
      self.h_dim = args.h_dim
      self.z_dim = args.z_dim
      self.beta = 1
      scaler = args.structure_scaler
      self.L = args.L
      self.K = args.K
      norm_type = args.norm_type
      norm_type_features = args.norm_type_features
      self.temperature = args.temperature
      self.prior_structure = args.prior_structure
      self.encoder_structure = args.encoder_structure
      condition_size_list = []
      self.free_bits = args.free_bits
      self.skip_connection_flow = args.skip_connection_flow
      self.downscaler_tanh = args.downscaler_tanh

      self.skip_connection_features = args.skip_connection_features
      self.upscaler_tanh = args.upscaler_tanh

      down_structure = args.extractor_structure
      up_structure = args.upscaler_structure

      if self.skip_connection_flow == "without_skip" and not self.skip_connection_features:
          skip = False
      else:
          skip = True
      self.extractor = VGG_downscaler(down_structure,L=self.L, in_channels = self.x_dim[1],
                                      norm_type=norm_type_features, non_lin = "relu", scale = scaler,
                                      skip_con=skip, tanh = self.downscaler_tanh)

      # adjust channel dims to match up_structure. Reversed.
      channel_dims = [i[-1] for i in up_structure][::-1]

      dims_skip = self.extractor.get_layer_size(down_structure, self.x_dim)
      hu, wu = (self.u_dim[2], self.u_dim[3])

      for i in range(0, self.L):
        hu, wu = (hu//2, hu//2)
        if self.skip_connection_flow == "with_skip":
            condition_size_list.append([batch_size, channel_dims[i]+dims_skip[i][1], hu, wu])
        elif self.skip_connection_flow =="without_skip":
            condition_size_list.append([batch_size, channel_dims[i], hu, wu])
        elif self.skip_connection_flow == "only_skip":
            condition_size_list.append([batch_size, dims_skip[i][1], hu, wu])
        else:
            print("choose skip setting")

      c_features = dims_skip[-1][1]

      self.z_0 = nn.Parameter(torch.zeros(batch_size, self.z_dim, hu, wu))
      self.z_0x = nn.Parameter(torch.zeros(batch_size, self.z_dim, hu, wu))

      self.h_0 = nn.Parameter(torch.zeros(batch_size, self.h_dim, hu, wu))
      self.c_0 = nn.Parameter(torch.zeros(batch_size, self.h_dim, hu, wu))

      # Feature extractor and upscaler for flow
      self.upscaler = VGG_upscaler(up_structure, L=self.L, in_channels = self.h_dim + self.z_dim,
                                   norm_type = norm_type_features, non_lin = "leakyrelu",
                                   scale = scaler, skips = self.skip_connection_features,
                                   size_skips = dims_skip, tanh = self.upscaler_tanh)

      # ConvLSTM
      self.lstm = ConvLSTMOld(in_channels = c_features, hidden_channels=self.h_dim,
                           kernel_size=[3, 3], bias=True, peephole=True)

      # Prior
      prior_struct = self.prior_structure
      self.prior = SimpleParamNet(prior_struct, in_channels = self.h_dim + self.z_dim,
                                  out_channels = self.z_dim, norm_type = norm_type, non_lin = "leakyrelu")

      # Flow
      base_dim = (batch_size, self.h_dim + self.z_dim, hu, wu)
      self.flow = ListGlow(self.x_dim, condition_size_list, base_dim,
                           args=self.params)

      # Variational encoder
      enc_struct = self.encoder_structure
      self.encoder = SimpleParamNet(enc_struct, in_channels = c_features + self.h_dim + self.z_dim,
                                    out_channels = self.z_dim, norm_type = norm_type, non_lin = "leakyrelu")


    def get_inits(self):
      loss = 0
      kl_loss = 0
      nll_loss = 0
      return self.h_0, self.c_0, self.z_0, self.z_0x, loss, kl_loss, nll_loss

    def loss(self, x, logdet):
      assert len(x.shape) == 5, "x must be [bs, t, c, h, w]"
      hprev, cprev, zprev, zxprev, loss, kl_loss, nll_loss = self.get_inits()
      t = x.shape[1]
      condition_list = self.extractor(x[:, 0, :, :, :])

      for i in range(1, t):

        x_feature_list = self.extractor(x[:, i, :, :, :])

        if self.skip_connection_flow == "without_skip" and not self.skip_connection_features:
            condition = condition_list
            x_feature = x_feature_list
        else:
            condition = condition_list[-1]
            x_feature = x_feature_list[-1]


        _, ht, ct = self.lstm(condition.unsqueeze(1), hprev, cprev)

        prior_mean, prior_std = self.prior(torch.cat((ht, zprev), dim=1))
        dist_prior = td.Normal(prior_mean, prior_std)
        zt = dist_prior.rsample()

        enc_mean, enc_std = self.encoder(torch.cat((ht, zxprev, x_feature), dim = 1))
        dist_enc = td.Normal(enc_mean, enc_std)
        zxt = dist_enc.rsample()
        if self.skip_connection_features:
            flow_conditions = self.upscaler(torch.cat((ht, zxt), dim = 1), skip_list = condition_list)
        else:
            flow_conditions = self.upscaler(torch.cat((ht, zxt), dim = 1))

        base_conditions = torch.cat((ht, zxt), dim = 1)

        if self.skip_connection_flow == "with_skip":
            flow_conditions = self.combineconditions(flow_conditions, condition_list)
        elif self.skip_connection_flow == "only_skip":
            flow_conditions = condition_list

        b, nll = self.flow.log_prob(x[:, i, :, :, :], flow_conditions, base_conditions, logdet)


        kl_loss = kl_loss + td.kl_divergence(dist_enc, dist_prior)

        if self.free_bits>0:
            kl_free_bit = free_bits_kl(kl_loss, free_bits = self.free_bits)
        else:
            kl_free_bit = kl_loss

        nll_loss = nll_loss + nll

        zprev, zxprev, condition_list = zt, zxt, x_feature_list
        hprev, cprev = ht ,ct

      return batch_reduce(kl_free_bit).mean(), batch_reduce(kl_loss).mean(), nll_loss.mean()

    def combineconditions(self, flow_conditions, skip_conditions):
      flow_conditions_combined = []

      for k in range(0, len(flow_conditions)):
        flow_conditions_combined.append(torch.cat((flow_conditions[k], skip_conditions[k]),dim=1))

      return flow_conditions_combined

    def predict(self, x, n_predictions, n_conditions):
        assert len(x.shape) == 5, "x must be [bs, t, c, h, w]"
        hprev, cprev, zprev, _, _, _, _ = self.get_inits()

        predictions = torch.zeros((n_predictions, *x[:,0,:,:,:].shape))
        true_x = torch.zeros((n_conditions, *x[:,0,:,:,:].shape))

        # Warm-up
        true_x[0,:,:,:,:] = x[:, 0, :, :, :].detach()
        for i in range(1, n_conditions):
            condition_list = self.extractor(x[:, i-1, :, :, :])

            #Fix this so it is smarter
            if self.skip_connection_flow == "without_skip" and not self.skip_connection_features:
                condition = condition_list
            else:
                condition = condition_list[-1]

            _, ht, ct = self.lstm(condition.unsqueeze(1), hprev, cprev)

            # We do not need to update the encoder due to LSTM not being conditioned on it
            prior_mean, prior_std = self.prior(torch.cat((ht, zprev), dim=1))
            dist_prior = td.Normal(prior_mean, prior_std)
            zt = dist_prior.sample()

            true_x[i,:,:,:,:] = x[:, i, :, :, :].detach()
            zprev = zt
            hprev = ht
            cprev = ct

        prediction = x[:,n_conditions-1,:,:,:]
        for i in range(0, n_predictions):
            if self.skip_connection_flow == "without_skip" and not self.skip_connection_features:
                condition = self.extractor(prediction)
            else:
                condition_list = self.extractor(prediction)
                condition = condition_list[-1]
            _, ht, ct = self.lstm(condition.unsqueeze(1), hprev, cprev)

            prior_mean, prior_std = self.prior(torch.cat((ht, zprev), dim=1))
            dist_prior = td.Normal(prior_mean, prior_std)
            zt = dist_prior.sample()

            if self.skip_connection_features:
                flow_conditions = self.upscaler(torch.cat((ht, zt), dim = 1), skip_list = condition_list)
            else:
                flow_conditions = self.upscaler(torch.cat((ht, zt), dim = 1))

            if self.skip_connection_flow == "with_skip":
                flow_conditions = self.combineconditions(flow_conditions, condition_list)
            elif self.skip_connection_flow == "only_skip":
                flow_conditions = condition_list

            base_conditions = torch.cat((ht, zt), dim = 1)
            prediction = self.flow.sample(None, flow_conditions, base_conditions, self.temperature)
            predictions[i,:,:,:,:] = prediction.detach()
            hprev, cprev = ht, ct
            zprev = zt

        return true_x, predictions
    def get_zt_ht_from_seq(self, x, n_conditions):
        assert len(x.shape) == 5, "x must be [bs, t, c, h, w]"
        # Get zt ht from one batch
        hprev, cprev, zprev, _, _, _, _ = self.get_inits()

        zts = torch.zeros((n_conditions, *zprev.shape))
        hts = torch.zeros((n_conditions, *hprev.shape))

        zts[0, :, :, :, :] = zprev
        hts[0, :, :, :, :] = hprev
        for i in range(1, n_conditions):
            condition_list = self.extractor(x[:, i-1, :, :, :])

            #Fix this so it is smarter
            if self.skip_connection_flow == "without_skip" and not self.skip_connection_features:
                condition = condition_list
            else:
                condition = condition_list[-1]

            _, ht, ct = self.lstm(condition.unsqueeze(1), hprev, cprev)

            # We do not need to update the encoder due to LSTM not being conditioned on it
            prior_mean, prior_std = self.prior(torch.cat((ht, zprev), dim=1))
            dist_prior = td.Normal(prior_mean, prior_std)
            zt = dist_prior.sample()
            zts[i, :, :, :, :] = zt
            hts[i, :, :, :, :] = ht
            zprev = zt
            hprev = ht
            cprev = ct
        return zts, hts

    def predicts_from_zt_ht(self, x, zts, hts):
        n_predictions = zts.shape[0]
        predictions = torch.zeros((n_predictions, *x[:,0,:,:,:].shape))

        condition_list = self.extractor(x[:, 0, :, :, :])

        prediction = x[:, 0, :, :, :]
        for i in range(1, n_predictions):
            if self.skip_connection_flow == "without_skip" and not self.skip_connection_features:
                pass
            else:
                condition_list = self.extractor(prediction)
            ht = hts[i, :, :, :, :]
            zt = zts[i, :, :, :, :]

            if self.skip_connection_features:
                flow_conditions = self.upscaler(torch.cat((ht, zt), dim = 1), skip_list = condition_list)
            else:
                flow_conditions = self.upscaler(torch.cat((ht, zt), dim = 1))

            if self.skip_connection_flow == "with_skip":
                flow_conditions = self.combineconditions(flow_conditions, condition_list)
            elif self.skip_connection_flow == "only_skip":
                flow_conditions = condition_list

            base_conditions = torch.cat((ht, zt), dim = 1)
            prediction = self.flow.sample(None, flow_conditions, base_conditions, self.temperature)
            predictions[i,:,:,:,:] = prediction.detach()


        return predictions

    def reconstruct(self, x):
        assert len(x.shape) == 5, "x must be [bs, t, c, h, w]"
        hprev, cprev, _, zxprev, _, _, _ = self.get_inits()

        t = x.shape[1]
        recons = torch.zeros((t, *x[:,0,:,:,:].shape))
        recons_flow = torch.zeros((t, *x[:,0,:,:,:].shape))
        for i in range(1, t):
            condition_list = self.extractor(x[:, i-1, :, :, :])
            x_feature_list = self.extractor(x[:, i, :, :, :])
            #Fix this so it is smarter
            if self.skip_connection_flow == "without_skip" and not self.skip_connection_features:
                condition = condition_list
                x_feature = x_feature_list
            else:
                condition = condition_list[-1]
                x_feature = x_feature_list[-1]

            _, ht, ct = self.lstm(condition.unsqueeze(1), hprev, cprev)

            enc_mean, enc_std = self.encoder(torch.cat((ht, zxprev, x_feature), dim = 1))
            dist_enc = td.Normal(enc_mean, enc_std)
            zxt = dist_enc.sample()

            if self.skip_connection_features:
                flow_conditions = self.upscaler(torch.cat((ht, zxt), dim = 1), skip_list = condition_list)
            else:
                flow_conditions = self.upscaler(torch.cat((ht, zxt), dim = 1))

            if self.skip_connection_flow == "with_skip":
                flow_conditions = self.combineconditions(flow_conditions, condition_list)
            elif self.skip_connection_flow == "only_skip":
                flow_conditions = condition_list

            base_conditions = torch.cat((ht, zxt), dim = 1)

            # To check bijection of the flow we use z to reconstruct the image
            z, _ = self.flow.log_prob(x[:, i, :, :, :], flow_conditions, base_conditions, 0.0)
            recon_flow_sample = self.flow.sample(z, flow_conditions, base_conditions, self.temperature)

            # To look at a normal reconstruction we just use the posterior distribution
            recon_sample = self.flow.sample(None, flow_conditions, base_conditions, self.temperature)

            recons[i,:,:,:,:] = recon_sample.detach()
            recons_flow[i,:,:,:,:] = recon_flow_sample.detach()

            hprev = ht
            cprev = ct
            zxprev = zxt
        return recons, recons_flow

    def probability_future(self, x, n_conditions):
        """
        This function gets the temporal probablity of the future frames,
        when conditioned on n_condition frames.
        Both over the posterior and the prior.
        """
        assert len(x.shape) == 5, "x must be [bs, t, c, h, w]"
        hprev, cprev, zprev, zxprev, _, _, _ = self.get_inits()

        t = x.shape[1]
        # Do this to make the code more efficient.
        prob_NLL_future = torch.zeros((x[:,0,:,:,:].shape[0], 2, t-n_conditions-1))

        for i in range(1, n_conditions):
            condition_list = self.extractor(x[:, i-1, :, :, :])
            x_feature_list = self.extractor(x[:, i, :, :, :])
            #Fix this so it is smarter
            if self.skip_connection_flow == "without_skip" and not self.skip_connection_features:
                condition = condition_list
                x_feature = x_feature_list
            else:
                condition = condition_list[-1]
                x_feature = x_feature_list[-1]

            _, ht, ct = self.lstm(condition.unsqueeze(1), hprev, cprev)

            prior_mean, prior_std = self.prior(torch.cat((ht, zprev), dim=1))
            dist_prior = td.Normal(prior_mean, prior_std)
            zt = dist_prior.sample()

            enc_mean, enc_std = self.encoder(torch.cat((ht, zxprev, x_feature), dim = 1))
            dist_enc = td.Normal(enc_mean, enc_std)
            zxt = dist_enc.sample()
            zprev = zt
            hprev = ht
            cprev = ct
            zxprev = zxt
        # The probability of the future frames.
        for i in range(n_conditions, t):
            for zk, count in zip(list([zt,zxt]),range(0,2)):
                if self.skip_connection_features:
                    flow_conditions = self.upscaler(torch.cat((ht, zk), dim = 1), skip_list = condition_list)
                else:
                    flow_conditions = self.upscaler(torch.cat((ht, zk), dim = 1))

                if self.skip_connection_flow == "with_skip":
                    flow_conditions = self.combineconditions(flow_conditions, condition_list)
                elif self.skip_connection_flow == "only_skip":
                    flow_conditions = condition_list

                base_conditions = torch.cat((ht, zk), dim = 1)

                z, nll = self.flow.log_prob(x[:, i, :, :, :], flow_conditions, base_conditions, 0.0)
                prob_NLL_future[:, count, i-n_conditions-1] = nll
        return prob_NLL_future

    def reconstruct_elbo_gap(self, x, sample = True):
        assert len(x.shape) == 5, "x must be [bs, t, c, h, w]"
        hprev, cprev, zprev, zxprev, _, _, _ = self.get_inits()

        t = x.shape[1]
        # Do this to make the code more efficient.
        averageKLDseq = torch.zeros((t, x[:,0,:,:,:].shape[0]))
        averageNLLseq = torch.zeros((2, t, x[:,0,:,:,:].shape[0]))
        if sample:
            recons = torch.zeros((2, t, *x[:,0,:,:,:].shape))
            recons_flow = torch.zeros((2, t, *x[:,0,:,:,:].shape))
        else:
            recons = 0
            recons_flow = 0

        for i in range(1, t):
            condition_list = self.extractor(x[:, i-1, :, :, :])
            x_feature_list = self.extractor(x[:, i, :, :, :])
            #Fix this so it is smarter
            if self.skip_connection_flow == "without_skip" and not self.skip_connection_features:
                condition = condition_list
                x_feature = x_feature_list
            else:
                condition = condition_list[-1]
                x_feature = x_feature_list[-1]

            _, ht, ct = self.lstm(condition.unsqueeze(1), hprev, cprev)

            prior_mean, prior_std = self.prior(torch.cat((ht, zprev), dim=1))
            dist_prior = td.Normal(prior_mean, prior_std)
            zt = dist_prior.sample()

            enc_mean, enc_std = self.encoder(torch.cat((ht, zxprev, x_feature), dim = 1))
            dist_enc = td.Normal(enc_mean, enc_std)
            zxt = dist_enc.sample()
            for zk, count in zip(list([zt,zxt]),range(0,2)):
                if self.skip_connection_features:
                    flow_conditions = self.upscaler(torch.cat((ht, zk), dim = 1), skip_list = condition_list)
                else:
                    flow_conditions = self.upscaler(torch.cat((ht, zk), dim = 1))

                if self.skip_connection_flow == "with_skip":
                    flow_conditions = self.combineconditions(flow_conditions, condition_list)
                elif self.skip_connection_flow == "only_skip":
                    flow_conditions = condition_list

                base_conditions = torch.cat((ht, zk), dim = 1)

                # To check bijection of the flow we use z to reconstruct the image
                z, nll = self.flow.log_prob(x[:, i, :, :, :], flow_conditions, base_conditions, 0.0)
                averageNLLseq[count, i, :] = nll
                if sample:
                    recon_flow_sample = self.flow.sample(z, flow_conditions, base_conditions, self.temperature)
                    # To look at a normal reconstruction we just use the posterior distribution
                    recon_sample = self.flow.sample(None, flow_conditions, base_conditions, self.temperature)
                    recons[count, i,:,:,:,:] = recon_sample.detach()
                    recons_flow[count, i, :,:,:,:] = recon_flow_sample.detach()
            KLavg = td.kl_divergence(dist_enc, dist_prior).sum([1,2,3]) # sum over everything expect batches
            averageKLDseq[i,:] = KLavg
             # KL batch
            zprev = zt
            hprev = ht
            cprev = ct
            zxprev = zxt
        return recons, recons_flow, averageKLDseq, averageNLLseq

    def sample(self, x, n_samples):
        assert len(x.shape) == 5, "x must be [bs, t, c, h, w]"
        hprev, cprev, zprev, _, _, _, _ = self.get_inits()

        samples = torch.zeros((n_samples, *x[:,0,:,:,:].shape))

        condition_list = self.extractor(x[:, 0, :, :, :])
        #
        for i in range(0, n_samples):

            if self.skip_connection_flow == "without_skip" and not self.skip_connection_features:
                condition = condition_list
            else:
                condition = condition_list[-1]

            _, ht, ct = self.lstm(condition.unsqueeze(1), hprev, cprev)

            # We do not need to update the encoder due to LSTM not being conditioned on it
            prior_mean, prior_std = self.prior(torch.cat((ht, zprev), dim=1))
            dist_prior = td.Normal(prior_mean, prior_std)
            zt = dist_prior.sample()

            if self.skip_connection_features:
                flow_conditions = self.upscaler(torch.cat((ht, zt), dim = 1), skip_list = condition_list)
            else:
                flow_conditions = self.upscaler(torch.cat((ht, zt), dim = 1))

            if self.skip_connection_flow == "with_skip":
                flow_conditions = self.combineconditions(flow_conditions, condition_list)
            elif self.skip_connection_flow == "only_skip":
                flow_conditions = condition_list

            base_conditions = torch.cat((ht, zt), dim = 1)
            sample = self.flow.sample(None, flow_conditions, base_conditions, self.temperature)


            samples[i,:,:,:,:] = sample.detach()
            zprev = zt
            hprev = ht
            cprev = ct
            condition_list = self.extractor(sample)

        return samples

    def param_analysis(self, x, n_predictions, n_conditions):
        assert len(x.shape) == 5, "x must be [bs, t, c, h, w]"
        hprev, cprev, zprev, zxprev, _, _, _ = self.get_inits()
        hw = x.shape[-1]//(2**self.L)
        std_p = torch.zeros((n_predictions+n_conditions, x.shape[0], self.z_dim, hw, hw))
        mu_p = torch.zeros((n_predictions+n_conditions, x.shape[0], self.z_dim, hw, hw))
        std_q = torch.zeros((n_conditions-1, x.shape[0], self.z_dim, hw, hw))
        mu_q = torch.zeros((n_conditions-1, x.shape[0], self.z_dim, hw, hw))
        std_flow = []
        mu_flow = []

        # Warm-up
        for i in range(1, n_conditions):
            condition_list = self.extractor(x[:, i-1, :, :, :])
            x_feature_list = self.extractor(x[:, i, :, :, :])
            #Fix this so it is smarter
            if self.skip_connection_flow == "without_skip" and not self.skip_connection_features:
                condition = condition_list
                x_feature = x_feature_list
            else:
                condition = condition_list[-1]
                x_feature = x_feature_list[-1]

            _, ht, ct = self.lstm(condition.unsqueeze(1), hprev, cprev)


            prior_mean, prior_std = self.prior(torch.cat((ht, zprev), dim=1))
            dist_prior = td.Normal(prior_mean, prior_std)
            zt = dist_prior.sample()

            enc_mean, enc_std = self.encoder(torch.cat((ht, zxprev, x_feature), dim = 1))
            dist_enc = td.Normal(enc_mean, enc_std)
            zxt = dist_enc.rsample()

            zxprev = zxt
            zprev = zt
            hprev = ht
            cprev = ct
            std_p[i-1,:,:,:,:] = prior_std.detach()
            mu_p[i-1,:,:,:,:] = prior_mean.detach()
            std_q[i-1,:,:,:,:] = enc_std.detach()
            mu_q[i-1,:,:,:,:] = enc_mean.detach()

        prediction = x[:,n_conditions-1,:,:,:]
        for i in range(0, n_predictions):
            if self.skip_connection_flow == "without_skip" and not self.skip_connection_features:
                condition = self.extractor(prediction)
            else:
                condition_list = self.extractor(prediction)
                condition = condition_list[-1]
            _, ht, ct = self.lstm(condition.unsqueeze(1), hprev, cprev)

            prior_mean, prior_std = self.prior(torch.cat((ht, zprev), dim=1))
            dist_prior = td.Normal(prior_mean, prior_std)
            zt = dist_prior.sample()


            if self.skip_connection_features:
                flow_conditions = self.upscaler(torch.cat((ht, zt), dim = 1), skip_list = condition_list)
            else:
                flow_conditions = self.upscaler(torch.cat((ht, zt), dim = 1))

            if self.skip_connection_flow == "with_skip":
                flow_conditions = self.combineconditions(flow_conditions, condition_list)
            elif self.skip_connection_flow == "only_skip":
                flow_conditions = condition_list

            base_conditions = torch.cat((ht, zt), dim = 1)
            prediction, params = self.flow.sample(None, flow_conditions, base_conditions, self.temperature, eval_params = True)
            hprev, cprev = ht, ct
            zprev = zt
            std_p[i+n_conditions-1,:,:,:,:] = prior_std.detach()
            mu_p[i+n_conditions-1,:,:,:,:] = prior_mean.detach()
            std_flow.append(params[1].detach())
            mu_flow.append(params[0].detach())

        return mu_p, std_p, mu_q, std_q, torch.stack(mu_flow), torch.stack(std_flow)
