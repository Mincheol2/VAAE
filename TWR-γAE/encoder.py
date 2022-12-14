import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import random
import math
import numpy as np
from loss import *
import scipy.special as scp



class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, z_dim, n_layers, dropout, device, bidirectional=False, rnn_type='lstm', partial='last75', z_mode=None, partial_lag=None, df=0):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.device = device
        self.rnn_type = rnn_type
        self.partial = partial
        self.partial_lag = partial_lag
        self.z_mode = z_mode
        self.layer_dim = (n_layers*2 if bidirectional else n_layers)*hid_dim
        self.df = df

        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers,
                           dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'rnn':
            self.rnn = nn.RNN(emb_dim, hid_dim, n_layers,
                           dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(emb_dim, hid_dim, n_layers,
                           dropout=dropout, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        
        self.linear_mu = nn.Linear(self.layer_dim, z_dim)
        self.linear_var = nn.Linear(self.layer_dim, z_dim)
        

    def reparameterize(self, mu, logvar, mi=False):
        if not self.training and not mi:
            return mu
        else:
            std = torch.exp(0.5*logvar)
            if self.df == 0:
                eps = torch.randn_like(std) # Normal dist
            else:
                Tdist = torch.distributions.studentT.StudentT(self.df)
                eps = Tdist.sample(sample_shape = torch.Size(mu.shape)).to(self.device) # Student T dist

            if not self.training and mi:
                return mu, mu + eps*std
            else:
                return mu + eps*std

    def get_sen_len(self, sens):
        length = torch.sum(sens > 0, dim=0)
        return length.to(dtype=torch.float)

    def forward(self, src):
        # src = [src sent len, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src sent len, batch size, emb dim]
        mu_ = []
        logvar_ = []
        hx = None
        mi_per_batch = None

        if self.rnn_type == 'lstm':
            output, _ = self.rnn(embedded, hx)
            mu_ = self.linear_mu(output)
            logvar_ = self.linear_var(output)
            # mu_, logvar_: [seq_len, batch, z_dim]
        elif self.rnn_type == 'rnn' or 'gru':
            output, _ = self.rnn(embedded, hx)
            mu_ = self.linear_mu(output)
            logvar_ = self.linear_var(output)
            # mu_, logvar_: [seq_len, batch, z_dim]

        if self.z_mode == 'normal':
            mu = mu_[-1]
            logvar = logvar_[-1]
            z = self.reparameterize(mu, logvar)
        elif self.z_mode == 'mean':
            mu = mu_[-1]
            logvar = logvar_[-1]
            z = self.reparameterize(mu_, logvar_) # [seq_len, batch, z_dim]
            z = torch.mean(z,0) # [batch, z_dim]
        elif self.z_mode == 'sum':
            mu = mu_[-1]
            logvar = logvar_[-1]
            z = self.reparameterize(mu_, logvar_) # [seq_len, batch, z_dim]
            z = torch.sum(z,0) # [batch, z_dim]

        if not self.training:
           mi_per_batch = self.cal_mi(mu.squeeze(0), logvar.squeeze(0))

        if self.partial_lag and self.partial in ['last1','last25','last50', 'last75']:
            if self.partial == 'last1':
                mu_ = mu_[-1].unsqueeze(0)
                logvar_ = logvar_[-1].unsqueeze(0)
            elif self.partial == 'last25':
                start_index = round(mu_.shape[0] * 0.25)
                mu_ = mu_[-start_index:]
                logvar_ = logvar_[-start_index:]
            elif self.partial == 'last50':
                start_index = round(mu_.shape[0] * 0.5)
                mu_ = mu_[-start_index:]
                logvar_ = logvar_[-start_index:]
            elif self.partial == 'last75':
                start_index = round(mu_.shape[0] * 0.75)
                mu_ = mu_[-start_index:]
                logvar_ = logvar_[-start_index:]

        return  z, mu_, logvar_, self.get_sen_len(src), mi_per_batch

    def loss(self, mu, logvar, prior_mu, prior_logvar, alpha, beta, df):
        # Alpha div
        if df == 0:
            Alpha_div = Alpha_Family(mu, logvar, prior_mu, prior_logvar)
            div_loss = Alpha_div.alpha_divergence(alpha)
            
        # Gamma div
        else:
            Gamma_div = Gamma_Family(mu, logvar, prior_mu, prior_logvar)
            div_loss = Gamma_div.gamma_divergence(df)
        
        return div_loss * beta

    def log_sum_exp(self, value, dim=None, keepdim=False):
        """Numerically stable implementation of the operation
        value.exp().sum(dim, keepdim).log()
        """
        if dim is not None:
            m, _ = torch.max(value, dim=dim, keepdim=True)
            value0 = value - m
            if keepdim is False:
                m = m.squeeze(dim)
            return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
        else:
            m = torch.max(value)
            sum_exp = torch.sum(torch.exp(value - m))
            return m + torch.log(sum_exp)

    def cal_mi(self, last_mu, last_logvar):
        if self.df == 0:
            return self.cal_normal_mi(last_mu, last_logvar)
        else:
            return self.cal_tdist_mi(last_mu, last_logvar)
        

    def cal_normal_mi(self, last_mu, last_logvar):
        """Approximate the mutual information between x and z
        I(x, z) = E_xE_{q(z|x)}log(q(z|x)) - E_xE_{q(z|x)}log(q(z))
        Returns: Float
        """

        # [x_batch, nz]
        # mu, logvar = self.forward(x)

        x_batch, nz = last_mu.size()

        # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+last_logvar).sum(-1)
        neg_entropy = (-0.5 * nz * math.log(2 * math.pi)- 0.5 * (1 + last_logvar).sum(-1)).mean()

        # [z_batch, 1, nz]
        _, z_samples = self.reparameterize(last_mu, last_logvar, True)
        z_samples = z_samples.unsqueeze(1)
        # [1, x_batch, nz]
        last_mu, last_logvar = last_mu.unsqueeze(0), last_logvar.unsqueeze(0)
        var = last_logvar.exp()

        # (z_batch, x_batch, nz)
        dev = z_samples - last_mu

        # (z_batch, x_batch)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
            0.5 * (nz * math.log(2 * math.pi) + last_logvar.sum(-1))

        # log q(z): aggregate posterior
        # [z_batch]
        log_qz = self.log_sum_exp(log_density, dim=1) - math.log(x_batch)

        return (neg_entropy - log_qz.mean(-1)).item()
    
    def cal_tdist_mi(self, last_mu, last_logvar):
        """Approximate the mutual information between x and z
        when x and z follow student t-distribtuion.
        I(x, z) = E_xE_{q(z|x)}log(q(z|x)) - E_xE_{q(z|x)}log(q(z))
        Returns: Float
        """
        x_batch, nz = last_mu.size() # [Batch, zdim]

        df_const = 0.5*(self.df+nz)
        log_determinant = 0.5*last_logvar.sum(-1)
        log_gamma_diff = scp.loggamma(df_const) - scp.loggamma(self.df/2) - 0.5*nz*math.log(self.df*math.pi)
        digamma_diff = scp.digamma(df_const) - scp.digamma(self.df/2)
        
        # E_{q(z|x)}log(q(z|x))
        neg_entropy = (log_gamma_diff - df_const * digamma_diff - log_determinant).mean()
        
        # [z_batch, 1, nz]
        _, z_samples = self.reparameterize(last_mu, last_logvar, True)
        z_samples = z_samples.unsqueeze(1)
        
        # [1, x_batch, nz]
        last_mu, last_logvar = last_mu.unsqueeze(0), last_logvar.unsqueeze(0)
        var = last_logvar.exp()

        # (z_batch, x_batch, nz)
        dev = z_samples - last_mu
        
        # (z_batch, x_batch)
        log_density = - df_const * torch.log(1+(1/self.df)*((dev ** 2) / var).sum(dim=-1)) + log_gamma_diff - log_determinant

        # log q(z): aggregate posterior
        # [z_batch]
        log_qz = self.log_sum_exp(log_density, dim=1) - math.log(x_batch)

        return (neg_entropy - log_qz.mean(-1)).item()
