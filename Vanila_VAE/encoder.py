import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import random
import math
import numpy as np
from loss import *



class Encoder(nn.Module):
    def __init__(self, input_dim, z_dim,device, df=0):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.df = df
        self.device = device
        
        self.encConv1 = nn.Conv2d(1, 16, 5)
        self.norm1 = nn.BatchNorm2d(16)
        self.encConv2 = nn.Conv2d(16, 32, 5)
        self.norm2 = nn.BatchNorm2d(32)

        self.latent_mu = nn.Linear(32*20*20, self.z_dim)
        self.latent_var = nn.Linear(32*20*20, self.z_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        if self.df == 0:
            eps = torch.randn_like(std) # Normal dist
        else:
            Tdist = torch.distributions.studentT.StudentT(self.df)
            eps = Tdist.sample(sample_shape = torch.Size(prior_mu.shape)).to(self.device) # Student T dist
            
        
        
        return mu + std * eps

    def forward(self, x):
        x = F.leaky_relu(self.norm1(self.encConv1(x)))
        x = F.leaky_relu(self.norm2(self.encConv2(x)))
        x = x.view(-1, 32*20*20)
        mu = self.latent_mu(x)
        logvar = self.latent_var(x)
        z = self.reparameterize(mu, logvar)

        return  z, mu, logvar

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
