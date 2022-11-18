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
    def __init__(self, input_dim, hid_dim1, hid_dim2, z_dim,df=0):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.hid_dim1 = hid_dim1
        self.hid_dim2 = hid_dim2
        self.z_dim = z_dim
        self.df = df
        
        self.fc1 = nn.Linear(self.input_dim, self.hid_dim1)
        self.fc2 = nn.Linear(self.hid_dim1, self.hid_dim2)
        self.latent_mu = nn.Linear(self.hid_dim2, self.z_dim)
        self.latent_var = nn.Linear(self.hid_dim2, self.z_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        if self.df == 0:
            eps = torch.randn_like(std) # Normal dist
        else:
            Tdist = torch.distributions.studentT.StudentT(self.df)
            eps = Tdist.sample() # Student T dist
            
        
        
        return mu + std * eps

    def forward(self, x):
        flat_x = x.view(-1,self.input_dim)
        hidden_state_1 = F.relu(self.fc1(flat_x))
        hidden_state_2 = F.relu(self.fc2(hidden_state_1))

        mu = self.latent_mu(hidden_state_2)
        logvar = self.latent_var(hidden_state_2)
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
