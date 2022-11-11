import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

'''
    Default option assumes that
    prior p ~ N(0,I) and posterior q ~ N(mu, var). (Note that var is diagonal.)
    For numerical stablity, we use an argument 'logvar' instead of var.
    
    You can change prior's mean and variance by modifying the argument 'prior_mu' and 'prior_logvar'.
'''


class Alpha_Family():
    def __init__(self, post_mu, post_logvar, prior_mu=None, prior_logvar=None):
        self.post_mu = post_mu
        self.post_logvar = post_logvar
        self.prior_mu = torch.zeros_like(post_mu) if prior_mu is None else prior_mu
        self.prior_logvar = torch.zeros_like(post_logvar) if prior_logvar is None else prior_logvar
        self.post_var = self.post_logvar.exp()
        self.prior_var = self.prior_logvar.exp()


    def KL_loss(self, is_reversed=False):
        kl_div = 0
        logvar_diff = self.post_logvar - self.prior_logvar
        mu_square = (self.post_mu - self.prior_mu).pow(2)
        if is_reversed:
            sq_term = (self.prior_var + mu_square) / self.post_var
            kl_div = -0.5 * torch.mean(- logvar_diff + 1.0 - sq_term)
        else:
            sq_term = (self.post_var + mu_square) / self.prior_var
            kl_div = -0.5 * torch.mean(logvar_diff + 1.0 - sq_term)
        return kl_div
    
    
    def alpha_divergence(self, alpha):
        '''
        Generalized alpha divergence
        
        cf) Special cases
        * KL Divergence (alpha = 1, 0)
        * Hellinger distance (alpha = 0.5)
        * Pearson divergence (alpha = 2)
        * Neyman divergence (alpha = -1)
        '''
        if alpha == 1:
            return self.KL_loss()
        elif alpha == 0:
            return self.KL_loss(is_reversed=True)
        
        else:
            var_denom = (1-alpha) * self.post_var + alpha * self.prior_var
            # Check the well-definedness
            if torch.min(var_denom) <= 0:
                raise Exception(f'min(var_denom) = {torch.min(var_denom)} is not positive. Divergence may not be well-defined.')
            
            const_alpha = 1 / (alpha * (1-alpha))
            prod_const = 0.5 * ((1-alpha) * self.post_logvar + alpha * self.prior_logvar - var_denom.log())
            exp_term = -0.5 * alpha * (1-alpha) * (self.prior_mu - self.post_mu).pow(2) / var_denom
             
            log_prodterm = torch.sum(prod_const + exp_term)
            
            alpha_div = const_alpha * (1 - log_prodterm.exp())
            
            return alpha_div
    
    
    def renyi_divergence(self, alpha):
        if alpha == 1:
            raise Exception('The divergence is not well-defined when alpha = 1')
        
        exp_renyi = 1 + alpha * (alpha - 1 ) * self.alpha_divergence(alpha)

        return exp_renyi.log() / (alpha - 1)



