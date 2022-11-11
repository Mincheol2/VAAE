import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import random
import math
import numpy as np



class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim1, hid_dim2, z_dim, device):
        super(Decoder, self).__init__()
        self.hid_dim1 = hid_dim1
        self.hid_dim2 = hid_dim2
        self.output_dim = output_dim
        self.z_dim = z_dim
        self.device = device

        self.fc1 = nn.Linear(self.z_dim, self.hid_dim2)
        self.fc2 = nn.Linear(self.hid_dim2, self.hid_dim1)
        self.img_layer = nn.Linear(self.hid_dim1, output_dim)


    def forward(self, enc_z):
        hidden_state_1 = F.relu(self.fc1(enc_z))
        hidden_state_2 = F.relu(self.fc2(hidden_state_1))
        prediction = torch.sigmoid(self.img_layer(hidden_state_2))

        return prediction


    def loss(self, recon_x, x, input_dim):
        recon_loss = F.binary_cross_entropy(recon_x, x.view(-1, input_dim), reduction = 'sum')
        return recon_loss

    