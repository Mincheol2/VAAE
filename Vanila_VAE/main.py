import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.nn import functional as F
import random
import os
from util import *
import numpy as np
from encoder import Encoder
from decoder import Decoder
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='vanila VAE')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='alpha div parameter (default: 1)')
parser.add_argument('--beta', type=float, default=1.0,
                    help='div weight parameter (default: 1)')
parser.add_argument('--seed', type=int, default=999,
                    help='set seed number (default: 999)')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training (default: 64)')
parser.add_argument('--hidden_size1', type=int, default=512,
                    help='the first hidden size for training (default: 512)')
parser.add_argument('--hidden_size2', type=int, default=256,
                    help='the second hidden size for training (default: 256)')
parser.add_argument('--zdim',  type=int, default=32,
                    help='the z size for training (default: 512)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--no_cuda', action='store_true',
                    help='enables CUDA training')
parser.add_argument('-s', '--save', action='store_true', default=True,
                    help='save model every epoch')
parser.add_argument('-l', '--load', action='store_true',
                    help='load model at the begining')
parser.add_argument('-dt','--dataset', type=str, default="mnist",
                    help='Dataset name')
parser.add_argument('--model_dir', type=str, default='',
                    help='model storing path')
parser.add_argument('--gpu_id', type=str, default='0',
                    help='gpu id')

args = parser.parse_args()

SEED = args.seed
## Reproducibility
random.seed(SEED)
torch.manual_seed(SEED)
## Deterministic operations are often slower than nondeterministic operations.
torch.backends.cudnn.deterministic = True


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

if args.load:
    model_dir = args.model_dir
    recon_dir = './'+args.dataset+'_recon_save_alpha' + str(args.alpha) + '_beta' + str(args.beta) + '/'
else:
    model_dir = './'+args.dataset+'_model_save_alpha' + str(args.alpha) + '_beta' + str(args.beta) + '/'
    recon_dir = './'+args.dataset+'_recon_save_alpha' + str(args.alpha) + '_beta' + str(args.beta) + '/'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(recon_dir):
    os.makedirs(recon_dir)

## For tensorboard ##
writer = SummaryWriter(model_dir + 'Tensorboard_results')

def train(train_loader, encoder, decoder, opt, epoch, alpha, beta):
    encoder.train()
    decoder.train()
    total_loss = []
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(DEVICE)
        opt.zero_grad()
        z, mu, logvar = encoder(data)
        div_loss = encoder.loss(mu, logvar, alpha, beta)

        recon_img = decoder(z)
        recon_loss = decoder.loss(recon_img, data, input_dim)

        current_loss = div_loss + recon_loss
        writer.add_scalar("Train/Reconstruction Error", recon_loss.item(), batch_idx + epoch * (len(train_loader.dataset)/args.batch_size) )
        writer.add_scalar("Train/KL-Divergence", div_loss.item(), batch_idx + epoch * (len(train_loader.dataset)/args.batch_size) )
        writer.add_scalar("Train/Total Loss" , current_loss.item(), batch_idx + epoch * (len(train_loader.dataset)/args.batch_size) )

        current_loss.backward()

        total_loss.append(current_loss.item())
        opt.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       current_loss.item() / len(data)))
    return total_loss


def reconstruction(test_loader, encoder, decoder, ep, alpha, beta):
    encoder.eval()
    decoder.eval()
    vectors = []

    for batch_idx, (data, labels) in enumerate(test_loader):
        with torch.no_grad():

            data = data.to(DEVICE)
            z, mu, logvar = encoder(data)
            div_loss = encoder.loss(mu, logvar, alpha, beta)

            recon_img = decoder(z)
            recon_loss = decoder.loss(recon_img, data, input_dim)

            current_loss = div_loss + recon_loss

            writer.add_scalar("Test/Reconstruction Error", recon_loss.item(), batch_idx + epoch * (len(test_loader.dataset)/args.batch_size) )
            writer.add_scalar("Test/KL-Divergence", div_loss.item(), batch_idx + epoch * (len(test_loader.dataset)/args.batch_size) )
            writer.add_scalar("Test/Total Loss" , current_loss.item(), batch_idx + epoch * (len(test_loader.dataset)/args.batch_size) )
            
            #temp = list(zip(labels.tolist(), mu.tolist()))

            recon_img = recon_img.view(-1, 1, image_size, image_size)

            #for x in temp:
            #    vectors.append(x)

            if batch_idx % 100 == 0:
                print('Test Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.4f}'.format(
                    epoch, batch_idx * len(data), len(test_loader.dataset),
                           100. * batch_idx / len(test_loader),
                           current_loss.item() / len(data)))
        if batch_idx == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n], recon_img.view(args.batch_size, 1, 28, 28)[:n]]) # (16, 1, 28, 28)
            grid = torchvision.utils.make_grid(comparison.cpu()) # (3, 62, 242)
            writer.add_image("Test image - Above: Real data, below: reconstruction data", grid, epoch)
        #show_images(recon_img.cpu())
        #img_name = recon_dir + "recon_imgs/" + str(batch_idx).zfill(3)
        #torchvision.utils.save_image(recon_img, img_name)
    return




# Loading trainset, testset and trainloader, testloader
transformer = transforms.Compose([transforms.ToTensor()])

if args.dataset == "mnist":
    trainset = torchvision.datasets.MNIST(root='./MNIST', train=True,
                                          download=True, transform=transformer)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root='./MNIST', train=False,
                                         download=True, transform=transformer)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=2)

else:
    pass
    #TBD




image_size = 28
input_dim = 784 # 28**2 : MNIST (I'll generalize this param for any dataset)

encoder = Encoder(input_dim, args.hidden_size1, args.hidden_size2, args.zdim).to(DEVICE)
decoder = Decoder(input_dim, args.hidden_size1, args.hidden_size2, args.zdim, device=DEVICE).to(DEVICE)
lr = args.lr
opt = optim.Adam(list(encoder.parameters()) +
                 list(decoder.parameters()), lr=lr, eps=1e-6, weight_decay=1e-5)


alpha = args.alpha
beta = args.alpha

print(f'Current alpha : {alpha}')
print(f'Current beta : {beta}')



# ============== run ==============
if args.load:
    state = torch.load(model_dir+'Vanilavae.tch')
    encoder.load_state_dict(state["encoder"])
    decoder.load_state_dict(state["decoder"])
    state2 = torch.load(model_dir+'Vanilavae.tchopt')
    ep = state2["ep"]+1
    opt.load_state_dict(state2["opt"])
      
    reconstruction(testloader, encoder, decoder, ep,alpha, beta)
    writer.close()
else:

    for epoch in tqdm(range(0, args.epochs)):
        train(trainloader, encoder, decoder, opt, epoch, alpha, beta)
        reconstruction(testloader, encoder, decoder, epoch, alpha, beta)
    writer.close()

