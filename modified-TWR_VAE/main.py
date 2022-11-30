import argparse
from itertools import islice
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import random
import os
from BatchIter import BatchIter
import numpy as np

from train import *
from Corpus import Corpus
from encoder import Encoder
from decoder import Decoder
from tqdm import tqdm

parser = argparse.ArgumentParser(description='TWR-VAE for PTB/Yelp/Yahoo')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='alpha div parameter (default: 1.0)')
parser.add_argument('--beta', type=float, default=1.0,
                    help='div weight parameter (default: 1.0)')
parser.add_argument('--df', type=float, default=0.0,
                    help='gamma div parameter (default: 0)')
parser.add_argument('--prior_mu', type=float, default=0,
                    help='prior_mu')
parser.add_argument('--prior_logvar', type=float, default=0,
                    help='prior_logvar')
parser.add_argument('--seed', type=int, default=999,
                    help='set seed number (default: 999)')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training (default: 64)')
parser.add_argument('--embedding_size', type=int, default=512,
                    help='embedding size for training (default: 512)')
parser.add_argument('--hidden_size', type=int, default=256,
                    help='hidden size for training (default: 256)')
parser.add_argument('--zdim',  type=int, default=32,
                    help='the z size for training (default: 512)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 200)')
parser.add_argument('--layers', type=int, default=1,
                    help='number of layers of rnns in encoder and decoder')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout values')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--no_cuda', action='store_true',
                    help='enables CUDA training')
parser.add_argument('-s', '--save', action='store_true', default=True,
                    help='save model every epoch')
parser.add_argument('-l', '--load', action='store_true',
                    help='load model at the begining')
parser.add_argument('-dt','--dataset', type=str, default="ptb",
                    help='Dataset name')
parser.add_argument('-mw','--min_word_count',type=int, default=1,
                    help='minimum word count')
parser.add_argument('-rnn','--rnn_type', type=str, default="lstm",
                    help='RNN types (rnn, lstm and gru)')
parser.add_argument('-par','--partial', action='store_true',
                    help='partially optimise KL')
parser.add_argument('-party','--partial_type', type=str, default='last75',
                    help='partial type: last1 last25 last50 last75')
parser.add_argument('--z_type', type=str, default='normal',
                    help='z mode for decoder: normal, mean, sum')
parser.add_argument('--model_dir', type=str, default='',
                    help='model storing path')
parser.add_argument('--gpu_id', type=str, default='0',
                    help='gpu id')
args = parser.parse_args()

print(args)

base_path = "."
print("base_path=", base_path)
SEED = args.seed
lr = args.lr


if args.dataset == "ptb":
    Train = Corpus(base_path+'/dataset/ptb/ptb_train.txt', min_word_count=args.min_word_count)
    Test = Corpus(base_path+'/dataset/ptb/ptb_test.txt', word_dic=Train.word_id, min_word_count=args.min_word_count)
elif args.dataset == "yelp":
    Train = Corpus(base_path+'/dataset/yelp/yelp.train.txt', min_word_count=args.min_word_count)
    Test = Corpus(base_path+'/dataset/yelp/yelp.test.txt', word_dic=Train.word_id, min_word_count=args.min_word_count)
elif args.dataset == "yahoo":
    Train = Corpus(base_path+'/dataset/yahoo/yahoo_train.txt', min_word_count=args.min_word_count)
    Test = Corpus(base_path+'/dataset/yahoo/yahoo_test.txt', word_dic=Train.word_id, min_word_count=args.min_word_count)

    


voca_dim = Train.voca_size
emb_dim = args.embedding_size
hid_dim = args.hidden_size
batch_size = args.batch_size


## Reproducibility
random.seed(SEED)
torch.manual_seed(SEED)
## Deterministic operations are often slower than nondeterministic operations.
torch.backends.cudnn.deterministic = True
##


device = torch.device('cuda:'+args.gpu_id if torch.cuda.is_available()
                      and not args.no_cuda else 'cpu')


dataloader_train = BatchIter(Train, batch_size)
dataloader_test = BatchIter(Test, batch_size)

alpha = args.alpha
beta = args.beta
df = args.df



encoder = Encoder(voca_dim, emb_dim, hid_dim, args.zdim, args.layers, args.dropout,
              rnn_type=args.rnn_type,
              partial=args.partial_type,
              z_mode=args.z_type,
              partial_lag=args.partial,
              df = args.df).to(device)
decoder = Decoder(voca_dim, emb_dim, hid_dim, args.zdim, args.layers, args.dropout,
                  rnn_type=args.rnn_type,
                  z_mode=args.z_type,
                  device=device).to(device)
opt = optim.Adam(list(encoder.parameters()) +
                 list(decoder.parameters()), lr=lr, eps=1e-6, weight_decay=1e-5)





if args.load:
    model_dir = args.model_dir
    recon_dir = base_path+'/'+args.dataset+ f'_recon_save_alpha{alpha}_beta{beta}_df{df}/'

else:
    model_dir = base_path+'/'+args.dataset+ f'_model_save_alpha{alpha}_beta{beta}_df{df}/'
    recon_dir = base_path+'/'+args.dataset+ f'_recon_save_alpha{alpha}_beta{beta}_df{df}/'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(recon_dir):
    os.makedirs(recon_dir)
    
    
print(f'Current alpha : {alpha}')
print(f'Current beta : {beta}')
if df != 0:
    print(f'Current df: {df} -> gamma : {-2 / (df+1)}')


# ============== run ==============
if args.load:
    state = torch.load(model_dir+'TWRvae.tch')
    encoder.load_state_dict(state["encoder"])
    decoder.load_state_dict(state["decoder"])
    state2 = torch.load(model_dir+'TWRvae.tchopt')
    ep = state2["ep"]+1
    opt.load_state_dict(state2["opt"])
      
    test_ppl = reconstruction(dataloader_test, ep)
else:
    ep = 0
    history = []
    Best_ppl = 1e5
    eval_ppl = 1e5

    with open(model_dir+'train_TWRvae_loss.txt', 'w') as f:
        f.write("ep \t recon_loss \t div_loss \t acc \t nll_loss \t ppl \n")
    with open(recon_dir+'test_TWRvae_loss.txt', 'w') as f:
        f.write("ep \t recon_loss \t div_loss \t NLL \t PPL \n")


    for ep in tqdm(range(ep+1, args.epochs+1)):
        recon_loss, var_loss, acc, nll_loss, ppl = train(encoder, decoder, opt, device, dataloader_train, ep, args.prior_mu, args.prior_logvar, alpha, beta, df)
        history.append(f"{ep}\t{recon_loss}\t{var_loss}\t{acc}\t{nll_loss}\t{ppl}")
        with open(model_dir+'train_TWRvae_loss.txt', 'w') as f:
            f.write("\n".join(history))
      
        test_ppl = reconstruction(encoder, decoder, opt, device, dataloader_test, Test, recon_dir, ep, args.prior_mu, args.prior_logvar, alpha, beta, df)

        if args.save and eval_ppl < Best_ppl:
            Best_ppl = eval_ppl

            state = {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
            }
            torch.save(state, model_dir + 'TWRvae.tch')
            state2 = {
                "opt": opt.state_dict(),
                "ep": ep
            }
            torch.save(state2, model_dir + 'TWRvae.tchopt')
            text = []

