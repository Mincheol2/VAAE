# VAAE : Image Dataset

This is the modification of Vanila-VAE for my DL projects.

I contruct original VAE model (w\ MLP layer) and add new codes to help my projects.

## Requirements

*torchmetrics* is used for caculating SSIM score. There is no any other specific package.

To install this package,
```
pip install torchmetrics 
```


## Usage

- For our experiments, you just change the below four arguments.

--alpha : Parameter for alpha-divergence. (Default = 1.0)

--beta : Weight for alpha-divergence. ( Default = 1.0)

--df : Paramter for gamma-divergence. (Default = 0)

--epochs : Paramter for gamma-divergence. (Default = 100)

To train the model,

```
python main.py -dt mnist --epochs 50 --alpha 1.0 --beta 1.0 #Default KL Div
```

If you test gamma-divergence, use positive-valued **df** instead. 

```
python main.py -dt mnist --epochs 50 --beta 1.0 --df 3.0 #Gamma Div

```

## Loss.py : generalized gamma and alpha-divergence module

- We can use generalized alpha divergence and gamma divergence. (For details, see that file.)

- Compared to text data, please consider the difference of dimension. (image = 2, text = 3)


## Encoder.py : Reparametrize trick with normal and T distribution

- If you use gamma divergence (i.e. df > 2), prior and posterior are **student T distribution**, not normal.

- Ohter divergences use normal distribution, like the original model.

```
if self.df == 0:
    eps = torch.randn_like(std) # Normal dist
else:
    Tdist = torch.distributions.studentT.StudentT(self.df)
    eps = Tdist.sample() # Student T dist
```

### Result (MNIST)
TBU

## Reference

### Pytorch-ssim

https://github.com/Po-Hsun-Su/pytorch-ssim

### MNIST-C: A Robustness Benchmark for Computer Vision

Paper : https://arxiv.org/abs/1906.02337

Repo : https://github.com/google-research/mnist-c
