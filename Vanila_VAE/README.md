# VAAE : Image Dataset

# Modified-TWR-VAE

This is the modification of Vanila-VAE for my DL projects.

I contruct original VAE model (w\ MLP layer) and add new codes to help my projects.


## Usage

- For our experiment, you change the below three arguments.

--alpha : Parameter for alpha-divergence. (Default = 1.0)

--beta : Weight for alpha-divergence. ( Default = 1.0)

--df : Paramter for gamma-divergence. (Default = 0)

To train the model,

```
python main.py -dt ptb --alpha 1.0 --beta 1.0 #Default KL Div
```

If you test gamma-divergence, use positive-valued **df** instead. 

```
python main.py -dt ptb --beta 1.0 --df 1 #Gamma Div

```

## Loss.py : generalized gamma and alpha-divergence module

- We can use generalized alpha-divergence. (For details, see that file.)


## Encoder.py : Reparametrize trick with normal and T distribution

- If you use gamma divergence (i.e. df > 0), prior and posterior are **student T distribution**, not normal.

- Ohter divergence use normal distribution, like the original model.

```
if self.df == 0:
    eps = torch.randn_like(std) # Normal dist
else:
    Tdist = torch.distributions.studentT.StudentT(self.df)
    eps = Tdist.sample() # Student T dist
```

### Result (MNIST)
TBU
