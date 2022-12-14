# TWR-γAE : Timestep-Wise Regularisation with γ-divergence VAE


This repository is the implementaion of our final NLP project, by **Mincheol Cho** and **Juno Kim**.

We edit some vague parts in the original baseline, and modify some archidecture designs.

The baseline is [Here](https://github.com/ruizheliUOA/TWR-VAE/).

## Tutorial

[Effect of γ Divergence](https://github.com/Mincheol2/VAAE/blob/main/TWR-%CE%B3AE/TWR_VAE_colab.ipynb) : run the model & visulize the result.

### Subtask Code

- [Masking Task](https://github.com/Mincheol2/VAAE/blob/main/TWR-%CE%B3AE/TWR-VAE%20masking.ipynb) : masking task code.

- [Making metric plots](https://github.com/Mincheol2/VAAE/blob/main/TWR-%CE%B3AE/Make_PPLplot.ipynb): use pandas to summarize the result.

## Basic Usage

### Baseline : KL divergence.

- To train the baseline model,

```
python main.py -dt ptb #Default KL Div
```

- If you want to change the default parameters(epoch, zdim, .. etc.), see the main.py.


### Experiments : γ-divergence.

- For repoducing our experiments, you may fine-tune these arguments.

|argument|description|default value|
|------|---|---|
|--beta|Weight for divergence loss. |1.0|
|--df |Paramter for γ-divergence.|1.0|
|--rnn|rnn architecture type : rnn/gru/lstm|'rnn'|
|--epochs| the number of epochs| 100 |

- If you test γ-divergence, please use **df > 2**. (Because the variance of T distribution exists when df > 2)

```
python main.py -dt ptb --beta 1.0 --df 3 #Gamma Div
```

cf) We also implement alpha-divergence. If you are intereted in it, use the argument alpha.

--alpha : Parameter for alpha-divergence. (Default = 1.0)


## Result generation

- This model updates **train_TWRvae_loss.txt** in the train time. 

- In the test time, both **test_TWRvae_loss.txt** and **TWRvae_outcome.txt** are updated.

- You can see the metrics in loss.txt, and see reconstructed sentences(per 10 epochs) in outcome.txt. 



## Implementation Details

### Main.py & Train.py

- Split the setup part(main.py) and the train part(train.py).

- We modify some logic, so that running is faster than the baseline.

### Loss.py : generalized γ and alpha-divergence module

- We implement γ-divergence and generalized KL divergence(alpha-divergence).

```
def gamma_divergence(self, df):
    # Check the well-definedness
    if df <= 2:
        raise Exception(f'the degree of freedom is not larger than 2. Divergence is not well-defined.')

    # dimension : [T : timestep, B : Batch, Z : z_dim]
    zdim = self.post_mu.shape[2]
    log_det_ratio = (df + zdim) / (2*(df + zdim - 2)) * (torch.sum(self.prior_logvar,dim=2) - torch.sum(self.post_logvar,dim=2))
    log_term = (df + zdim)/2 * torch.log(1 + 1/(df-2) * torch.sum(self.post_var / self.prior_var,dim=2) + 1/df * torch.sum( (self.prior_mu-self.post_mu).pow(2) / self.prior_var,dim=2))

    gamma_div = torch.mean(log_det_ratio + log_term) # Batch mean
    return gamma_div
```

### Encoder.py

#### Reparametrize trick with normal and T distribution

- When using γ-divergence, prior and posterior should be **student T distribution**, not normal. (For conserving the flatness of manifold)

- In other case, use normal distribution, as the original model.

```
if self.df == 0:
    eps = torch.randn_like(std) # Normal dist
else:
    Tdist = torch.distributions.studentT.StudentT(self.df)
    eps = Tdist.sample(sample_shape = torch.Size(mu.shape)).to(self.device) # Student T dist
```

#### Caclulate Mutual Information in t-distribution

- Because of the same reason, there are two versions of calculate MI. : cal_normal_mi and cal_tdist_mi. 

```
def cal_tdist_mi(self, last_mu, last_logvar):
    ...
    df_const = 0.5*(self.df+nz)
    log_determinant = 0.5*last_logvar.sum(-1)
    log_gamma_diff = scp.loggamma(df_const) - scp.loggamma(self.df/2) - 0.5*nz*math.log(self.df*math.pi)
    digamma_diff = scp.digamma(df_const) - scp.digamma(self.df/2)
    ...
    # E_{q(z|x)}log(q(z|x))
    neg_entropy = (log_gamma_diff - df_const * digamma_diff - log_determinant).mean()
    log_density = - df_const * torch.log(1+(1/self.df)*((dev ** 2) / var).sum(dim=-1)) + log_gamma_diff - log_determinant
```

### decoder.py

- Implement rnn-based decoder as the baseline.

### BatchIter.py / Corpus.py

- Tokenization and data preprocessing code. (from the baseline code)
