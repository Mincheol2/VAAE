# TWR-γAE : Timestep-Wise Regularisation with γ-divergence VAE


This repository is the implementaion of our final NLP project, by Mincheol Cho* and Juno Kim.

We edit some vague parts in the original baseline, and modify some archidecture designs.

The baseline is from [Here](https://github.com/ruizheliUOA/TWR-VAE/).

## Tutorials

1) [Effect of γ Divergence]

2) [Masking Task]

## Basic Usage

### Baseline : KL divergence.

- To train the baseline model,

```
python main.py -dt ptb #Default KL Div
```

- If you want to change the default parameters(epoch, zdim, .. etc.), see the main.py.


### Experiments : γ-divergence.

- For repoducing our experiments, you may fine-tune these arguments.

--beta : Weight for divergence loss. (Default = 1.0)

--df : Paramter for γ-divergence. (Default = 0, it means γ-divergence is not used)

--rnn : rnn architecture type. rnn/gru/lstm (Default : rnn)

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

### Loss.py : generalized γ and alpha-divergence module

- We implement γ-divergence and generalized KL divergence(alpha-divergence). (For details, see the loss.py.)


### Encoder.py : Reparametrize trick with normal and T distribution

- When using γ-divergence, prior and posterior should be **student T distribution**, not normal. (For conserving the flatness of manifold)

- In other case, use normal distribution, as the original model.

```
if self.df == 0:
    eps = torch.randn_like(std) # Normal dist
    
else:
    Tdist = torch.distributions.studentT.StudentT(self.df)
    eps = Tdist.sample() # Student T dist
```

###
