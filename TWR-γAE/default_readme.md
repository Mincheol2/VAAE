# Modified-TWR-VAE (VAAE for text dataset)

This is the modification of TWR-VAE(Timestep-Wise Regularization Variational AutoEncooder) for my NLP / DL projects.

I edit some vague parts in the original model and write new codes to help my projects.

Currently, there is only **lang-model part**, not dialogue part.

The baseline is from [Here](https://github.com/ruizheliUOA/TWR-VAE/)



## Usage

To train the model,

```
python main.py -dt ptb --alpha 1.0 --beta 1.0 #Default KL Div
```

- Compared to the origin, I add two arguments.

--alpha : Parameter for alpha-divergence. (Default = 1.0)

--beta : Weight for alpha-divergence. (Default = 1.0)

--df : Paramter for gamma-divergence. (Default = 0, it means gamma divergence is not used)

If you test gamma-divergence, use **df > 2**. (Because the variance of T distribution exists when df > 2)

```
python main.py -dt ptb --beta 1.0 --df 3 #Gamma Div
```

## Run the model in Colab & Visualization

- See [TWR_VAE_colab.ipynb](https://github.com/Mincheol2/modified-TWR_VAE/blob/main/TWR_VAE_colab.ipynb)



## Loss.py : generalized gamma and alpha-divergence module

- We can use generalized alpha-divergence. (For details, see that file.)


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

### Result (& generate loss.txt)

- Use tqdm time bar : In training process, you can see the remaining time.

<img width="1000" alt="image" src="https://user-images.githubusercontent.com/43122330/200512800-a28aa7b4-1293-4981-9333-206ea7e4d833.png">

- This model generates both train_loss.txt and **test_loss.txt**

- You can see the PPL plot and reconstructed sentence in TWR_VAE_colab.ipynb


### Noticeable results(*Need more experiments with different settings.*)

1) reversed KL(alpha = 0) << alpha div // ptb, epoch = 150 (in terms of stability)

![image](https://user-images.githubusercontent.com/43122330/201835778-7f67a418-ce56-4f11-9636-23a32b0ebafe.png)

2) How about KL(alpha = 1) vs alpha 0.9 ? 

![image](https://user-images.githubusercontent.com/43122330/201835644-d2e968b8-a084-4068-9b09-9625e6bf740b.png)
 
3) alpha 0.5 ~ 0.7 are not so good..

![image](https://user-images.githubusercontent.com/43122330/201836137-3ad3e2d8-ef76-4edf-9f22-e148878b58ce.png)

![image](https://user-images.githubusercontent.com/43122330/201836252-9d2428ed-4664-4d1e-a301-3ffb1a3056e3.png)
