# Modified-TWR-VAE

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

--beta : Weight for alpha-divergence. ( Default = 1.0)

--df : Paramter for gamma-divergence. (Default = 0)

If you test gamma-divergence, use positive-valued **df** instead. 

```
python main.py -dt ptb --beta 1.0 --df 1 #Gamma Div
```
## Run the model in Colab & Visualization

- See [TWR_VAE_colab.ipynb](https://github.com/Mincheol2/modified-TWR_VAE/blob/main/TWR_VAE_colab.ipynb)



## Loss.py : generalized alpha-divergence module

- We can use generalized alpha-divergence. (For details, see that file.)


### Result (& generate loss.txt)

- Use tqdm time bar : In training process, you can see the remaining time.

<img width="1000" alt="image" src="https://user-images.githubusercontent.com/43122330/200512800-a28aa7b4-1293-4981-9333-206ea7e4d833.png">


- I think we don't need to use 'valid' set in the original model. So we use only train/test set.

- This model generates both train_loss.txt and **test_loss.txt**


### Noticeable results(*Need more experiments with different settings.*)

1) reversed KL(alpha = 0) << alpha div // ptb, epoch = 150 (in terms of stability)

![image](https://user-images.githubusercontent.com/43122330/201835778-7f67a418-ce56-4f11-9636-23a32b0ebafe.png)

2) How about KL(alpha = 1) vs alpha 0.9 ? 

![image](https://user-images.githubusercontent.com/43122330/201835644-d2e968b8-a084-4068-9b09-9625e6bf740b.png)
 
3) alpha 0.5 ~ 0.7 are not so good..

![image](https://user-images.githubusercontent.com/43122330/201836137-3ad3e2d8-ef76-4edf-9f22-e148878b58ce.png)

![image](https://user-images.githubusercontent.com/43122330/201836252-9d2428ed-4664-4d1e-a301-3ffb1a3056e3.png)
