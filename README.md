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

--alpha : Parameter for alpha-divergence. ( Default = 1.0)

--beta : Weight for alpha-divergence. ( Default = 1.0)



## Run the model in Colab

- See TWR_VAE_colab.ipynb



## Loss.py : generalized alpha-divergence module

- We can use generalized alpha-divergence. (For details, see that file.)


### Result (& generate loss.txt)

- Use tqdm time bar : In training process, you can see the remaining time.

<img width="1000" alt="image" src="https://user-images.githubusercontent.com/43122330/200512800-a28aa7b4-1293-4981-9333-206ea7e4d833.png">


- I think we don't need to use 'valid' set in the original model. So we use only train/test set.

- This model generates both train_loss.txt and **test_loss.txt**


### Plot.ipynb

- I'll upload plot.ipynb which helps to make loss plot.

