# Modified-TWR-VAE

This is the modification of TWR-VAE(Timestep-Wise Regularization Variational AutoEncooder) for my NLP / DL projects.

The baseline is from [Here](https://github.com/ruizheliUOA/TWR-VAE/)

I edit some vague parts in original model, and write new codes to help my projects.

Currently, there is only **lang-model part**, not dialogue part.

## What things are changed?

To train the model,

```
python main.py -dt ptb --alpha 1 #KL Div
```

- Comparing to origin, I add two arguments and remove one argument.

--alpha : Parameter for alpha-divergence. ( Default = 1.0)

--seed : Set seed number (Default = 999)


And remove '''--setting''' : This is for Teacher forcing. But I checked this argument didn't work.

cf) Teacher Forcing is the technique where the target word is passed as the next input to the decoder


### Loss.py

- We can use generalized alpha-divergence. (For details, see that file.)


### Result (& generate loss.txt)

- Use tqdm time bar : It is very simple module. In training process, you can see the remaining time.

- I think we don't need to use 'recon' set in this model. We use only train/test set.

- This model generates both train_loss.txt and **test_loss.txt**


### Plot.ipynb

- I'll upload plot.ipynb which helps to make loss plot.

