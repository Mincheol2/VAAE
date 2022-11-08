# Modified-TWR-VAE

This is the modification of TWR-VAE(Timestep-Wise Regularization Variational AutoEncooder) for my NLP / DL projects.

I edit some vague parts in original model, and write new codes to help my projects.

Currently, there is only **lang-model part**, not dialogue part.

The baseline is from [Here](https://github.com/ruizheliUOA/TWR-VAE/)



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

- Use tqdm time bar : In training process, you can see the remaining time.

<img width="1000" alt="image" src="https://user-images.githubusercontent.com/43122330/200512800-a28aa7b4-1293-4981-9333-206ea7e4d833.png">


- I think we don't need to use 'valid' set in the original model. So use only train/test set.

- This model generates both train_loss.txt and **test_loss.txt**


### Plot.ipynb

- I'll upload plot.ipynb which helps to make loss plot.

