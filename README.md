# Modified-TWR-VAE

This is for the Final Project of Deep Learning : Statistics Perspective, 2022 Fall .

The baseline is from [Here](https://github.com/ruizheliUOA/TWR-VAE/), and I modified some vague codes, and add some convenient tools.

Note that there is only **lang-model part**, not dialogue part.

## What things are changed?

### Result (loss.txt)

- I think we don't need to use 'recon' set in this model. So this model become using only train/test set.

- This model generates both train_loss.txt and **test_loss.txt**


### Plot.ipynb

- For convenient analysis, I uploaded plot.ipynb, which helps to make loss plot.

### tqdm bar

- In training process, you can see the remaining time.



### Args

#### Add

--alpha : Parameter for alpha-divergence. ( Default = 1.0)

--seed : Set seed number (Default = 999)


#### Remove 

- remove '''--setting''' : This is for Teacher forcing. But I checked that baseline code didn't implement it.

cf) Teacher Forcing is the technique where the target word is passed as the next input to the decoder
