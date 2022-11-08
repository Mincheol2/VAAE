# TWR-VAE

This is for the Final Project of Deep Learning : Statistics Perspective, 2022 Fall .

The baseline is from [Here](https://github.com/ruizheliUOA/TWR-VAE/), and I modified some vague codes, and add some convenient code.


## args 

### Add

--alpha : Parameter for alpha-divergence. ( Default = 1.0)

--seed : Set seed number (Default = 999)


### Remove 

- remove '''--setting''' : This is for Teacher forcing. But I checked that baseline code didn't implement it.

cf) Teacher Forcing is the technique where the target word is passed as the next input to the decoder
