# unet_generative
UNet generative neural network for rotoscoping.

# Branches:
* **dev1-frozen-enc-dec-latent-loss**: Original experiments. It contains encoder and decoder from stable diffusion fixed. All the optimization and training is done in the latent space, even the loss fuction is between latent target and latent output. 
* **dev2-frozen-enc-dec-output-loss**: it is the same as dev1, but the output is between the actual output and the target. 
* **dev3-frozen-enc-scratch-dec-output-loss**: The decoder now is done from scratch. Just keeping the encoder fixed. 
