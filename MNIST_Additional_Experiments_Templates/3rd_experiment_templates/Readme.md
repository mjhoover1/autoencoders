3rd experiment:

The Following changes were made consistently for all three templates (AE, VAE, mod VAE). 

All digits except 8 as normal. Digit 8 as abnormal. 

The bottleneck dimension of 16 (instead of 32). 

A dense layer of 256 was added to the encoder and decoder. 

Batch size decreased to 128 from 512 (We can do this with lower batch size of 16 or less, but because the number of instances is higher 
because of using all the digits in MNIST training set, it will take longer to do this experiment). 

The number of epochs was the same (100).

I will repeat this experiment with a lower batch size.
