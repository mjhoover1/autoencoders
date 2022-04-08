3rd experiment:

The Following changes were made consistently for all three templates (AE, VAE, mod VAE). 

All digits except 8 as normal. Digit 8 as abnormal. 

The bottleneck dimension of 16 (instead of 32). 

A dense layer of 256 was added to the encoder and decoder. 

Batch size decreased to 128 from 512 (We can do this with lower batch size of 16 or less, but because the number of instances is higher 
because of using all the digits in MNIST training set, it will take longer to do this experiment). 

The number of epochs was the same (100).

I will repeat this experiment with a lower batch size.

Overfitting needs to be avoided.  Our gauge for detecting overfitting is the plot of validation loss. Validation loss is not used in training but we observe it for this purpose.  It is recorded in the history of the fit of the neural net and shown on the plot.

We should use the normal instances in the test set to gauge validation loss, because when it starts to go higher, we have overfitting. For this reason, I changed the validation_data to be normal_test_data.
