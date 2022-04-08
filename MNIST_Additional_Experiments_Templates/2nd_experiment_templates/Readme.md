In all the three templates the following consistent modifications were done and the experiment was run:

Digits 2 and 4 were used as normal instead of 4. The abnormal digit 8 (the same).

The bottleneck dimension was decreased to 16 from 32 (in the case of VAE and modified VAE this is the coding size).

A dense layer of 256 was added to the encoder and decoder.

Batch size decreased to 16 from 512.

The number of epochs was the same (100).

Overfitting needs to be avoided.  Our gauge for detecting overfitting is the plot of validation loss. Validation loss is not used in training but we observe it for this purpose.  It is recorded in the history of the fit of the neural net and shown on the plot.

We should use the normal instances in the test set to gauge validation loss, because when it starts to go higher, we have overfitting. For this reason, I changed the validation_data to be normal_test_data. 
