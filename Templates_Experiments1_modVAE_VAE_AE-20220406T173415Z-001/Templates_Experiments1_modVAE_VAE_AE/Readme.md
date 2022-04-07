Please see the Readme files in MNIST_Initial_Files folder also.

From slack:

The results of all these experiments are summarized in the three template files in the template_experiment1_modVAE_VAE_AE folder in the root
directory (corrections needed to be made that I explain below). I added references, initial descriptions, titles, and comments to these files. 
For the experiments with the other data sets, we can make a copy of these templates and modify them as needed and run the experiments.
Please use these templates to move forward with the experiments on Fashion MNIST and CIFAR.

The following things needed to be corrected:

The threshold that we should use should be determined by just the distribution of the reconstruction loss of the training data, 
which in this experimental design that we are using are just normal instances.

(In a real application of the anomaly detection, the model is trained on real data also, for which we do not know the labels. 
But because the percentage of the abnormal instances are small in the real data also, the model will mostly learn the normal 
data in that case also.)  

The calculation of the threshold should not include threshold2 from the distribution of the reconstruction loss for the test data, 
because in a real application, we use the model to test the new test instances to determine if they are abnormal one by one as we get them, 
and so we do not know this distrubtion.

I did some more experiments and setting the threshold as threshold1 = mean + 2.5 * std of the distribution of the reconstruction loss 
of the training data resulted in a better balance of precision and recall and the best accuracy as far as the experiments showed. 
If the case is that we do not want to miss any anomaly, then we want higher precision and setting the threshold = mean + 1 * std of 
this distribution would be more appropriate, but in that case, a higher percentage of normal instances will be labeled as abnormal 
resulting in less recall.

The other thing is that I did more experiments that showed that:

- AE with ‘optimizer=’rmsprop’ and ‘loss =binary_crossentropy’ has much less reconstruction loss compared to AE with 
  ‘optimizer=’adam’ and loss=’mae’
- 
- For AE with ‘optimizer=’rmsprop’ and ‘loss =binary_crossentropy’, the reconstruction loss is actually less than VAE 
  with all the other experimental conditions kept the same (please see the files in templates_experiments1 folder)
- 
- But this should be viewed in light of the fact, that is known, that VAE performs better with smaller batch sizes, and keeping 
  the experimental condition of batch size of VAE and AE the same (=512) is not good for the VAE.
- 
- But accuracy of anomaly detection is almost the same for both of them (at least for MNIST) for the reason that I explained above.
- 
- But reconstruction loss of the modified VAE is still better than AE with ‘optimizer=’rmsprop’ and ‘loss =binary_crossentropy’,
  even with batch size of 512 (same as the AE). Accuracy of anomaly detection of the modified VAE is also still better by a small margin.
