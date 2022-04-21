Based on the approximate formal derivation of the loss function based on VAE theory, I did more experiments and found out very important
results. Also, in order to have the best reliability and validity for our results, we really need to find a best threshold. We cannot use
the test set for this, as there is a rule that we can never use the test set during the development of a model, and we only use it
to measure the performance of the model after the model is complete.  But we can use a valid set to determine the best threshold, 
and this is what we should do.

Based on these, I made updated versions of the best templates that I think we should use. I made two sets of them: one set for 
two digits (categories) as normal and one digit as abnormal and another set for all digits except one as normal. I put each set 
in its own folder (with names “Updated_templates_2_Nl_1_Abn” and “Updated_templates_1_Abn_rest_Nl”) both on our shared google drive 
and github.

In each set there are five experiments that I explain below.

Briefly, the most important changes are as follows:

1) In the first cells, a validation set is created by separating 10% of the training data with the train_test_split function of SciKit learn.
2) A new file is created for a VAE with no latent loss added to the loss (beta of 0, please see https://www.semanticscholar.org/ 
   paper/beta-VAE%3A-Learning-Basic-Visual-Concepts-with-a-Higgins-Matthey/a90226c41b79f8b06007609f39f82757073641e2). 
   This VAE just have reconstruction loss. We need this, because removing the latent loss in VAE, decreases the reconstruction loss in VAE 
   and may even increase the accuracy of anomaly detection.
3) The loss function in the Modified VAE is changed to a corrected loss function derived based on VAE theory. The loss function is 
   -0.5 * (sum1 + sum2), where sum1 = sum over [1 + p (log(variance_1) – (mean_1)^2 – variance_1], 
   and sum2 = sum over [1 + p (log(variance_1) – (mean_1)^2 – variance_1]. 
   The approximation to the posterior distribution is a mixture of two Gaussians (p * MVN_1 + (1-p) * MVN_2). 
   I changed p to be a single number because it is better to model the data as two MVN’s that are being summed together with 
   a fraction p of one of them and (1-p) of the other, rather than having a different fraction of each of them in each of the dimensions.  
   In the experiments that I did, this loss function resulted in better accuracy of anomaly detection by a small margin.
   
   Taking the maximum of sum1 + sum2 for the latent loss is not accurate based on VAE theory, and for some reason it appears that 
   this loss is ignored by Keras when doing gradient descent on the loss function. When I multiply the latent loss by 32, 
   the reconstructed images do not become blurry and there is no change in reconstruction loss, where they should be, 
   which means that this loss is ignored in gradient descent. So, it is as if there is no latent loss (beta = 0).  
 
   The following lines are changed: 
   
   codings_p = keras.layers.Dense(1, activation='sigmoid')(z)   [1 instead of codings_size]
   p_mean = K.mean(codings_p)
   array1 = p_mean*(codings_log_var_1 - K.exp(codings_log_var_1) - K.square(codings_mean_1))
   array2 = (1-p_mean)*(codings_log_var_2 - K.exp(codings_log_var_2) - K.square(codings_mean_2))
   sum1 = K.sum(1 + array1, axis=-1)
   sum2 = K.sum(1 + array2, axis=-1)
   latent_loss = -0.5 * (sum1 + sum2)

   latent_loss = latent_loss * 0.5        (or latent_loss = latent_loss * 16)        

4) Two Modified VAE files: 
   
   One with the above latent loss multiplied by 0.5 (beta of 0.5) as the accuracy of anomaly detection was better with this factor 
   by a small margin, 
   
   And the other one with the above latent loss multiplied by 16 (beta of 16) to show that this results in 
   a large decrease in the accuracy of detection.
   
5) The validation set is used as validation data for training the network (to gauge overfitting):
history = variational_ae.fit(normal_data, normal_data, epochs=100, batch_size=16, 
               validation_data=(normal_valid_data, normal_valid_data), shuffle=True)
               
6) In order to have the most reliable and valid results, I wrote more code to calculate the threshold that gives the best accuracy of 
   anomaly detection based on the validation set. This is threshold5 which is used as the threshold. Then, the accuracy of 
   anomaly detection is determined on the test set with this threshold. It will give us reliable and valid comparison data for 
   all experiments and for all data sets.
   
   For this part, you can replace everything after the plot of the history of the network’s training and validation loss with the new code.

I think we should run the experiments as follows:

For MNIST: 

For the 10 combinations of one digit as abnormal and the rest of the digits as normal:
Average of 3 experiments for AE
Average of 2 experiments for VAE
Average of 2 experiments for VAE without latent loss
Average of 3 experiments for Modified VAE with latent loss * 0.5
1 experiment for Modified VAE with latent loss * 16

For 2 selected combinations of two digits as normal and one digit as normal [selected heuristically]:
The same

For 2 selected combinations of one digit as normal and one digit as normal [selected heuristically]:
The same

(This will be 154 experiments)

For Fashion MNIST:

Replace 10 combination of “one as normal and the rest as normal” with heuristic combinations of “two or more as normal and one as abnormal”.

The rest the same.

For CIFAR-10: 

Similarly.

For ECG5000, there are less experiments to do as there is just one combination of normal and abnormal, but we can do more experiments 
for this combination for better results, for example:

Average of 10 experiments for AE
Average of 10 experiments for VAE
Average of 10 experiments for VAE without latent loss
Average of 10 experiments for Modified VAE with latent loss * 0.5
1 experiment for Modified VAE with latent loss * 16

(This will be 41 experiments)

More experiments as needed.

The best way to do these stochastic experiments is to make a copy of each file for each experiment and then run 
the stochastic experiment on Colab. The resulting file serves as both the record of the experiment and the source of the data. 
Since, the experiment is stochastic, the data contained in the file cannot be reproduced in another run, and so 
a copy needs to be made for each experiment. 
