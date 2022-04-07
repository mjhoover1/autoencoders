From Slack:

### (about the sampling layer, reparametrization trick, and implementation of this in the modified VAE)

For the regular VAE, let say the coding size (the number of dimensions of the MVN we are training in the sampling layer) is 10.  
The layers for the means and stds (each having the dimension of the coding size = 10) are parallel in the sampling layer of VAE. 
These are being trained by gradient descent on the loss function and backpropagation is done through these two parallel 
layers during training. 

### For performing the reparametrization: 
we first draw a random stochastic vector (10-dim) from the MVN (0, 1) 
(with dimension of the coding size) where in each dimension the mean of the MVN distribution in that dimension is 0 and std is 1,
and then, in each of the dimensions we multiply the corresponding stochastic number by the std in that dimension from the stds layer
and add the mean from the means layer for that dimension.

This is the reparametrization trick. Because the source of stochasticity is MVN(0, 1) which is separate and we are not training. 
We just draw a vector from that in the forward pass and using the stds and means of our MVN under training produce
a stochastic 10-dim (coding-size) vector with the same result as if it was a stochastic vector that we draw directly 
from the MVN under training.

Then this produced 10-dim vector is given as the input to the first layer of the decoder.
In the case of the modified VAE, the distribution we have (the superposition of two MVNs) requires 5 parameter vectors 
(each set 10-dim (dimension of the coding size): the 10-dim vectors of mean1, std1, mean2, std2, and a paramter to determine 
the 10 (coding size dim) ratios that we add up each of the MVNs in the mix in each of the 10 dimensions (coding size dim). 
For the 5 parameter vectors we need 5 parallel layers to do the reparametrization trick in a fashion similar to the above.

(These ratios are like p's in each dimension where we can add 
(p * first normal distribution + (1 - p) * second normal distribution) 
to have the superposition or mix of two normals distributions that are normalized (with the integral from -inf to +inf = 1). 
I just used a ratio in each dimension (which does not normalize the distributions). 
But we should do an experiment with normalizing the mixture using p and 1-p, because for that to remain 
a probability distribution by definition we should do that. I will do that now, because that is more correct.)

### The reparametrization trick for the modified VAE is done following the same principle.  
But we need two MVN(0, 1)'s as the source of stochasiticity for two random MVN(0,1) 10-dim (coding size dim) 
vectors to use to produce a final vector from the two MVNs (each with a means vector and stds vector)
and the ratio vector (will replace with p's vector) under training, where this final vector is as if we drew 
a random vector from the 10-dim probability distribution of the mixture of the two MVNs under training in each forward pass.

### changed 
(first normal distribution + second normal distribution * ratio) to (p * first normal distribution + (1 - p) * second normal distribution) 
to make it normalized (as that is more correct as a probability distribution).

I did some experiments that shows that using p and 1-p as compared with using just a ratio doesn't really make a difference. 
And this makes sense because even though with a ratio the resulting distribution is not normalized and technically not 
a formal probability distribution because the integral of p dx from -inf to +inf is not 1, but still the probability of 
drawing a vector in an interval from the distribution is the same as the normalized case with 1-p/p = the ratio.

But replaced one ratio with p and 1-p. But we may still need to enforce in some way that 0 < p < 1, because right now 
the Keras network initializes p's with random weights and then trains them. So, it is possible that p ends up 
being negative or more than 1.
