# Unlearning

A set of experiments where a neural network should "forget" a subset of samples.
### Idea
A solution should be architecture agnostic. We just train a parallel branch named Controller
which learns how a sample affects the gradient. For this purpose we count gradients per sample.
Finally, when we want to unlearn some specific subset, we predict their impacts on the gradient
and substitute these gradients from the model parameters.

In general, this approach gives positive result
```
AUC no poisson 1.0
AUC poisson 0.858974358974359
AUC healed 0.9807692307692308
```
However, for some seeds this has no effect

Under development
