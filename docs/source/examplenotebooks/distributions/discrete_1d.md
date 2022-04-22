# Discrete 1D Distribution

Let's use `UQpy.distributions` to manipulate a 1D, univariate, discrete distributon.

Through this example, we'll complete the following tasks:

 - Construct a built-in discrete univariate distribution
 - View the distribution's moments
 - Draw random samples from the distribution

```{tip}
Refer to the [Continuous 1D](./continuous_1d.md) example for a more detailed, step-by-step guide to using distributions in general.
```

Out of UQpy's built-in discrete distributions, we'll use the `Binomial` class. With that, we'll use Matplotlib to plot results.

```{code-cell} ipython3
from UQpy.distributions import Binomial

import matplotlib.pyplot as plt
```

Now, we can construct a binomial distribution `dist` with a number of trials `n` equal to five and a success probability `p` of 40% for each trial.

```{code-cell} ipython3
dist = Binomial(n=5, p=0.4)

dist.parameters
```

Let's retreive the distribution's moments using its `moments` method. Alone, this method returns the moments in an array in a default order or in the order specified by the user in the method's first argument.

Here, we're matching up the reported moments with their names for clarity.

```{code-cell} ipython3
{name: float(value)
 for value, name in zip(dist.moments(),
                        ["mean",
                         "variance",
                         "skewness",
                         "kurtosis"])}
```

Finally, we'll use the `rvs` method to generate 5000 random samples as an array `samples` and plot them.

```{code-cell} ipython3
samples = dist.rvs(nsamples=5000)
```

The shape of `samples` is `(nsamples, 1)`.

```{code-cell} ipython3
samples.shape
```

Now that we have our samples, let's view them in a histogram.

```{code-cell} ipython3
plt.hist(samples, bins=6)
plt.xlabel("x")
plt.ylabel("count")
plt.show()
```
