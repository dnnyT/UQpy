# Continuous 1D Distribution

In this example, we'll make use of a class in `UQpy.distributions` to manipulate a 1D, univariate, continuous distribution.

Throughout the example, we'll learn to:

- Construct a built-in univariate distribution
- Plot the distribution's PDF and Log PDF
- Modify its parameters
- View its moments
- Draw random samples

We'll be using UQpy's lognormal distribution class to exemplify how distribution classes work in UQpy. As well, we'll use Numpy for its math functionalities and random state management and Matplotlib to display results graphically.

```{code-cell} ipython3
from UQpy.distributions import Lognormal

import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt
```

Let's start by constructing a lognormal distribution `dist` with parameters of shape `s` equal to one, location `loc` equal to zero, and scale `scale` equal to $e^5$.

After construction, we can access our parameters via the `parameters` attribute.

```{code-cell} ipython3
dist = Lognormal(s=1,
                 loc=0,
                 scale=np.exp(5))

dist.parameters
```

We'll use the functions `dist.pdf` and `dist.log_pdf` to plot the distribution's PDF and Log PDF. Both functions accept a sequence of some number `nsamples` of values at which to report the probability density of the distribution. That sequence must be in the form of an `ndarray` or similar object of shape `(nsamples,)` or `(nsamples, 1)`; it must be a vector or a 1D array.

Here, we use a linear space `x` (approximately) constituting integers from zero to one thousand.

```{code-cell} ipython3
x = np.linspace(0.01, 1000, 1000).reshape((-1, 1))
```

On `x`, we can now plot the distribution's PDF and Log PDF.

In this example, we also bundle plotting into a function `plot` for later use.

```{code-cell} ipython3
def plot():
    fig, ax = plt.subplots(ncols=2, figsize=(15,4))

    for axes, name, f in zip(ax,
                            ["PDF", "Log PDF"],
                            [dist.pdf, dist.log_pdf]):
        axes.plot(x, f(x))
        axes.set_xlabel("x")
        axes.set_ylabel(name.lower() + "(x)")
        axes.set_title(name)

    plt.show()

plot()
```

Now, let's modify some of the distribution object's parameters. To modify parameters after construction, we use the object's `update_parameters` method with the same kinds of arguments we used to construct the distribution.

We'll change the location parameter to equal one hundred.

```{code-cell} ipython3
dist.update_parameters(loc=100)

dist.parameters
```

To visualize how the distribution has changed, let's plot it again.

```{code-cell} ipython3
plot()
```

To see a distribution's moments, we can use its `moments` method. By default, this method reports the moments in the following order:

1. Mean
2. Variance
3. Skewness
4. Kurtosis

Note that each value is technically wrapped as a zero-dimensional array, in accordance with SciPy's reporting of distribution moments.

```{code-cell} ipython3
dist.moments()
```

This is the same as calling the method specifying the order string `"mvsk"` as the argument.

Here, we'll unwrap the values and map them to their names for clarity.

```{code-cell} ipython3
{name: float(value)
 for value, name in zip(dist.moments("mvsk"),
                        ["mean",
                         "variance",
                         "skewness",
                         "kurtosis"])}
```

Finally, we can generate a collection of samples `samples` from the distribution via the `rvs` method. We'll generate five thousand samples by supplying the argument for number of samples `nsamples`.

For this example, we're also supplying a pseudorandom state `random_state` so that results are consistent each time this code snippet runs.

```{code-cell} ipython3
samples = dist.rvs(nsamples=5000,
                   random_state=RandomState(9))

samples
```

The output of `rvs` is an `ndarray` of shape `(nsamples, 1)`.

```{code-cell} ipython3
samples.shape
```

This in mind, we can represent the samples with a histogram.

```{code-cell} ipython3
plt.hist(samples[:, 0], bins=50)
plt.xlabel("x")
plt.ylabel("count")
plt.show()
```

Since we've taken samples according to the distribution, the histogram resembles the distribution's PDF.
