# Multivariate From Independent Marginals

In this example, we use some distribution classes from `UQpy.distributions` to manipulate a multivariate distribution defined by two independent marginal distributions.

We'll learn to:

- Construct a multivariate distribution from built-in, independent marginals.
- Plot the multivariate distribution's PDF
- View the distribution's moments
- Draw random samples

:::{tip}
Refer to the [Continuous 1D](./continuous_1d.md) example for a more detailed, step-by-step guide to using distributions in general.
:::

To achieve our goals, we'll be using UQpy's joint independent distribution class in conjunction with its normal and lognormal distribution classes for our multivariate's marginals. We'll also make use of Numpy's math functionalities and 

```{code-cell} ipython3
from UQpy.distributions import Normal, Lognormal, JointIndependent

import numpy as np
import matplotlib.pyplot as plt
```

First, our joint distribution needs some marginal distributions. Let's define a collection `marginals` of marginals in which we have a `Normal` and `Lognormal` distribution object.

```{code-cell} ipython3
marginals = [Normal(loc=1, scale=2),
             Lognormal(s=2, loc=0, scale=np.exp(5))]
```

From these marginals, we can now construct a joint independent distribution object `dist`.

```{code-cell} ipython3
dist = JointIndependent(marginals)
```

The parameters of each independent marginal can be accessed as usual using each one's `parameters` attribute. After construction, the marginals are accessible in the `JointIndependent` object's `marginals` attribute.

```{code-cell} ipython3
[d.parameters for d in dist.marginals]
```

It's important to note that `JointIndependent` objects themselved (like other descendants of the `DistributionND` class) do not have a `parameters` attribute. Instead, we can get the marginals' parameters through the `JointIndependent` object using its `get_parameters` method.

```{code-cell} ipython3
dist.get_parameters()
```

Notice that each parameter is keyed to indicate the original parameter's key **and** the distribution it came from. This is realized with keys in the form: `"(parameter name)_(marginal)"`.

We can also update these parameters with the `update_parameters` method, using the same keys.

```{code-cell} ipython3
dist.update_parameters(loc_0=2, s_1=1)

print(dist.get_parameters())
```

Moments are queried the same way as with 1D distributions—using the `moments` method. However, they're also returned in a similar fashion to parameters. Here, each moment's entry is an array with one value per marginal.

Requesting only the mean moments (specified by argument `"m"`), this returns a single array. This array's first element is the first maginal's mean, and the second element is the second marginal's mean.

```{code-cell} ipython3
dist.moments("m")
```

Requesting multiple moments returns an array. Each subarray is just like the array of individual means.

```{code-cell} ipython3
dist.moments("mvsk")
```

Now, let's take 1000 samples from the distribution using its `rvs` method. These samples will be placed in the collection `samples`.

```{code-cell} ipython3
samples = dist.rvs(nsamples=1000)
```

Finally, we'll plot these samples alongside the contour of the PDF.

```{code-cell} ipython3
fig, ax = plt.subplots(ncols=2, figsize=(10, 4))

for axes in ax:
    axes.set_xlabel("dimension 1")
    axes.set_ylabel("dimension 2")
    axes.set_ylim([0, 500])
    axes.set_xlim([-2, 6])

ax[0].scatter(samples[:, 0], samples[:, 1], alpha=0.2)
ax[0].set_title('random samples')

X, Y = np.meshgrid(np.arange(-2, 6, 0.2),
                   np.arange(0.01, 500, 1))

Z = dist.pdf(x=np.concatenate([X.reshape((-1, 1)),
                               Y.reshape((-1, 1))],
             axis=1)).reshape(X.shape)

contour = ax[1].contour(X, Y, Z)

ax[1].clabel(contour, inline=1, fontsize=10)
ax[1].set_title('PDF contour')

plt.show()
```
