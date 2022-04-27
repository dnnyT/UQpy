# Multivariate From Independent Marginals & Copula

Let's use distribution classes in `UQpy.distributions` to define and manipulate a multivariate distribution defined by two independent marginal distributions and a copula.

Throughout, we'll accomplish the following:

- Relate two marginals in a bivariate distribution using a UQpy-supoorted copula
- Plot the bivariate distribution's PDF
- Modify the distribution's parameters

:::{admonition,tip}
Refer to the [Continuous 1D](./continuous_1d.md) example for a step-by-step guide to using distributions and to the [Multivariate From Independent Marginals](./joint_independent.md) example for detailed info on multivariate distributions.
:::

In addition to using a normal marginals and a Gumbel copula, UQpy's `JointCopula` class will provide us with the functionality to define a multivariate distribution with covarying marginals. Additionally, we'll use the `JointIndependent` class to compare our covariant distribution to an independent distribution. Numpy and Matplotlib will allow us extra mathematical as well as plotting abilities.

```{code-cell} ipython3
from UQpy.distributions import Normal, Gumbel, JointCopula, JointIndependent

import numpy as np
import matplotlib.pyplot as plt
```

Consider a collection `marginals` of two normal marginals, both with location parameter 0 and scale parameter 1.

```{code-cell} ipython3
marginals = [Normal(loc=0, scale=1),
             Normal(loc=0, scale=1)]
```

Let's say that those two marginals are related by correlation coefficient $\theta=3$ as described by a Gumbel copula. This in mind, we can construct a Gumbel copula object `copula` describing that kind of relationship.

```{code-cell} ipython3
copula = Gumbel(theta=3.)
```

Bundling this together, we have all the info we need to define a joint copula via the `JointCopula` class. We'll supply the marginals and the copula to construct a distribution object `covariant` to represent this bivariate distribution.

Like other descendants of the `DistributionND` class (as explained in the [Multivariate From Independent Marginals](./joint_independent.md) example), one way to view the parameters of the distribution is using the `get_parameters` method. Along with the marginals' individual parameters, we also get the value of $\theta$—keyed `theta_c`.

```{code-cell} ipython3
covariant = JointCopula(marginals, copula)

covariant.get_parameters()
```

To compare to a joint distribution with independent marginals, we'll construct a `JointIndependent` object from the marginals as well.

```{code-cell} ipython3
independent = JointIndependent(marginals)

independent.get_parameters()
```

Now, let's plot the two distributions side-by-side. This plot procedure has been consolidated to a `plot` method to perform again later.

```{code-cell} ipython3
def plot():
    fig, ax = plt.subplots(ncols=2, figsize=(10, 4))

    fig.suptitle("Joint PDF Contours")

    x = y = np.arange(-3, 3, 0.1)
    X, Y = np.meshgrid(x, y)

    for axes, dist, label in zip(ax,
                                [independent, covariant],
                                ["Independent Normal Marginals",
                                "Normal Marginals With Gumbel Copula ($\\theta=3$)"]):

        Z = dist.pdf(x=np.concatenate([X.reshape((-1, 1)), Y.reshape((-1, 1))],
                                    axis=1)).reshape(X.shape)
        CS = axes.contour(X, Y, Z)
        axes.clabel(CS, inline=1, fontsize=10)
        axes.set_title(label)

    plt.show()

plot()
```

Finally, we'll try changing some of the covariant distribution's parameters. The interface for this is the same as with other `DistributionND` classes: the `update_parameters` method, supplied with the parameter name to update (as keyed in the results of `get_parameters`) and the value update each parameter to.

Here, we're updating the correlation coefficient.

```{code-cell} ipython3
covariant.update_parameters(theta_c=2.)

covariant.get_parameters()
```

Plotting the distribution shows its new PDF contour.

```{code-cell} ipython3
plot()
```
