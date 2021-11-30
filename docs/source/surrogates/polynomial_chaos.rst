
Polynomial Chaos Expansion - PCE
----------------------------------------

Polynomial Chaos Expansions (PCE) represent a class of methods which employ orthonormal polynomials to construct approximate response surfaces (metamodels or surrogate models) to identify a mapping between inputs and outputs of a numerical model [2]_. PCE methods can be directly used for moment estimation and sensitivity analysis (Sobol indices). A PCE object can be instantiated from the class ``PCE``. The method can be used for models of both one-dimensional and multi-dimensional outputs.

Let us consider a computational model :math:`Y = \mathcal{M}(x)`, with :math:`Y \in \mathbb{R}` and a random vector with independent components :math:`X \in \mathbb{R}^M` described by the joint probability density function :math:`f_X`. The polynomial chaos expansion of :math:`\mathcal{M}(x)` is

.. math:: Y = \mathcal{M}(x) = \sum_{\alpha \in \mathbb{N}^M} y_{\alpha} \Psi_{\alpha} (X)

where the :math:`\Psi_{\alpha}(X)` are multivariate polynomials orthonormal with respect to :math:`f_X` and :math:`y_{\alpha} \in \mathbb{R}` are the corresponding coefficients.

Practically, the above sum needs to be truncated to a finite sum so that :math:`\alpha \in A` where :math:`A \subset \mathbb{N}^M`. The polynomial basis :math:`\Psi_{\alpha}(X)` is built from a set of *univariate orthonormal polynomials* :math:`\phi_j^{i}(x_i)` which satisfy the following relation

.. math:: \Big< \phi_j^{i}(x_i),\phi_k^{i}(x_i) \Big> = \int_{D_{X_i}} \phi_j^{i}(x_i),\phi_k^{i}(x_i) f_{X_i}(x_i)dx_i = \delta_{jk}

The multivariate polynomials :math:`\Psi_{\alpha}(X)` are assembled as the tensor product of their univariate counterparts as follows

.. math:: \Psi_{\alpha}(X) = \prod_{i=1}^M \phi_{\alpha_i}^{i}(x_i)

which are also orthonormal.


PCE Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: UQpy.surrogates.polynomial_chaos.PCE.PolyChaosExp
    :members:

Univariate Orthonormal Polynomials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Different families of univariate polynomials can be used for the PCE method. These polynomials must always be orthonormal with respect to the arbitrary distribution. In UQpy, two families of polynomials are currently available, namely the ``Legendre`` and ``Hermite`` polynomials, which are appropriate for data generated from a Uniform and a Normal distribution respectively.

Univariate Chaos Polynomials Class Description
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: UQpy.surrogates.polynomial_chaos.ChaosPolynomials.ChaosPolynomial1d
    :members:

Multivariate Orthonormal Polynomials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Multivariate chaos polynomials are constructed as products of univariate chaos polynomials of given polynomial degrees. These polynomials are orthonormal with respect to the joint probability distribution that characterises the input random variables (RVs), which are assumed to be mutually independent.

Multivariate Chaos Polynomials Class Description
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: UQpy.surrogates.ChaosPolynomialNd
    :members:

Calculation of the PCE coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Several methods exist for the calculation of the PCE coefficients. In UQpy, three non-intrusive methods can be used, namely least squares regression (``fit_lstsq`` function), LASSO regression (``fit_lasso`` function) and ridge regression (``fit_ridge`` function).


Least Squares Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Least Squares regression is a method for estimating the parameters of a linear regression model. The goal is to minimize the sum of squares of the differences of the observed dependent variable and the predictions of the regression model. In other words, we seek for the vector :math:`\beta`, that approximatively solves the equation :math:`X \beta \approx y`. If matrix :math:`X` is square then the solution is exact.

If we assume that the system cannot be solved exactly, since the number of equations :math:`n` is not equal to the number of unknowns :math:`p`, we are seeking the solution that is associated with the smallest difference between the right-hand-side and left-hand-side of the equation. Therefore, we are looking for the solution that satisfies the following

.. math:: \hat{\beta} = \underset{\beta}{\arg\min} \| y - X \beta \|_{2}

where :math:`\| \cdot \|_{2}` is the standard :math:`L^{2}` norm in the :math:`n`-dimensional Eucledian space :math:`\mathbb{R}^{n}`. The above function is also known as the cost function of the linear regression.

The equation may be under-, well-, or over-determined. In the context of Polynomial Chaos Expansion (PCE) the computed vector corresponds to the polynomial coefficients. The above method can be used from the function ``fit_lstsq``.


fit_lstsq Function Description
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: UQpy.surrogates.fit_lstsq


Lasso Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A drawback of using Least Squares regression for calculating the PCE coefficients, is that this method considers all the features (polynomials) to be equally relevant for the prediction. This technique often results to overfitting and complex models that do not have the ability to generalize well on unseen data. For this reason, the Least Absolute Shrinkage and Selection Operator or LASSO can be employed (from the ``fit_lasso`` function). This method, introduces an :math:`L_{1}` penalty term (which encourages sparcity) in the loss function of linear regression as follows

.. math:: \hat{\beta} = \underset{\beta}{\arg\min} \{ \frac{1}{N} \| y - X \beta \|_{2} + \lambda \| \beta \|_{1} \}


where :math:`\lambda` is called the regularization strength.

Parameter :math:`\lambda` controls the level of penalization. When it is close to zero, Lasso regression is identical to Least Squares regression, while in the extreme case when it is set to be infinite all coefficients are equal to zero.

The Lasso regression model needs to be trained on the data, and for this gradient descent is used for the optimization of coefficients. In gradient descent, the gradient of the loss function with respect to the weights/coefficients :math:`\nabla Loss_{\beta}` is used and deducted from :math:`\beta^{i}` at each iteration as follows

.. math:: \beta^{i+1} = \beta^{i} - \epsilon \nabla Loss_{\beta}^{i}

where :math:`i` is the iteration step, and :math:`\epsilon` is the learning rate (gradient descent step) with a value larger than zero.


fit_lasso Function Description
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: UQpy.surrogates.CoefficientFit.fit_lasso


Ridge Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ridge regression (also known as :math:`L_{2}` regularization) is another variation of the linear regression method and a special case of the Tikhonov regularization. Similarly to the Lasso regression, it introduces an additional penalty term, however Ridge regression uses an :math:`L_{2}` norm in the loss function as follows

.. math:: \hat{\beta} = \underset{\beta}{\arg\min} \{ \frac{1}{N} \| y - X \beta \|_{2} + \lambda \| \beta \|_{2} \}

where :math:`\lambda` is called the regularization strength.

Due to the penalization of terms, Ridge regression constructs models that are less prone to overfitting. The level of penalization is similarly controlled by the hyperparameter :math:`\lambda` and the coefficients are optimized with gradient descent. The Ridge regression method can be used from the ``fit_ridge`` function.


fit_ridge Function Description
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: UQpy.surrogates.fit_ridge


Moment Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
There exist two tailored functions for the estimation of the first (mean) and second (variance) moments by post-processing the terms of the PCE, namely ``pce_mean`` and ``pce_variance``.

pce_mean Function Description
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: UQpy.surrogates.pce_mean

pce_variance Function Description
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: UQpy.surrogates.pce_variance



Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
There exist methods for univariate and multivariate sensitivity analysis (SA). Global, variance-based (Sobol) sensitivity indices are computed by post-processing the PCE terms.
The methods are:
``pce_sobol_first``: computes first-order sensitivity indices for a scalar QoI.
``pce_sobol_total``: computes total-order sensitivity indices for a scalar QoI.
``pce_generalized_sobol_first``: computes first-order sensitivity indices for a vector-valued QoI.
``pce_generalized_sobol_total``: computes total-order sensitivity indices for a vector-valued QoI.

pce_sobol_first Function Description
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: UQpy.surrogates.pce_sobol_first

pce_sobol_total Function Description
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: UQpy.surrogates.pce_sobol_total

pce_generalized_sobol_first Function Description
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: UQpy.surrogates.polynomial_chaos.SobolEstimation.pce_generalized_sobol_first

pce_generalized_sobol_total Function Description
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: UQpy.surrogates.polynomial_chaos.SobolEstimation.pce_generalized_sobol_total
