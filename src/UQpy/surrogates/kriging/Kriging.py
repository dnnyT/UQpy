import logging
import numpy as np
from scipy.linalg import cholesky
import scipy.stats as stats
from beartype import beartype
import inspect
from scipy.optimize import minimize

from UQpy.surrogates.baseclass.Surrogate import Surrogate
from UQpy.utilities.ValidationTypes import RandomStateType
from UQpy.surrogates.kriging.correlation_models.baseclass.Correlation import Correlation
from UQpy.surrogates.kriging.regression_models.baseclass.Regression import Regression
from UQpy.surrogates.kriging.constraints.baseclass.Constraints import Constraints


class Kriging(Surrogate):
    @beartype
    def __init__(
        self,
        regression_model: Regression,
        correlation_model: Correlation,
        correlation_model_parameters: list,
        bounds=None,
        optimize: bool = True,
        optimize_constraints: Constraints = None,
        optimizations_number: int = 1,
        normalize: bool = True,
        optimizer='L-BFGS-B',
        random_state: RandomStateType = None,
        **kwargs_optimizer
    ):
        """
        Κriging generates an Gaussian process regression-based surrogate model to predict the model output at new sample
        points.

        :param regression_model: `regression_model` specifies and evaluates the basis functions and their coefficients,
         which defines the trend of the model. Built-in options: Constant, Linear, Quadratic
        :param correlation_model: `corr_model` specifies and evaluates the correlation function.
         Built-in options: Exponential, Gaussian, Linear, Spherical, Cubic, Spline
        :param correlation_model_parameters: List or array of initial values for the correlation model
         hyperparameters/scale parameters.
        :param bounds: Bounds on the hyperparameters used to solve optimization problem to estimate maximum likelihood
         estimator. This should be a closed bound.
         Default: [0.001, 10**7] for each hyperparameter.
        :param optimize: Indicator to solve MLE problem or not. If 'True' corr_model_params will be used as initial
         solution for optimization problem. Otherwise, corr_model_params will be directly use as the hyperparamters.
         Default: True.
        :param optimizations_number: Number of times MLE optimization problem is to be solved with a random starting
         point. Default: 1.
        :param normalize: Boolean flag used in case data normalization is required.
        :param optimizer: String of the :class:`scipy.stats` optimizer used during the Kriging surrogate.
        Default: 'L-BFGS-B'.
        :param random_state: Random seed used to initialize the pseudo-random number generator. If an integer is
         provided, this sets the seed for an object of :class:`numpy.random.RandomState`. Otherwise, the
         object itself can be passed directly.
        """
        self.regression_model = regression_model
        self.correlation_model = correlation_model
        self.correlation_model_parameters = np.array(correlation_model_parameters)
        self.bounds = bounds
        self.optimizer = optimizer
        self.optimizations_number = optimizations_number
        self.optimize = optimize
        self.optimize_constraints = optimize_constraints
        self.normalize = normalize
        self.logger = logging.getLogger(__name__)
        self.random_state = random_state
        self.kwargs_optimizer = kwargs_optimizer

        # Variables are used outside the __init__
        self.samples = None
        self.values = None
        self.sample_mean, self.sample_std = None, None
        self.value_mean, self.value_std = None, None
        self.rmodel, self.cmodel = None, None
        self.beta = None
        """Regression coefficients."""
        self.gamma = None
        self.err_var = None
        """Variance of the Gaussian random process."""
        self.F_dash = None
        self.C_inv = None
        """Inverse Cholesky decomposition of the correlation matrix."""
        self.G = None
        self.F, self.R = None, None

        # Initialize and run preliminary error checks.
        if self.regression_model is None:
            raise NotImplementedError("UQpy: Regression model is not defined.")

        if self.correlation_model is None:
            raise NotImplementedError("Uqpy: Correlation model is not defined.")

        if self.correlation_model_parameters is None:
            raise NotImplementedError("UQpy: corr_model_params is not defined.")

        if self.bounds is None:
            self.bounds = [[0.001, 10 ** 7]] * self.correlation_model_parameters.shape[
                0
            ]

        # if self.optimizer is None:
        #     from scipy.optimize import fmin_l_bfgs_b
        #
        #     self.optimizer = fmin_l_bfgs_b
        #
        # elif not callable(self.optimizer):
        #     raise TypeError(
        #         "UQpy: Input optimizer should be None (set to scipy.optimize.minimize) or a callable."
        #     )

        self.kwargs_optimizer["method"] = self.optimizer

        self.kwargs_optimizer["jac"] = False
        if self.optimizer in ['CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP', 'dogleg', 'trust-ncg',
                              'trust-krylov', 'trust-exact', 'trust-constr']:
            self.kwargs_optimizer["jac"] = True

        if self.optimizer in ['Nelder-Mead', 'L-BFGS-B', 'TNC', 'SLSQP', 'Powell', 'trust-constr']:
            self.kwargs_optimizer["bounds"] = self.bounds

        if not isinstance(self.regression_model, Regression):
            raise NotImplementedError("UQpy: Doesn't recognize the Regression model.")

        if not isinstance(self.correlation_model, Correlation):
            raise NotImplementedError("UQpy: Doesn't recognize the Correlation model.")

        if isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError(
                "UQpy: random_state must be None, an int or an np.random.RandomState object."
            )

    def fit(
        self,
        samples,
        values,
        optimizations_number=None,
        correlation_model_parameters=None,
    ):
        """
        Fit the surrogate model using the training samples and the corresponding model values.

        The user can run this method multiple time after initiating the :class:`.Kriging` class object.

        This method updates the samples and parameters of the :class:`.Kriging` object. This method uses
        `corr_model_params` from previous run as the starting point for MLE problem unless user provides a new starting
        point.

        :param samples: `ndarray` containing the training points.
        :param values: `ndarray` containing the model evaluations at the training points.
        :param optimizations_number: number of optimization iterations
        :param correlation_model_parameters: List or array of initial values for the correlation model
         hyperparameters/scale parameters.

        The :meth:`fit` method has no returns, although it creates the `beta`, `err_var` and `C_inv` attributes of the
        :class:`.Kriging` class.
        """
        self.logger.info("UQpy: Running kriging.fit")

        if optimizations_number is not None:
            self.optimizations_number = optimizations_number
        if correlation_model_parameters is not None:
            self.correlation_model_parameters = np.array(correlation_model_parameters)
        self.samples = np.array(samples)

        # Number of samples and dimensions of samples and values
        nsamples, input_dim = self.samples.shape
        output_dim = int(np.size(values) / nsamples)

        self.values = np.array(values).reshape(nsamples, output_dim)

        # Normalizing the data
        if self.normalize:
            self.sample_mean, self.sample_std = np.mean(self.samples, 0), np.std(self.samples, 0)
            self.value_mean, self.value_std = np.mean(self.values, 0), np.std(self.values, 0)
            s_ = (self.samples - self.sample_mean) / self.sample_std
            y_ = (self.values - self.value_mean) / self.value_std
        else:
            s_ = self.samples
            y_ = self.values

        self.F, jf_ = self.regression_model.r(s_)

        # Maximum Likelihood Estimation : Solving optimization problem to calculate hyperparameters
        if self.optimize:
            starting_point = self.correlation_model_parameters
            if self.optimize_constraints is not None:
                if self.optimizer not in ['COBYLA', 'SLSQP', 'trust-constr']:
                    import warnings
                    warnings.warn("UQpy: Scipy optimize method doesn't support constraints. Thus ignoring constraints.")
                else:
                    cons = self.optimize_constraints.constraints(self.samples, self.values, self.predict)
                    self.kwargs_optimizer['constraints'] = cons

            minimizer, fun_value = np.zeros([self.optimizations_number, input_dim]),\
                                   np.zeros([self.optimizations_number, 1])

            print(self.kwargs_optimizer)
            for i__ in range(self.optimizations_number):
                p_ = minimize(
                    Kriging.log_likelihood,
                    starting_point,
                    args=(self.correlation_model, s_, self.F, y_, self.kwargs_optimizer['jac']),
                    **self.kwargs_optimizer)
                print(self.kwargs_optimizer)
                minimizer[i__, :] = p_.x
                fun_value[i__, 0] = p_.fun
                # Generating new starting points using log-uniform distribution
                if i__ != self.optimizations_number - 1:
                    starting_point = stats.reciprocal.rvs([j[0] for j in self.bounds], [j[1] for j in self.bounds], 1,
                                                          random_state=self.random_state)

            if min(fun_value) == np.inf:
                raise NotImplementedError("Maximum likelihood estimator failed: Choose different starting point or "
                                          "increase nopt")
            t = np.argmin(fun_value)
            self.correlation_model_parameters = minimizer[t, :]

        # Updated Correlation matrix corresponding to MLE estimates of hyperparameters
        self.R = self.correlation_model.c(x=s_, s=s_, params=self.correlation_model_parameters)

        self.beta, self.gamma, tmp = self._compute_additional_parameters(self.R)
        self.C_inv, self.F_dash, self.G, self.err_var = tmp[1], tmp[3], tmp[2], tmp[5]

        self.logger.info("UQpy: kriging fit complete.")

    def _compute_additional_parameters(self, correlation_matrix):
        if self.normalize:
            y_ = (self.values - self.value_mean) / self.value_std
        else:
            y_ = self.values
        # Compute the regression coefficient (solving this linear equation: F * beta = Y)
        # Eq: 3.8, DACE
        c = cholesky(correlation_matrix + (10 + self.samples.shape[0]) * 2 ** (-52) * np.eye(self.samples.shape[0]),
                     lower=True, check_finite=False)
        c_inv = np.linalg.inv(c)
        f_dash = np.linalg.solve(c, self.F)
        y_dash = np.linalg.solve(c, y_)
        q_, g_ = np.linalg.qr(f_dash)  # Eq: 3.11, DACE
        # Check if F is a full rank matrix
        if np.linalg.matrix_rank(g_) != min(np.size(self.F, 0), np.size(self.F, 1)):
            raise NotImplementedError("Chosen regression functions are not sufficiently linearly independent")
        # Design parameters (beta: regression coefficient)
        beta = np.linalg.solve(g_, np.matmul(np.transpose(q_), y_dash))

        # Design parameter (R * gamma = Y - F * beta = residual)
        gamma = np.linalg.solve(c.T, (y_dash - np.matmul(f_dash, beta)))

        # Computing the process variance (Eq: 3.13, DACE)
        err_var = np.zeros(self.values.shape[1])
        for i in range(self.values.shape[1]):
            err_var[i] = (1 / self.samples.shape[0]) * (np.linalg.norm(y_dash[:, i] -
                                                                       np.matmul(f_dash, beta[:, i])) ** 2)

        return beta, gamma, (c, c_inv, g_, f_dash, y_dash, err_var)

    def predict(self, points, return_std: bool = False, correlation_model_parameters: list = None):
        """
        Predict the model response at new points.

        This method evaluates the regression and correlation model at new sample points. Then, it predicts the function
        value and standard deviation.

        :param points: Points at which to predict the model response.
        :param  return_std: Indicator to estimate standard deviation.
        :param correlation_model_parameters: Hyperparameters for correlation model.
        :return: Predicted values at the new points, Standard deviation of predicted values at the new points
        """
        x_ = np.atleast_2d(points)
        if self.normalize:
            x_ = (x_ - self.sample_mean) / self.sample_std
            s_ = (self.samples - self.sample_mean) / self.sample_std
        else:
            s_ = self.samples
        fx, jf = self.regression_model.r(x_)
        if correlation_model_parameters is None:
            correlation_model_parameters = self.correlation_model_parameters
        rx = self.correlation_model.c(
            x=x_, s=s_, params=correlation_model_parameters
        )
        if correlation_model_parameters is None:
            beta, gamma = self.beta, self.gamma
            c_inv, f_dash, g_, err_var = self.C_inv, self.F_dash, self.G, self.err_var
        else:
            beta, gamma, tmp = self._compute_additional_parameters(
                self.correlation_model.c(x=s_, s=s_, params=correlation_model_parameters))
            c_inv, f_dash, g_, err_var = tmp[1], tmp[3], tmp[2], tmp[5]
        y = np.einsum("ij,jk->ik", fx, beta) + np.einsum(
            "ij,jk->ik", rx, gamma
        )
        if self.normalize:
            y = self.value_mean + y * self.value_std
        if x_.shape[1] == 1:
            y = y.flatten()
        if return_std:
            r_dash = np.matmul(c_inv, rx.T)
            u = np.matmul(f_dash.T, r_dash) - fx.T
            norm1 = np.linalg.norm(r_dash, 2, 0)
            norm2 = np.linalg.norm(np.linalg.solve(g_, u), 2, 0)
            mse = np.sqrt(err_var * np.atleast_2d(1 + norm2 - norm1).T)
            if self.normalize:
                mse = self.value_std * mse
            if x_.shape[1] == 1:
                mse = mse.flatten()
            return y, mse
        else:
            return y

    def jacobian(self, points):
        """
        Predict the gradient of the model at new points.

        This method evaluates the regression and correlation model at new sample point. Then, it predicts the gradient
        using the regression coefficients and the training second_order_tensor.

        :param points: Points at which to evaluate the gradient.
        :return: Gradient of the surrogate model evaluated at the new points.
        """
        x_ = np.atleast_2d(points)
        if self.normalize:
            x_ = (x_ - self.sample_mean) / self.sample_std
            s_ = (self.samples - self.sample_mean) / self.sample_std
        else:
            s_ = self.samples

        fx, jf = self.regression_model.r(x_)
        rx, drdx = self.correlation_model.c(
            x=x_, s=s_, params=self.correlation_model_parameters, dx=True
        )
        y_grad = np.einsum("ikj,jm->ik", jf, self.beta) + np.einsum(
            "ijk,jm->ki", drdx.T, self.gamma
        )
        if self.normalize:
            y_grad = y_grad * self.value_std / self.sample_std
        if x_.shape[1] == 1:
            y_grad = y_grad.flatten()
        return y_grad

    @staticmethod
    def log_likelihood(p0, cm, s, f, y, return_grad):
        # Return the log-likelihood function and it's gradient. Gradient is calculate using Central Difference
        m = s.shape[0]
        n = s.shape[1]
        r__, dr_ = cm.c(x=s, s=s, params=p0, dt=True)
        try:
            cc = cholesky(r__ + 2 ** (-52) * np.eye(m), lower=True)
        except np.linalg.LinAlgError:
            return np.inf, np.zeros(n)

        # Product of diagonal terms is negligible sometimes, even when cc exists.
        if np.prod(np.diagonal(cc)) == 0:
            return np.inf, np.zeros(n)

        cc_inv = np.linalg.inv(cc)
        r_inv = np.matmul(cc_inv.T, cc_inv)
        f__ = cc_inv.dot(f)
        y__ = cc_inv.dot(y)

        q__, g__ = np.linalg.qr(f__)  # Eq: 3.11, DACE

        # Check if F is a full rank matrix
        if np.linalg.matrix_rank(g__) != min(np.size(f__, 0), np.size(f__, 1)):
            raise NotImplementedError(
                "Chosen regression functions are not sufficiently linearly independent"
            )

        # Design parameters
        beta_ = np.linalg.solve(g__, np.matmul(np.transpose(q__), y__))

        # Computing the process variance (Eq: 3.13, DACE)
        sigma_ = np.zeros(y.shape[1])

        ll = 0
        for out_dim in range(y.shape[1]):
            sigma_[out_dim] = (1 / m) * (
                    np.linalg.norm(y__[:, out_dim] - np.matmul(f__, beta_[:, out_dim]))
                    ** 2
            )
            # Objective function:= log(det(sigma**2 * R)) + constant
            ll = (
                    ll
                    + (
                            np.log(np.linalg.det(sigma_[out_dim] * r__))
                            + m * (np.log(2 * np.pi) + 1)
                    )
                    / 2
            )

        # Gradient of loglikelihood
        # Reference: C. E. Rasmussen & C. K. I. Williams, Gaussian Processes for Machine Learning, the MIT Press,
        # 2006, ISBN 026218253X. (Page 114, Eq.(5.9))
        residual = y - np.matmul(f, beta_)
        gamma = np.matmul(r_inv, residual)
        grad_mle = np.zeros(n)
        for in_dim in range(n):
            r_inv_derivative = np.matmul(r_inv, np.matmul(dr_[:, :, in_dim], r_inv))
            tmp = np.matmul(residual.T, np.matmul(r_inv_derivative, residual))
            for out_dim in range(y.shape[1]):
                alpha = gamma / sigma_[out_dim]
                tmp1 = np.matmul(alpha, alpha.T) - r_inv / sigma_[out_dim]
                cov_der = sigma_[out_dim] * dr_[:, :, in_dim] + tmp * r__ / m
                grad_mle[in_dim] = grad_mle[in_dim] - 0.5 * np.trace(
                    np.matmul(tmp1, cov_der)
                )

        if return_grad:
            return ll, grad_mle
        else:
            return ll
