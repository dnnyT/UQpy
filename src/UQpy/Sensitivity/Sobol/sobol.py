import os
import numpy as np
import matplotlib.pyplot as plt
from UQpy.Distributions import *
from UQpy.SampleMethods import MCS, LHS
from UQpy.Surrogates import Kriging
from sklearn.gaussian_process import GaussianProcessRegressor


class Sobol:
    """
    Parent class for computing first order Sobol Indices.

    This is the parent class to compute first order of Sobol Indices. This parent class only provides the
    framework for sensitivity analysis. Estimates are identified by calling the child class for the desired surrogate
    (Kriging or Gaussian Process Regressor).

    **Inputs:**

    * **samples** (`ndarray`):
        A numpy array containing the data set used to train the surrgate model.

    * **surr_object** (`class` object):
        A object defining a Kriging surrogate model, this object must have ``fit`` and ``predict`` methods.

        May be an object of the ``UQpy`` ``Kriging`` class or an object of the ``scikit-learn``
        ``GaussianProcessRegressor``

        `surr_object` is only used to compute the gradient in gradient-enhanced refined stratified sampling. It must
        be provided if a `runmodel_object` is provided.

    * **dist_object** ((list of) ``Distribution`` object(s)):
        List of ``Distribution`` objects corresponding to each random variable.

    * **mcs_object** (`class` object):
        A class object of UQpy.SampleMethods.MCS class to compute a monte carlo estimate of the output variance using
        surrogate's prediction method.

    * **single_int** (`callable`):
        A method to compute the single integration of correlation model. If None, a numerical estimate is identified
        using `scipy.integrate.quad`.

    * **double_int** (`callable`):
        A method to compute the double integration of correlation model. If None, a numerical estimate is identified
        using `scipy.integrate.dblquad`.

    * **step_size** (`float`)
        Defines the size of the step to use for gradient estimation using central difference method.

        Used only in gradient-enhanced refined stratified sampling.

    * **n_randv** (`int`):
        Number of random points along each dimension, generated using Latin Hypercube Sampling, to compute Sobol
        Indices.

    * **n_sim** (`int`):
        Number of estimates of Sobol Indices to compute mean and standard deviation.

    * **lower_bound** (`float`):
        A float between 0 and 1, which defines the lower bound for integration of correlation model. The lower bound is
        computed by taking the inverse transform of the provide value.

        Eg: If `dist_object`=Uniform(loc=1, scale=1) and `lower_bound`=0.02
        then lower bound for integration is dist_object.icdf(0.02) = 1.02.

        This value is used if a callable is not provided for  `single_int` and 'double_int' attribute.
        Default: 0.01

    * **lower_bound** (`float`):
        A float between 0 and 1, which defines the upper bound for integration of correlation model. The upper bound is
        computed by taking the inverse transform of the provide value.

        Eg: If `dist_object`=Uniform(loc=1, scale=1) and `upper_bound`=0.98
        then upper bound for integration is dist_object.icdf(0.98) = 1.98.

        This value is used if a callable is not provided for  `single_int` and 'double_int' attribute.
        Default: 0.99

    * **transform_x** (`class` object):
        A class object to transform and inverse transform the input samples. This class object should have `transform`
        and `inverse_transform` methods.

    * **transform_y** (`class` object):
        A class object to transform and inverse transform the output samples. This class object should have `transform`
        and `inverse_transform` methods.

    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
        Random seed used to initialize the pseudo-random number generator. Default is None.

        If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
        object itself can be passed directly.

    * **verbose** (`Boolean`):
        A boolean declaring whether to write text to the terminal.

        Default value: False

    **Attributes:**

    Each of the above inputs are saved as attributes, in addition to the following created attributes.

        * **sobol_mean** (`ndarray`):
            The generated stratified samples following the prescribed distribution.

        * **sobol_std** (`ndarray`)
            The generated samples on the unit hypercube.

        **Methods:**
    """
    def __init__(self, surr_object=None, dist_object=None, samples=None, mcs_object=None, single_int=None,
                 double_int=None, n_randv=200, n_sim=1000, lower_bound=0.01, upper_bound=0.99, random_state=None,
                 **kwargs):
        self.samples, self.samples_t = samples, None
        self.surr_object = surr_object
        self.dist_object = dist_object
        self.mcs_object = mcs_object
        self.single_int = single_int
        self.double_int = double_int
        self.n_randv = n_randv
        self.n_sim = n_sim
        self.lower_bound, self.upper_bound = lower_bound, upper_bound
        self.lower, self.upper = None, None
        self.dimension = 1
        self.dist_moments = None
        self.single_int_corr_f, self.double_int_corr_f = None, None
        self.mean_vec, self.cov_mat = None, None
        self.sobol_mean, self.sobol_std = None, None
        self.discrete_samples, self.transformed_discrete_samples = None, None
        self._transform_x, self._inverse_transform_x = kwargs['_transform_x'], kwargs['_inverse_transform_x']
        self._transform_y, self._inverse_transform_y = kwargs['_transform_y'], kwargs['_inverse_transform_y']
        self.compute_mean_vector, self.compute_cov_matrix = kwargs['compute_mean_vector'], kwargs['compute_cov_matrix']
        self.kwargs = kwargs
        self.realizations, self.sobol_estimates = None, None
        self.total_var = None

        self.random_state = random_state
        if isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')

        if self.samples is not None and not isinstance(self.samples, np.ndarray):
            raise NotImplementedError("Attribute 'samples' should be a numpy array")

        if self.mcs_object is None:
            self.mcs_object = MCS(dist_object=self.dist_object, nsamples=100000, random_state=self.random_state)

        if not isinstance(self.surr_object, (Kriging, GaussianProcessRegressor)):
            raise NotImplementedError("Attribute 'surr_object' should be an UQpy.Surrogates.Kriging object or "
                                      "sklearn.gaussian_process.GaussianProcessRegressor object")

        if isinstance(self.dist_object, list):
            for i in range(len(self.dist_object)):
                if not isinstance(self.dist_object[i], DistributionContinuous1D):
                    raise TypeError('UQpy: A DistributionContinuous1D object must be provided.')
            self.dimension = len(self.dist_object)
        else:
            self.dimension = 1
            if not isinstance(self.dist_object, DistributionContinuous1D):
                raise TypeError('UQpy: A DistributionContinuous1D object must be provided.')

    def framework(self):
        self.sobol_mean, self.sobol_std = [], []
        self.cov_mat, self.mean_vec = np.ones([self.dimension, self.n_randv, self.n_randv]), \
                                      np.ones([self.n_randv, self.dimension])

        self.discrete_samples = LHS(dist_object=self.dist_object, nsamples=self.n_randv).samples
        self.transformed_discrete_samples = self._transform_x(self.discrete_samples)
        self.realizations, self.sobol_estimates = [], np.zeros([self.n_sim, self.dimension])
        for i in range(self.dimension):
            # sort_index = np.argsort(self.transformed_discrete_samples[:, i])
            # transformed_discrete_samples_i = self.transformed_discrete_samples[:, i].copy()[np.argsort(sort_index)]
            transformed_discrete_samples_i = self.transformed_discrete_samples[:, i].copy()

            # ##Compute the mean of Gaussian process (A(X^i)=E[Y|X^i])
            # Mean vector at learning/candidate input points
            mean = self.compute_mean_vector(transformed_discrete_samples_i, i)

            # ##Compute the covariance of Gaussian process (A(X^i)=E[Y|X^i])
            cov = self.compute_cov_matrix(transformed_discrete_samples_i, i)

            self.cov_mat[i, :, :] = cov.copy()
            self.mean_vec[:, i] = mean.reshape(-1, ).copy()
            # self.cov_mat[i, :, :] = cov[np.argsort(sort_index), np.argsort(sort_index)]
            # self.mean_vec[:, i] = mean.reshape(-1, )[np.argsort(sort_index)]

            transformed_realizations = self._generate_rv(cov, mean, self.n_sim)
            realizations = np.zeros_like(transformed_realizations)
            for ij_ in range(self.n_sim):
                realizations[:, ij_] = self._inverse_transform_y(transformed_realizations[:, ij_]).reshape(-1,)

            self.realizations.append(realizations)
            self.sobol_estimates[:, i] = np.var(realizations, axis=0)/self.total_var

            self.sobol_mean.append(np.mean(self.sobol_estimates[:, i]))
            self.sobol_std.append(np.std(self.sobol_estimates[:, i]))

        print('(Total variance) MCS estimate: ', self.total_var)
        print('Sobol Indices (using MCS estimate of variance): ', self.sobol_mean)

    def _generate_rv(self, cov_matrix, mean_vector, nsamples):
        e_val, e_vec = np.linalg.eigh(cov_matrix)
        # idx = e_val.argsort()[::-1]
        # eigen_values = e_val[idx]
        # eigen_vectors = e_vec[:, idx]
        eigen_values = e_val
        eigen_vectors = e_vec

        # Remove negative eigenvalues
        n_positive_ev = np.sum(eigen_values > 0)
        eigen_values = np.diag(eigen_values[-n_positive_ev:])
        eigen_vectors = eigen_vectors[:, -n_positive_ev:]

        xi = self.random_state.normal(size=(n_positive_ev, nsamples))
        realiz = np.matmul(eigen_vectors, np.matmul(np.sqrt(eigen_values), xi))
        return mean_vector + realiz

    def plot_conditional_gp(self, directory=None, actual_function=None, err_bar=None, title=None):
        """

        :param directory:
        :param actual_function: List of callables, which returns conditional GP value.
        :param title:
        :return:
        """
        for i in range(self.dimension):
            sort_index = np.argsort(self.discrete_samples[:, i])
            x_points = self.discrete_samples[:, i][sort_index]
            y_estimate = self._inverse_transform_y(self.mean_vec[:, i])[sort_index]
            plt.figure()
            if err_bar is None:
                plt.plot(x_points, y_estimate, label='Estimate')
            else:
                y_err = self.sample_std[i]**2 * np.diag(self.cov_mat[i, :, :])[sort_index]
                plt.errorbar(x_points, y_estimate, yerr=y_err, label='Estimate')
            if actual_function is not None and isinstance(actual_function, list):
                y_actual = actual_function[i](x_points)
                plt.plot(x_points, y_actual, label='Actual')
            if directory is not None:
                plt.savefig(os.path.join(directory, 'conditional_gp_{}.jpeg'.format(i+1)), dpi=300)
            if title is not None:
                plt.title(title+' (Dim={})'.format(i+1))
            else:
                plt.title(r'Conditional GP $E[Y|X_{}]$ (Dim={})'.format(i + 1, i + 1))
            plt.xlabel(r'Input $X_{}$'.format(i + 1))
            plt.ylabel(r'$E[Y|X_{}]$'.format(i + 1))
            plt.ylim(self._inverse_transform_y(np.min(self.mean_vec)), self._inverse_transform_y(np.max(self.mean_vec)))
            plt.legend()
            plt.show()
            plt.close()

    def plot_gp_realization(self, n_realization=None, directory=None, title=None):

        if n_realization is None:
            n_realization = self.n_sim
        for i in range(self.dimension):
            sort_index = np.argsort(self.discrete_samples[:, i])
            x_points = self.discrete_samples[:, i][sort_index]
            fig = plt.figure()
            for i__ in range(n_realization):
                plt.scatter(x_points, self.realizations[i][:, i__][sort_index])
            plt.title('Realizations of Conditional GP (N={})'.format(self.samples.shape[0]))
            plt.xlabel('Input ($x^{}$)'.format(i + 1))
            plt.ylabel('Realization of Conditional GP $A(X^{})$'.format(i + 1))
            if directory is not None:
                plt.savefig(os.path.join(directory, 'GP_realizations_{}.jpeg'.format(i+1)), dpi=300)
            if title is not None:
                plt.title(title+' (Dim={})'.format(i+1))
            else:
                plt.title(r'Realizations of Conditional GP $E[Y|X_{}]$ (Dim={})'.format(i + 1, i + 1))
            plt.show()
            plt.clf()
            plt.close(fig)

    def hist_si_estimates(self, directory=None, title=None):
        for i in range(self.dimension):
            plt.hist(self.sobol_estimates, density=True)
            plt.xlabel('Sobol estimate: Main effect of input variable {}'.format(i + 1))
            if directory is not None:
                plt.savefig(os.path.join(directory, 'sobol_estimates_{}.jpeg'.format(i+1)), dpi=300)
            if title is not None:
                plt.title(title+' (Dim={})'.format(i+1))
            else:
                plt.title(r'Conditional GP $E[Y|X_{}]$ (Dim={})'.format(i + 1, i + 1))
            plt.show()
            plt.close()
