import numpy as np
from scipy.integrate import quad, dblquad
from UQpy.Sensitivity.Sobol.sobol import Sobol
from scipy.linalg import solve


class SobolKriging(Sobol):
    def __init__(self, surr_object=None, dist_object=None, samples=None, mcs_object=None, single_int=None,
                 double_int=None, n_randv=200, n_sim=1000, lower_bound=0.01, upper_bound=0.99,  random_state=None,
                 **kwargs):

        self.sample_std = None

        super().__init__(surr_object=surr_object, dist_object=dist_object, samples=samples, mcs_object=mcs_object,
                         single_int=single_int, double_int=double_int, n_randv=n_randv, n_sim=n_sim,
                         lower_bound=lower_bound,  upper_bound=upper_bound, random_state=random_state,
                         _transform_x=self._transform_x, _inverse_transform_x=self._inverse_transform_x,
                         _transform_y=self._transform_y, _inverse_transform_y=self._inverse_transform_y,
                         compute_mean_vector=self.compute_mean_vector, compute_cov_matrix=self.compute_cov_matrix)

        if self.single_int is None:
            self.single_int = self._numerical_approx_single

        if self.double_int is None:
            self.double_int = self._numerical_approx_double

    def corr_model(self, x, s, dimension_index):
        return self.surr_object.corr_model(x, s, self.surr_object.corr_model_params[dimension_index])

    def _reg_model_moments(self, scaled_sam, i_):
        """
        Evaluates the kriging regression model on the transformed mean of input random variables.

        :param scaled_sam:
        :param i_:
        :return:
        """
        scaled_mean_moment = self._transform_x(self.dist_moments[0])
        m_reg_, jb = self.surr_object.reg_model(scaled_mean_moment)
        mean_reg_ = np.array([m_reg_[0]] * scaled_sam.shape[0])
        mean_reg_[:, i_ + 1] = scaled_sam.reshape(-1, )
        return mean_reg_

    def compute_mean_vector(self, scaled_sam, i_):
        # 1. Compute first term
        mean_reg_ = self._reg_model_moments(scaled_sam, i_)
        mean_term1 = np.matmul(mean_reg_, self.surr_object.beta)

        # 2. Compute second term
        rx = np.zeros([self.samples.shape[0], scaled_sam.shape[0]])
        tmp = np.delete(self.single_int_corr_f, i_, 1)
        for k_ in range(scaled_sam.shape[0]):
            r_i = self.surr_object.corr_model(scaled_sam[k_].reshape(-1, 1), self.samples_t[:, i_].reshape(-1, 1),
                                              self.surr_object.corr_model_params[i_])[0, :]
            rx[:, k_] = np.prod(tmp, axis=1) * r_i
        mean_term2 = np.matmul(rx.T, self.surr_object.gamma)
        mean_ = mean_term1 + mean_term2
        return mean_

    def compute_cov_matrix(self, scaled_sam, i_):
        tmp_db = np.prod(np.delete(self.double_int_corr_f, i_))

        # Compute R^{-1}
        r_inv = np.linalg.inv(self.surr_object.R)

        r_inv_f = solve(self.surr_object.R, self.surr_object.F, check_finite=False, assume_a='pos')
        w_inv = np.matmul(self.surr_object.F_dash.T, self.surr_object.F_dash)
        w = np.linalg.inv(w_inv)
        mean_reg_ = self._reg_model_moments(scaled_sam, i_)

        corr = np.zeros([self.n_randv, self.n_randv])
        for p in range(self.n_randv):
            for q in range(p, self.n_randv):
                u = np.prod(tmp_db) * self.surr_object.corr_model(scaled_sam[p], scaled_sam[q],
                                                                  self.surr_object.corr_model_params[i_])

                tmp1 = np.delete(self.single_int_corr_f, i_, 1)
                tp = np.atleast_2d(
                    np.prod(tmp1, axis=1) * self.surr_object.corr_model(scaled_sam[p],
                                                                        self.samples_t[:, i_].reshape(-1, 1),
                                                                        self.surr_object.corr_model_params[i_])).T
                tq = np.atleast_2d(
                    np.prod(tmp1, axis=1) * self.surr_object.corr_model(scaled_sam[q],
                                                                        self.samples_t[:, i_].reshape(-1, 1),
                                                                        self.surr_object.corr_model_params[i_])).T

                rp = np.atleast_2d(mean_reg_[p, :])
                rq = np.atleast_2d(mean_reg_[q, :]).T

                term2 = np.matmul(tp.T, np.matmul(r_inv, tq))
                term3p = rp - np.matmul(tp.T, r_inv_f)
                term3q = rq - np.matmul(tq.T, r_inv_f).T
                term3 = np.matmul(term3p, np.matmul(w, term3q))
                corr[p, q] = u - term2 + term3
                corr[q, p] = corr[p, q]

        cov = self.surr_object.err_var * corr
        return cov

    def _transform_x(self, data, ind=None):
        """
        This function does the linear transformation of data, such that it is consistent with the domain of training set
        used to generate the kriging surrogate model.

        :param data: Data in the actual domain.
        :param ind: Index of random input needed to be transform.
        :return: The Linear transformation of data.
        """
        if not self.surr_object.normalize:
            return data

        if ind is None:
            return (data - self.surr_object.sample_mean) / self.surr_object.sample_std
        else:
            return (data - self.surr_object.sample_mean[ind]) / self.surr_object.sample_std[ind]

    def _transform_y(self, data):
        """
        This function does the linear transformation of data, such that it is consistent with the domain of training set
        used to generate the kriging surrogate model.

        :param data: Data in the actual domain.
        :return: The Linear transformation of data.
        """
        if not self.surr_object.normalize:
            return data

        return (data - self.surr_object.value_mean) / self.surr_object.value_std

    def _inverse_transform_x(self, data, ind=None):
        """
        This function does the inverse linear transformation on the data, and returns the raw inputs/output.

        :param data: Transformed data.
        :param ind: Index of random input needed to be inverse transformed.
        :return: The Raw data.
        """
        if not self.surr_object.normalize:
            return data

        if ind is None:
            return self.surr_object.sample_mean + data * self.surr_object.sample_std
        else:
            return self.surr_object.sample_mean[ind] + data * self.surr_object.sample_std[ind]

    def _inverse_transform_y(self, data):
        """
        This function does the inverse linear transformation on the data, and returns the raw inputs/output.

        :param data: Transformed data.
        :return: The Raw data.
        """
        if not self.surr_object.normalize:
            return data

        return self.surr_object.value_mean + data * self.surr_object.value_std

    def _numerical_approx_single(self, s_t, d_, corr_model, sam_std, k__, l_, u_, kg_, sam_mean):
        return quad(self.integrand1, l_, u_, args=(s_t, d_, corr_model, k__, kg_,
                                                   self._inverse_transform_x))[0]

    def _numerical_approx_double(self, d_, corr_model, sam_std, k__, l_, u_, kg_, sam_mean):
        return dblquad(self.integrand2, l_, u_, lambda x: l_, lambda x: u_,
                       args=(d_, corr_model, k__, kg_, self._inverse_transform_x))[0]

    @staticmethod
    def integrand1(x_t, s_t, d_, corr_model, i_, kg_, inv_t):
        """
        This function returns the product of the correlation model and density value. This function is integrated over
        the domain of input dimension X.

        :param x_t: Transformed sample x.
        :param s_t: Transformed sample s.
        :param d_: Distribution object.
        :param corr_model: Correlation function
        :param kg_: Kriging Object
        :param i_: Index of input variable.
        :param inv_t: Transform input variable such that input data has mean 0 and standard deviation 1.
        :return:
        """
        x_ = inv_t(x_t, ind=i_)
        corr = corr_model(x_t, s_t, i_)[0, 0]
        return corr * d_.pdf(x_) * kg_.sample_std[i_]

    @staticmethod
    def integrand2(x_t, s_t, d_, corr_model, i_, kg_, inv_t):
        """
        This function returns the product of the correlation model and density value. This function is integrated over
        the domain of input dimension X.

        :param x_t: Transformed sample x.
        :param s_t: Transformed sample s.
        :param d_: Distribution object.
        :param corr_model: Correlation function
        :param kg_: Kriging object
        :param i_: Index of input variable.
        :param inv_t: Transform input variable such that input data has mean 0 and standard deviation 1.
        :return:
        """
        x_ = inv_t(x_t, ind=i_)
        s_ = inv_t(s_t, ind=i_)
        corr = corr_model(x_t, s_t, i_)[0, 0]
        return corr * d_.pdf(x_) * d_.pdf(s_) * kg_.sample_std[i_] ** 2

    def run(self, samples=None):
        if samples is not None and isinstance(samples, np.ndarray):
            self.samples = samples

        self.lower, self.upper = np.zeros([1, self.dimension]), np.zeros([1, self.dimension])
        for k_ in range(self.dimension):
            self.lower[0, k_] = self.dist_object[k_].icdf(self.lower_bound)
            self.upper[0, k_] = self.dist_object[k_].icdf(self.upper_bound)
        self.lower = self._transform_x(self.lower).reshape(-1, )
        self.upper = self._transform_x(self.upper).reshape(-1, )

        self.corr_model_params = self.surr_object.corr_model_params

        # Store GPR variance
        self.total_var = np.var(self.surr_object.predict(self.mcs_object.samples))

        # Moments about origin for Distribution Object
        self.dist_moments = np.zeros([4, len(self.dist_object)])
        for k_ in range(len(self.dist_object)):
            self.dist_moments[:, k_] = self.dist_object[k_].moments()

        self.samples_t = self._transform_x(self.samples)

        sam_mean, sam_std = [0] * self.dimension, [1] * self.dimension
        if self.surr_object.normalize:
            sam_mean, sam_std = self.surr_object.sample_mean, self.surr_object.sample_std
        # Single integration components of the correlation matrix
        self.single_int_corr_f = np.zeros_like(self.samples)
        # start_time = time.time()
        for k_ in range(self.dimension):
            for l_ in range(self.samples.shape[0]):
                self.single_int_corr_f[l_, k_] = self.single_int(s_t=self.samples_t[l_, k_], d_=self.dist_object[k_],
                                                                 corr_model=self.corr_model, sam_std=sam_std,
                                                                 k__=k_, l_=self.lower[k_], u_=self.upper[k_],
                                                                 sam_mean=sam_mean, kg_=self.surr_object)

        # Double integration components of the correlation matrix
        self.double_int_corr_f = np.zeros(self.dimension)
        for l_ in range(self.dimension):
            self.double_int_corr_f[l_] = self.double_int(d_=self.dist_object[l_], corr_model=self.corr_model,
                                                         sam_std=sam_std, k__=l_, l_=self.lower[l_], u_=self.upper[l_],
                                                         sam_mean=sam_mean, kg_=self.surr_object)
        self.framework()

        self.sample_std = sam_std