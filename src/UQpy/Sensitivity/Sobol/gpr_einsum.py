import numpy as np
from scipy.integrate import quad, dblquad
from UQpy.Sensitivity.Sobol.sobol_parallel import SobolParallel



class SobolGPR_einsum(SobolParallel):
    def __init__(self, surr_object=None, dist_object=None, samples=None, mcs_object=None, single_int=None,
                 double_int=None, n_randv=200, n_sim=1000, lower_bound=0.01, upper_bound=0.99,  random_state=None,
                 transform_x=None, transform_y=None, num_cores=1, **kwargs):

        self.transform_x, self.transform_y = transform_x, transform_y
        self.sample_std = None

        super().__init__(surr_object=surr_object, dist_object=dist_object, samples=samples, mcs_object=mcs_object,
                         single_int=single_int, double_int=double_int, n_randv=n_randv, n_sim=n_sim,
                         lower_bound=lower_bound,  upper_bound=upper_bound, random_state=random_state,
                         _transform_x=self._transform_x, _inverse_transform_x=self._inverse_transform_x,
                         _transform_y=self._transform_y, _inverse_transform_y=self._inverse_transform_y,
                         compute_mean_vector=self.compute_mean_vector, compute_cov_matrix=self.compute_cov_matrix,
                         num_cores=num_cores)

        if self.single_int is None:
            self.single_int = self._numerical_approx_single

        if self.double_int is None:
            self.double_int = self._numerical_approx_double

    def corr_model(self, x, s, dimension_index):
        tmpx, tmps = np.zeros([x.shape[0], self.dimension]), np.zeros([s.shape[0], self.dimension])
        if isinstance(dimension_index, list):
            tmpx[:, dimension_index], tmps[:, dimension_index] = x, s
        else:
            tmpx[:, dimension_index], tmps[:, dimension_index] = x.reshape(-1, ), s.reshape(-1, )
        return self.surr_object.kernel_.k2(tmpx, tmps)

    def compute_mean_vector(self, scaled_sam, i_):
        # 1. Compute first term (#Not required)

        # 2. Compute second term
        k1_val = self.surr_object.kernel_.k1.constant_value
        # rx = np.zeros([self.samples.shape[0], scaled_sam.shape[0]])
        tmp = np.delete(self.single_int_corr_f, i_, 1)
        # for k_ in range(scaled_sam.shape[0]):
        #     if isinstance(i_, list):
        #         tmp1 = self.corr_model(scaled_sam[k_].reshape(1, -1), self.samples_t[:, i_], i_)[0, :]
        #     else:
        #         tmp1 = self.corr_model(scaled_sam[k_].reshape(-1, 1), self.samples_t[:, i_].reshape(-1, 1), i_)[0, :]
        #     rx[:, k_] = np.prod(tmp, axis=1) * tmp1

        rx_mat = np.prod(tmp, axis=1) * self.corr_model(scaled_sam, self.samples_t[:, i_], i_)
        mean_term2 = k1_val * np.matmul(rx_mat, self.surr_object.alpha_)
        return mean_term2

    def compute_cov_matrix(self, scaled_sam, i_):
        tmp_db = np.prod(np.delete(self.double_int_corr_f, i_))
        k1_val = self.surr_object.kernel_.k1.constant_value

        # Compute R^{-1}
        cc_inv = np.linalg.inv(self.surr_object.L_)
        r_inv = cc_inv.T.dot(cc_inv)
        scaled_sam_ = np.zeros([scaled_sam.shape[0], self.dimension])
        scaled_sam_[:, i_] = scaled_sam.reshape(scaled_sam_[:, i_].shape)
        # if isinstance(i_, int):
        #     n_col = 1
        # else:
        #     n_col = len(i_)
        # n_row = self.samples_t.shape[0]
        # cov = np.zeros([self.n_randv, self.n_randv])

        # tmp_p = []
        tmp1 = np.delete(self.single_int_corr_f, i_, 1)
        tmp_p_mat = np.prod(tmp1, axis=1) * self.corr_model(scaled_sam, self.samples_t[:, i_], i_)
        # for p in range(self.n_randv):
        #     tmp_p.append(np.atleast_2d(np.prod(tmp1, axis=1) * self.corr_model(scaled_sam[p].reshape(1, n_col),
        #                                                                        self.samples_t[:, i_].reshape(n_row,
        #                                                                                                      n_col),
        #                                                                        i_)))

        u_mat = self.corr_model(scaled_sam, scaled_sam, i_)
        # for p in range(self.n_randv):
        #     for q in range(p, self.n_randv):
        #         u = np.prod(tmp_db) * u_mat[p, q]
        #         tq = tmp_p[q].T
        #         term2 = np.matmul(tmp_p[p], np.matmul(r_inv, tq))
        #         cov[p, q] = k1_val * (u - k1_val * term2)
        #         cov[q, p] = cov[p, q]

        cov_mat = k1_val * (np.prod(tmp_db) * u_mat - k1_val * np.matmul(tmp_p_mat, np.matmul(r_inv, tmp_p_mat.T)))
        return cov_mat

    def _transform_x(self, data, ind=None):
        """
        This function does the linear transformation of data, such that it is consistent with the domain of training set
        used to generate the kriging surrogate model.

        :param data: Data in the actual domain.
        :param ind: Index of random input needed to be transform.
        :return: The Linear transformation of data.
        """
        if self.transform_x is None:
            return data
        else:
            if ind is None:
                return self.transform_x.transform(data)
            else:
                if np.size(data.shape) == 2 and data.shape[1] == self.dimension:
                    return self.transform_x.transform(data)[:, ind]
                else:
                    tmp = np.zeros([data.shape[0], self.dimension])
                    if data.shape[1] == 1:
                        tmp[:, ind] = data.reshape(-1, )
                    else:
                        tmp[:, ind] = data
                    return self.transform_x.transform(tmp)[:, ind]

    def _transform_y(self, data):
        """
        This function does the linear transformation of data, such that it is consistent with the domain of training set
        used to generate the kriging surrogate model.

        :param data: Data in the actual domain.
        :return: The Linear transformation of data.
        """
        if self.transform_y is None:
            return data
        else:
            tmp_data = data.reshape(-1, 1)
            return self.transform_y.transform(tmp_data)

    def _inverse_transform_x(self, data, ind=None):
        """
        This function does the inverse linear transformation on the data, and returns the raw inputs/output.

        :param data: Transformed data.
        :param ind: Index of random input needed to be inverse transformed.
        :return: The Raw data.
        """
        if self.transform_x is None:
            return data
        else:
            if np.size(data.shape) == 2 and data.shape[1] == self.dimension:
                return self.transform_x.inverse_transform(data)[:, ind]
            else:
                tmp = np.zeros([data.shape[0], self.dimension])
                if data.shape[1] == 1:
                    tmp[:, ind] = data.reshape(-1, )
                else:
                    tmp[:, ind] = data
                return self.transform_x.inverse_transform(tmp)[:, ind]

    def _inverse_transform_y(self, data):
        """
        This function does the inverse linear transformation on the data, and returns the raw inputs/output.

        :param data: Transformed data.
        :return: The Raw data.
        """
        if self.transform_y is None:
            return data
        else:
            tmp_data = data.reshape(-1, 1)
            return self.transform_y.inverse_transform(tmp_data)

    def _numerical_approx_single(self, s_t, d_, corr_model, sam_std, k__, l_, u_, kg_, sam_mean):
        return quad(self.integrand1, l_, u_, args=(s_t, d_, corr_model, k__, sam_std,
                                                   self._inverse_transform_x))[0]

    def _numerical_approx_double(self, d_, corr_model, sam_std, k__, l_, u_, kg_, sam_mean):
        return dblquad(self.integrand2, l_, u_, lambda x: l_, lambda x: u_,
                       args=(d_, corr_model, k__, sam_std, self._inverse_transform_x))[0]

    @staticmethod
    def integrand1(x_t, s_t, d_, corr_model, i_, sam_std, inv_t):
        """
        This function returns the product of the correlation model and density value. This function is integrated over
        the domain of input dimension X.

        :param x_t: Transformed sample x.
        :param s_t: Transformed sample s.
        :param d_: Distribution object.
        :param corr_model: Correlation function
        :param sam_std: Kriging Object
        :param i_: Index of input variable.
        :param inv_t: Transform input variable such that input data has mean 0 and standard deviation 1.
        :return:
        """
        x_ = inv_t(np.atleast_2d(x_t), ind=i_)
        corr = corr_model(np.atleast_2d(x_t), np.atleast_2d(s_t), i_)[0, 0]
        return corr * d_.pdf(x_) * sam_std[i_]

    @staticmethod
    def integrand2(x_t, s_t, d_, corr_model, i_, sam_std, inv_t):
        """
        This function returns the product of the correlation model and density value. This function is integrated over
        the domain of input dimension X.

        :param x_t: Transformed sample x.
        :param s_t: Transformed sample s.
        :param d_: Distribution object.
        :param corr_model: Correlation function
        :param sam_std: Kriging object
        :param i_: Index of input variable.
        :param inv_t: Transform input variable such that input data has mean 0 and standard deviation 1.
        :return:
        """
        x_ = inv_t(np.atleast_2d(x_t), ind=i_)
        s_ = inv_t(np.atleast_2d(s_t), ind=i_)
        corr = corr_model(np.atleast_2d(x_t), np.atleast_2d(s_t), i_)[0, 0]
        return corr * d_.pdf(x_) * d_.pdf(s_) * sam_std[i_] ** 2

    def run(self, samples=None):
        if samples is not None and isinstance(samples, np.ndarray):
            self.samples = samples

        self.lower, self.upper = np.zeros([1, self.dimension]), np.zeros([1, self.dimension])
        for k_ in range(self.dimension):
            self.lower[0, k_] = self.dist_object[k_].icdf(self.lower_bound)
            self.upper[0, k_] = self.dist_object[k_].icdf(self.upper_bound)
        self.lower = self._transform_x(self.lower).reshape(-1, )
        self.upper = self._transform_x(self.upper).reshape(-1, )

        self.corr_model_params = self.surr_object.kernel_.k2.length_scale

        # Store GPR variance
        mcs_samples_t = self._transform_x(self.mcs_object.samples)
        self.total_var = np.var(self._inverse_transform_y(self.surr_object.predict(mcs_samples_t)))

        # Moments about origin for Distribution Object
        self.dist_moments = np.zeros([4, len(self.dist_object)])
        for k_ in range(len(self.dist_object)):
            self.dist_moments[:, k_] = self.dist_object[k_].moments()

        self.samples_t = self._transform_x(self.samples)

        sam_mean, sam_std = [0] * self.dimension, [1] * self.dimension
        if self.transform_x is not None:
            sam_mean, sam_std = self.transform_x.mean_, self.transform_x.scale_
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

    def run_interaction(self):
        interaction_mean, interaction_std = np.zeros([self.dimension, self.dimension]), \
                                            np.zeros([self.dimension, self.dimension])
        int_mean_vector = {}
        for i in range(self.dimension):
            for j in range(i+1, self.dimension):
                transformed_discrete_samples_ij = self.transformed_discrete_samples[:, (i, j)].copy()

                mean = self.compute_mean_vector(transformed_discrete_samples_ij, [i, j])

                cov = self.compute_cov_matrix(transformed_discrete_samples_ij, [i, j])

                transformed_realizations_ij = self._generate_rv(cov, mean, self.n_sim)
                realizations = np.zeros_like(transformed_realizations_ij)
                for ij_ in range(self.n_sim):
                    realizations[:, ij_] = self._inverse_transform_y(transformed_realizations_ij[:, ij_]).reshape(-1, )

                sobol_ij_estimates = np.var(realizations, axis=0)/self.total_var

                interaction_mean[i, j] = np.mean(sobol_ij_estimates) - self.sobol_mean[i] - self.sobol_mean[j]
                interaction_std[i, j] = np.std(sobol_ij_estimates - self.sobol_estimates[:, i] -
                                               self.sobol_estimates[:, j])
                int_mean_vector['{}{}'.format(i, j)] = mean

        return int_mean_vector, interaction_mean, interaction_std
