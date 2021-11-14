import numpy as np
from UQpy.StochasticProcess import KLE_Two_Dimension

nx, nt = 20, 10
dx, dt = 0.05, 0.1

x = np.linspace(0, (nx - 1) * dx, nx)
t = np.linspace(0, (nt - 1) * dt, nt)
xt_list = np.meshgrid(x, x, t, t, indexing='ij')  # R(t_1, t_2, x_1, x_2)

nsamples = 10000

R = np.exp(-(xt_list[0] - xt_list[1]) ** 2 - (xt_list[2] - xt_list[3]) ** 2)
# R(x_1, x_2, t_1, t_2) = exp(-(x_1 - x_2) ** 2 -(t_1 - t_2) ** 2)

KLE_object = KLE_Two_Dimension(nsamples=nsamples, correlation_function=R, time_interval=0.1, thresholds=[4, 5])
samples = KLE_object.samples


def test_samples_shape():
    assert (samples.shape == (nsamples, 1, nx, nt))


def test_user_passed_random_variables():
    computer_generated_random_variables = KLE_object.xi
    KLE_object2 = KLE_Two_Dimension(nsamples=nsamples, correlation_function=R, time_interval=0.1, thresholds=[4, 5],
                                    random_variables=computer_generated_random_variables)
    samples2 = KLE_object2.samples
    assert (samples.shape, samples2.shape)
    assert (np.allclose(samples, samples2))
