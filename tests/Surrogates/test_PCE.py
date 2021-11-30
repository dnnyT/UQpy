from UQpy.distributions import *
from UQpy.surrogates import *
import numpy as np
from UQpy.surrogates.polynomial_chaos.ChaosPolynomials import hermite_eval, legendre_eval
from UQpy.surrogates.polynomial_chaos.MultiIndexSets import setsize, td_set_recursive
from UQpy.surrogates.polynomial_chaos.SobolEstimation import pce_generalized_sobol_first, pce_generalized_sobol_total

np.random.seed(1)
max_degree, n_samples = 2, 10
dist = Uniform(loc=0, scale=10)
pce = PolyChaosExp(dist)

def func(x):
    return x * np.sin(x) / 10

x = dist.rvs(n_samples)
x_test = dist.rvs(n_samples)
y = func(x)

def poly_td_func(pce, max_degree):
    construct_td_basis(pce, max_degree)
    p = pce.poly_basis
    return p

def poly_tp_func(pce, max_degree):
    construct_tp_basis(pce, max_degree)
    p = pce.poly_basis
    return p

def pce_coeff_lstsq(pce, x, y):
    fit_lstsq(pce, x, y)
    return pce.coefficients

def pce_coeff_lasso(pce, x, y):
    fit_lasso(pce, x, y)
    return pce.coefficients

def pce_coeff_ridge(pce, x, y):
    fit_ridge(pce, x, y)
    return pce.coefficients

def pce_predict(pce,x):
    return pce.predict(x)

# Unit tests
def test_1():
    """
    Test td basis
    """
    assert round(poly_td_func(pce, max_degree)[1].evaluate(x)[0], 4) == -0.2874

def test_2():
    """
    Test tp basis
    """
    assert round(poly_tp_func(pce, max_degree)[1].evaluate(x)[0], 4) == -0.2874

def test_3():
    """
    Test PCE coefficients w/ lasso
    """
    assert round(pce_coeff_lasso(pce, x, y)[0][0], 4) == 0.0004

def test_4():
    """
    Test PCE coefficients w/ ridge
    """
    assert round(pce_coeff_ridge(pce, x, y)[0][0], 4) == 0.0276

def test_5():
    """
    Test PCE coefficients w/ lstsq
    """
    assert round(pce_coeff_lstsq(pce, x, y)[0][0], 4) == 0.2175

def test_6():
    """
    Test PCE prediction
    """
    y_test = pce_predict(pce, x_test)
    assert round(y_test[0][0], 4) == -0.0794

def test_7():
    """
    Test Sobol indices
    """
    assert round(pce_sobol_first(pce)[0][0], 3) == 1.0

def test_8():
    """
    Test Sobol indices
    """
    assert round(pce_sobol_total(pce)[0][0], 3) == 1.0

def test_9():
    """
    Test Sobol indices
    """
    assert pce_generalized_sobol_first(pce) == None

def test_10():
    """
    Test Sobol indices
    """
    assert pce_generalized_sobol_total(pce) == None

def test_11():
    """
    PCE mean
    """
    assert round(pce_mean(pce)[0], 3) == 0.299

def test_12():
    """
    PCE variance
    """
    assert round(pce_variance(pce)[0], 3) == 0.185

def test_13():
    """
    Evaluation of Legendre polynomials
    """
    assert round(hermite_eval(x_test, 2, dist)[0], 4) == -0.5829

def test_14():
    """
    Evaluation of Hermite polynomials
    """
    assert round(legendre_eval(x_test, 2, dist)[0], 4) == -1.0304

def test_15():
    """
    set size
    """
    assert round(setsize(3, 2), 4) == 6

def test_16():
    """
    td_set_recursive
    """
    assert round(td_set_recursive(2, 3, 4)[0][1], 4) == 3.0


def function(x):
    # without square root
    u1 = x[:, 4] * np.cos(x[:, 0])
    u2 = x[:, 4] * np.cos(x[:, 0]) + x[:, 5] * np.cos(np.sum(x[:, :2], axis=1))
    u3 = x[:, 4] * np.cos(x[:, 0]) + x[:, 5] * np.cos(np.sum(x[:, :2], axis=1)) + x[:, 6] * np.cos(
        np.sum(x[:, :3], axis=1))
    u4 = x[:, 4] * np.cos(x[:, 0]) + x[:, 5] * np.cos(np.sum(x[:, :2], axis=1)) + x[:, 6] * np.cos(
        np.sum(x[:, :3], axis=1)) + x[:, 7] * np.cos(np.sum(x[:, :4], axis=1))

    v1 = x[:, 4] * np.sin(x[:, 0])
    v2 = x[:, 4] * np.sin(x[:, 0]) + x[:, 5] * np.sin(np.sum(x[:, :2], axis=1))
    v3 = x[:, 4] * np.sin(x[:, 0]) + x[:, 5] * np.sin(np.sum(x[:, :2], axis=1)) + x[:, 6] * np.sin(
        np.sum(x[:, :3], axis=1))
    v4 = x[:, 4] * np.sin(x[:, 0]) + x[:, 5] * np.sin(np.sum(x[:, :2], axis=1)) + x[:, 6] * np.sin(
        np.sum(x[:, :3], axis=1)) + x[:, 7] * np.sin(np.sum(x[:, :4], axis=1))

    return (u1 + u2 + u3 + u4) ** 2 + (v1 + v2 + v3 + v4) ** 2

dist_1 = Uniform(loc=0, scale=2*np.pi)
dist_2 = Uniform(loc=0, scale=1)

marg = [dist_1]*4
marg_1 = [dist_2]*4
marg.extend(marg_1)

joint = JointIndependent(marginals=marg)

n_samples_2 = 10
x_2 = joint.rvs(n_samples_2)
y_2 = function(x_2)
pce_2 = PolyChaosExp(joint)
construct_td_basis(pce_2, 2)
fit_lstsq(pce_2, x_2, y_2)

def test_17():
    """
    Test Sobol indices for vector-valued quantity of interest on the random inputs
    """
    assert round(pce_generalized_sobol_first(pce_2)[0], 4) == 0.1073

def test_18():
    """
    Test Sobol indices for vector-valued quantity of interest on the random inputs
    """
    assert round(pce_generalized_sobol_total(pce_2)[0], 4) == 0.1921
