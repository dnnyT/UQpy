import pytest

from UQpy.run_model.model_execution.PythonModel import PythonModel
from UQpy.utilities.MinimizeOptimizer import MinimizeOptimizer

from UQpy.surrogates.gaussian_process.GaussianProcessRegression import GaussianProcessRegression
from UQpy.sampling import MonteCarloSampling, AdaptiveKriging
from UQpy.run_model.RunModel import RunModel
from UQpy.distributions.collection import Normal
from UQpy.sampling.adaptive_kriging_functions import *
import shutil


def test_akmcs_weighted_u():
    from UQpy.surrogates.gaussian_process.regression_models.LinearRegression import LinearRegression
    from UQpy.surrogates.gaussian_process.kernels.RBF import RBF

    marginals = [Normal(loc=0., scale=4.), Normal(loc=0., scale=4.)]
    x = MonteCarloSampling(distributions=marginals, nsamples=20, random_state=0)
    model = PythonModel(model_script='series.py', model_object_name="series")
    rmodel = RunModel(model=model)
    regression_model = LinearRegression()
    correlation_model = RBF()
    K = GaussianProcessRegression(kernel=correlation_model, hyperparameters=[1, 1, 0.1], random_state=1,
                                  optimizer=MinimizeOptimizer('l-bfgs-b'), regression_model=regression_model,
                                  optimizations_number=10, bounds=[[1e-2, 20], [1e-2, 20], [1e-5, 1e-2]],)
    # OPTIONS: 'U', 'EFF', 'Weighted-U'
    learning_function = WeightedUFunction(weighted_u_stop=2)
    a = AdaptiveKriging(distributions=marginals, runmodel_object=rmodel, surrogate=K,
                        learning_nsamples=10 ** 3, n_add=1, learning_function=learning_function,
                        random_state=2)
    a.run(nsamples=25, samples=x.samples)

    assert a.samples[23, 0] == -2.6302040474312056
    assert a.samples[20, 1] == 0.20293978126855253


def test_akmcs_u():
    from UQpy.surrogates.gaussian_process.regression_models.LinearRegression import LinearRegression
    from UQpy.surrogates.gaussian_process.kernels.RBF import RBF

    marginals = [Normal(loc=0., scale=4.), Normal(loc=0., scale=4.)]
    x = MonteCarloSampling(distributions=marginals, nsamples=20, random_state=1)
    model = PythonModel(model_script='series.py', model_object_name="series")
    rmodel = RunModel(model=model)
    regression_model = LinearRegression()
    correlation_model = RBF()
    K = GaussianProcessRegression(kernel=correlation_model, hyperparameters=[1, 1, 0.1], random_state=1,
                                  optimizer=MinimizeOptimizer('l-bfgs-b'), regression_model=regression_model,
                                  optimizations_number=10, bounds=[[1e-2, 20], [1e-2, 20], [1e-5, 1e-2]],)
    # OPTIONS: 'U', 'EFF', 'Weighted-U'
    learning_function = UFunction(u_stop=2)
    a = AdaptiveKriging(distributions=marginals, runmodel_object=rmodel, surrogate=K,
                        learning_nsamples=10 ** 3, n_add=1, learning_function=learning_function,
                        random_state=2)
    a.run(nsamples=25, samples=x.samples)

    assert a.samples[23, 0] == 2.3364698907681873
    assert a.samples[20, 1] == -0.35419762304600505


def test_akmcs_expected_feasibility():
    from UQpy.surrogates.gaussian_process.regression_models.LinearRegression import LinearRegression
    from UQpy.surrogates.gaussian_process.kernels.RBF import RBF

    marginals = [Normal(loc=0., scale=4.), Normal(loc=0., scale=4.)]
    x = MonteCarloSampling(distributions=marginals, nsamples=20, random_state=1)
    model = PythonModel(model_script='series.py', model_object_name="series")
    rmodel = RunModel(model=model)
    regression_model = LinearRegression()
    correlation_model = RBF()
    K = GaussianProcessRegression(kernel=correlation_model, hyperparameters=[1, 1, 0.1], random_state=1,
                                  bounds=[[1e-2, 20], [1e-2, 20], [1e-5, 1e-2]],
                                  optimizer=MinimizeOptimizer('l-bfgs-b'), regression_model=regression_model,
                                  optimizations_number=10)
    # OPTIONS: 'U', 'EFF', 'Weighted-U'
    learning_function = ExpectedFeasibility(eff_a=0, eff_epsilon=2, eff_stop=0.001)
    a = AdaptiveKriging(distributions=marginals, runmodel_object=rmodel, surrogate=K,
                        learning_nsamples=10 ** 3, n_add=1, learning_function=learning_function,
                        random_state=2)
    a.run(nsamples=25, samples=x.samples)

    assert a.samples[23, 0] == -0.2710594625659133
    assert a.samples[20, 1] == 0.8104109752887586


def test_akmcs_expected_improvement():
    from UQpy.surrogates.gaussian_process.regression_models.LinearRegression import LinearRegression
    from UQpy.surrogates.gaussian_process.kernels.RBF import RBF

    marginals = [Normal(loc=0., scale=4.), Normal(loc=0., scale=4.)]
    x = MonteCarloSampling(distributions=marginals, nsamples=20, random_state=1)
    model = PythonModel(model_script='series.py', model_object_name="series")
    rmodel = RunModel(model=model)
    regression_model = LinearRegression()
    correlation_model = RBF()
    K = GaussianProcessRegression(kernel=correlation_model, hyperparameters=[1, 1, 0.1], random_state=1,
                                  bounds=[[1e-2, 20], [1e-2, 20], [1e-5, 1e-2]],
                                  optimizer=MinimizeOptimizer('l-bfgs-b'), regression_model=regression_model,
                                  optimizations_number=10)
    # OPTIONS: 'U', 'EFF', 'Weighted-U'
    learning_function = ExpectedImprovement()
    a = AdaptiveKriging(distributions=marginals, runmodel_object=rmodel, surrogate=K,
                        learning_nsamples=10 ** 3, n_add=1, learning_function=learning_function,
                        random_state=2)
    a.run(nsamples=25, samples=x.samples)

    assert a.samples[20, 1] == 2.1884195849577157


def test_akmcs_expected_improvement_global_fit():
    from UQpy.surrogates.gaussian_process.regression_models.LinearRegression import LinearRegression
    from UQpy.surrogates.gaussian_process.kernels.RBF import RBF

    marginals = [Normal(loc=0., scale=4.), Normal(loc=0., scale=4.)]
    x = MonteCarloSampling(distributions=marginals, nsamples=20, random_state=1)
    model = PythonModel(model_script='series.py', model_object_name="series")
    rmodel = RunModel(model=model)
    regression_model = LinearRegression()
    correlation_model = RBF()
    K = GaussianProcessRegression(kernel=correlation_model, hyperparameters=[1, 1, 0.1], random_state=1,
                                  optimizer=MinimizeOptimizer('l-bfgs-b'), regression_model=regression_model,
                                  optimizations_number=10, bounds=[[1e-2, 20], [1e-2, 20], [1e-5, 1e-2]],)
    # OPTIONS: 'U', 'EFF', 'Weighted-U'
    learning_function = ExpectedImprovementGlobalFit()
    a = AdaptiveKriging(distributions=marginals, runmodel_object=rmodel, surrogate=K,
                        learning_nsamples=10 ** 3, n_add=1, learning_function=learning_function,
                        random_state=2)
    a.run(nsamples=25, samples=x.samples)

    assert a.samples[23, 0] == -10.24267076486663
    assert a.samples[20, 1] == 0.17610325620498946


def test_akmcs_samples_error():
    from UQpy.surrogates.gaussian_process.regression_models.LinearRegression import LinearRegression
    from UQpy.surrogates.gaussian_process.kernels.RBF import RBF

    marginals = [Normal(loc=0., scale=4.), Normal(loc=0., scale=4.)]
    x = MonteCarloSampling(distributions=marginals, nsamples=20, random_state=0)
    model = PythonModel(model_script='series.py', model_object_name="series")
    rmodel = RunModel(model=model)
    regression_model = LinearRegression()
    correlation_model = RBF()
    K = GaussianProcessRegression(kernel=correlation_model, hyperparameters=[1, 1, 0.1], random_state=1,
                                  optimizer=MinimizeOptimizer('l-bfgs-b'), regression_model=regression_model,
                                  optimizations_number=10, bounds=[[1e-2, 20], [1e-2, 20], [1e-5, 1e-2]],)
    # OPTIONS: 'U', 'EFF', 'Weighted-U'
    learning_function = WeightedUFunction(weighted_u_stop=2)
    with pytest.raises(NotImplementedError):
        a = AdaptiveKriging(distributions=[Normal(loc=0., scale=4.)] * 3, runmodel_object=rmodel, surrogate=K,
                            learning_nsamples=10 ** 3, n_add=1, learning_function=learning_function,
                            random_state=2, samples=x.samples)


def test_akmcs_u_run_from_init():
    from UQpy.surrogates.gaussian_process.regression_models.LinearRegression import LinearRegression
    from UQpy.surrogates.gaussian_process.kernels.RBF import RBF

    marginals = [Normal(loc=0., scale=4.), Normal(loc=0., scale=4.)]
    x = MonteCarloSampling(distributions=marginals, nsamples=20, random_state=1)
    model = PythonModel(model_script='series.py', model_object_name="series")
    rmodel = RunModel(model=model)
    regression_model = LinearRegression()
    correlation_model = RBF()
    K = GaussianProcessRegression(kernel=correlation_model, hyperparameters=[1, 1, 0.1], random_state=1,
                                  optimizer=MinimizeOptimizer('l-bfgs-b'), regression_model=regression_model,
                                  optimizations_number=10, bounds=[[1e-2, 20], [1e-2, 20], [1e-5, 1e-2]])
    # OPTIONS: 'U', 'EFF', 'Weighted-U'
    learning_function = UFunction(u_stop=2)
    a = AdaptiveKriging(distributions=marginals, runmodel_object=rmodel, surrogate=K,
                        learning_nsamples=10 ** 3, n_add=1, learning_function=learning_function,
                        random_state=2, nsamples=25, samples=x.samples)

    assert a.samples[23, 0] == 2.3364698907681873
    assert a.samples[20, 1] == -0.35419762304600505
