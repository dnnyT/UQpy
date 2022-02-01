import numpy as np
from UQpy.Distributions import Uniform
from UQpy.SampleMethods import MH, ParallelTemperingMCMC


def log_rosenbrock(x):
    return -(100 * (x[:, 1] - x[:, 0] ** 2) ** 2 + (1 - x[:, 0]) ** 2) / 20


def log_intermediate(x, beta):
    return beta * log_rosenbrock(x)


def log_prior(x):
    loc, scale = -20., 40.
    return Uniform(loc=loc, scale=scale).log_pdf(x[:, 0]) + Uniform(loc=loc, scale=scale).log_pdf(x[:, 1])


def compute_potential(x, beta, log_intermediate_values):
    return log_intermediate_values / beta


random_state = np.random.RandomState(1234)
seed = -2. + 4. * random_state.rand(5, 2)
betas = [1. / np.sqrt(2.) ** i for i in range(10 - 1, -1, -1)]


def test_parallel():
    mcmc = ParallelTemperingMCMC(log_pdf_intermediate=log_intermediate, log_pdf_reference=log_prior,
                                 niter_between_sweeps=4, betas=betas, save_log_pdf=True,
                                 mcmc_class=MH, nburn=10, jump=2, seed=seed, dimension=2, random_state=3456)
    mcmc.run(nsamples_per_chain=100)
    assert mcmc.samples.shape == (500, 2)


def test_thermodynamic_integration():
    mcmc = ParallelTemperingMCMC(log_pdf_intermediate=log_intermediate, log_pdf_reference=log_prior,
                                 niter_between_sweeps=4, betas=betas, save_log_pdf=True,
                                 mcmc_class=MH, nburn=10, jump=2, seed=seed, dimension=2, random_state=3456)
    mcmc.run(nsamples_per_chain=100)
    log_ev = mcmc.evaluate_normalization_constant(compute_potential=compute_potential, log_p0=0.)
    assert np.round(np.exp(log_ev), 4) == 0.1885

