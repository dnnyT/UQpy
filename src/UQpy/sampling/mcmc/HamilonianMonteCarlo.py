import logging
from typing import Callable
from beartype import beartype

from UQpy.sampling.mcmc.baseclass.MCMC import MCMC
from UQpy.distributions import *
from UQpy.utilities.ValidationTypes import *
import autograd.numpy as np
from autograd import grad
import warnings

warnings.filterwarnings('ignore')


class HamiltonianMonteCarlo(MCMC):

    @beartype
    def __init__(
            self,
            pdf_target: Union[Callable, list[Callable]] = None,
            log_pdf_target: Union[Callable, list[Callable]] = None,
            U: Union[Callable, list[Callable]] = None,
            grad_U: Union[Callable, list[Callable]] = None,
            K: Union[Callable, list[Callable]] = None,
            args_target: tuple = None,
            burn_length: Annotated[int, Is[lambda x: x >= 0]] = 0,
            jump: int = 1,
            dimension: int = None,
            seed: list = None,
            save_log_pdf: bool = False,
            concatenate_chains: bool = True,
            n_chains: int = None,
            proposal: Distribution = None,
            proposal_is_symmetric: bool = False,
            random_state: RandomStateType = None,
            nsamples: PositiveInteger = None,
            nsamples_per_chain: PositiveInteger = None,
            epsilon: PositiveFloat = None,
            initialize_tf: bool = False,
            leapfrog_steps: PositiveInteger = None,
            mean_divider: PositiveFloat = 8.,
            cons_samples: PositiveInteger = 10,
    ):
        """
        Hamiltonian Monte Carlo

        :param pdf_target: Target density function from which to draw random samples. Either `pdf_target` or
         `log_pdf_target` must be provided (the latter should be preferred for better numerical stability).

         If `pdf_target` is a callable, it refers to the joint pdf to sample from, it must take at least one input
         **x**, which are the point(s) at which to evaluate the pdf. Within :class:`.MCMC` the pdf_target is evaluated
         as:
         :code:`p(x) = pdf_target(x, \*args_target)`

         where **x** is a :class:`numpy.ndarray  of shape :code:`(nsamples, dimension)` and `args_target` are additional
         positional arguments that are provided to :class:`.MCMC` via its `args_target` input.

         If `pdf_target` is a list of callables, it refers to independent marginals to sample from. The marginal in
         dimension :code:`j` is evaluated as:
         :code:`p_j(xj) = pdf_target[j](xj, \*args_target[j])` where **x** is a :class:`numpy.ndarray` of shape
         :code:`(nsamples, dimension)`
        :param log_pdf_target: Logarithm of the target density function from which to draw random samples.
         Either `pdf_target` or `log_pdf_target` must be provided (the latter should be preferred for better numerical
         stability).

         Same comments as for input `pdf_target`.

        :param grad_U: Gradient of the "potential energy U" (U = -log(pdf)). Define the gradient of U manually if
        autograd does not work with predefined function from scipy
        :param K: Equation for "kinetic energy K". The standard setting is K = 1/2 p^2.
        :param epsilon: stepsize of leapfrog algorithm
        :param leapfrog_steps: Number of steps performed in leapfrog algorithm. Simulated time is
         leapfrog_steps * epsilon.
        :param initialize_tf: Boolean to decide if trajectory length is calculated. The algorithm is adopted from
        Wang2020, it updates the number of leapfrog steps using the calculated trajectory length and the fixed epsilon.
        :param args_target: Positional arguments of the pdf / log-pdf target function. See `pdf_target`
        :param burn_length: Length of burn-in - i.e., number of samples at the beginning of the chain to discard (note:
         no thinning during burn-in). Default is :math:`0`, no burn-in.
        :param jump: Thinning parameter, used to reduce correlation between samples. Setting :code:`jump=n` corresponds
         to skipping :code:`n-1` states between accepted states of the chain. Default is :math:`1` (no thinning).
        :param dimension: A scalar value defining the dimension of target density function. Either `dimension` and
         `n_chains` or `seed` must be provided.
        :param seed: Seed of the Markov chain(s), shape :code:`(n_chains, dimension)`.
         Default: :code:`zeros(n_chains x dimension)`.

         If seed is not provided, both n_chains and dimension must be provided.
        :param save_log_pdf: Boolean that indicates whether to save log-pdf values along with the samples.
         Default: :any:`False`
        :param concatenate_chains: Boolean that indicates whether to concatenate the chains after a run, i.e., samples
         are stored as an :class:`numpy.ndarray` of shape :code:`(nsamples * n_chains, dimension)` if :any:`True`,
         :code:`(nsamples, n_chains, dimension)` if :any:`False`.
         Default: :any:`True`
        :param n_chains: The number of Markov chains to generate. Either dimension and `n_chains` or `seed` must be
         provided.
        :param proposal: Proposal distribution, must have a log_pdf/pdf and rvs method. Default: standard
         multivariate normal
        :param proposal_is_symmetric: Indicates whether the proposal distribution is symmetric, affects computation of
         acceptance probability alpha Default: :any:`False`, set to :any:`True` if default proposal is used
        :param random_state: Random seed used to initialize the pseudo-random number generator. Default is
         :any:`None`.


        :param nsamples: Number of samples to generate.
        :param nsamples_per_chain: Number of samples to generate per chain.
        """
        self.nsamples = nsamples
        self.nsamples_per_chain = nsamples_per_chain
        super().__init__(
            pdf_target=pdf_target,
            log_pdf_target=log_pdf_target,
            args_target=args_target,
            dimension=dimension,
            seed=seed,
            burn_length=burn_length,
            jump=jump,
            save_log_pdf=save_log_pdf,
            concatenate_chains=concatenate_chains,
            random_state=random_state,
            n_chains=n_chains,
        )
        # Initialize algorithm specific inputs
        self.grad_U = grad_U
        self.K = K
        self.U = U
        self.epsilon = epsilon
        self.leapfrog_steps = leapfrog_steps
        self.initialize_tf = initialize_tf
        self.cons_samples= cons_samples
        self.mean_divider = mean_divider

        self.logger = logging.getLogger(__name__)

        self.proposal = proposal
        self.proposal_is_symmetric = proposal_is_symmetric
        if self.proposal is None:
            if self.dimension is None:
                raise ValueError("UQpy: Either input proposal or dimension must be provided.")
            from UQpy.distributions import JointIndependent, Normal

            self.proposal = JointIndependent([Normal()] * self.dimension)
            self.proposal_is_symmetric = True
        else:
            self._check_methods_proposal(self.proposal)

        self.logger.info("\nUQpy: Initialization of " + self.__class__.__name__ + " algorithm complete.")

        if self.K is None:
            self.K = lambda p: p * p.T * 0.5

        if self.grad_U is None:
            U = lambda q: -self.log_pdf_target(q)
            self.grad_U = grad(U)
            self.logger.info("Grad U is defined within the HMC algorithm.")

        if (nsamples is not None) or (nsamples_per_chain is not None):
            self.run(nsamples=nsamples, nsamples_per_chain=nsamples_per_chain, )

        # Initialize simulation time with given epsilon and set leapfrog steps automatically
        if self.initialize_tf:
            self._eval_init_tf(self.seed, mean_divider=self.mean_divider,n_samples_considered=self.cons_samples)

    def _leapfrog(self, q_0, p_0):
        q = np.asarray(q_0)
        p = np.asarray(p_0)
        p = p - self.epsilon / 2 * self.grad_U(q)
        q = q + self.epsilon * p
        for i in range(self.leapfrog_steps - 1):
            p = p - self.epsilon * self.grad_U(q)
            q = q + self.epsilon * p
        p = p - self.epsilon / 2 * self.grad_U(q)
        return q, p

    def _calc_H(self, q, p):
        q = np.asarray(q)
        p = np.asarray(p)
        U = -self.evaluate_log_target(q)
        K = self.K(p)
        H = U + K
        return H

    def _eval_init_tf(self, init_samples, mean_divider=8, n_samples_considered=10, set_lpf_steps=True):
        self.logger.info("Trajectory length is calculated to set the number of leapfrog steps")
        p = self.proposal.rvs(
            nsamples=self.n_chains, random_state=self.random_state)
        q = init_samples
        T = []

        for i in range(n_samples_considered):
            self.leapfrog_steps = 1
            q_i_pos, p_i_pos = self._leapfrog([q[i, :]], [p[i, :]])
            q_i_neg, p_i_neg = self._leapfrog([q[i, :]], [[-1] * p[i, :]])
            for j in range(100000):
                val_p = np.matmul(p_i_pos, (q_i_pos - q_i_neg).T)
                val_n = np.matmul(p_i_neg, (q_i_neg - q_i_pos).T)
                if np.squeeze(val_p) < 0. and np.squeeze(val_n) < 0.:
                    T.append(2 * j * self.epsilon)
                    break
                q_i_pos, p_i_pos = self._leapfrog(q_i_pos, p_i_pos)
                q_i_neg, p_i_neg = self._leapfrog(q_i_neg, p_i_neg)

        t_f = np.mean(T) / mean_divider
        self.logger.info("The mean period T ", str(np.mean(T)), "divided by", str(mean_divider))
        self.logger.info("simulation time is:", str(t_f))
        if set_lpf_steps:
            self.leapfrog_steps = int(t_f / self.epsilon)
            self.logger.info("Number of leapfrog steps is set to:", str(self.leapfrog_steps))

        return t_f

    def run_one_iteration(self, current_state: np.ndarray, current_log_pdf: np.ndarray):
        """
        Run one iteration of the mcmc chain for HMC algorithm, starting at current state -
        see :class:`MCMC` class.
        """
        # Initialize simulation time with given epsilon and set leapfrog steps automatically
        #        if self.initialize_tf:
        #            self._eval_init_tf(self.seed)
        # Sample "momentum" from proposal distribution
        current_p = self.proposal.rvs(
            nsamples=self.n_chains, random_state=self.random_state)
        candidate_p = []
        candidate = []
        log_ratios = []
        log_prob_candidate = []
        for i in range(self.n_chains):
            # Leapfrog to
            q_i, p_i = self._leapfrog([current_state[i, :]], [current_p[i, :]])
            p_i = -p_i
            candidate_p.append(p_i)
            candidate.append(q_i)
            # Compute log_pdf_target of candidate sample
            log_pdf_i = self.evaluate_log_target(q_i)
            log_prob_candidate.append(log_pdf_i)
            # Calculate H
            H_candidate = self._calc_H(q_i, p_i)
            H_current = self._calc_H([current_state[i, :]], [current_p[i, :]])

            # Compute acceptance ratio
            r_i = (H_current - H_candidate)
            log_ratios.append(r_i)

        # Compare candidate with current sample and decide or not to keep the candidate (loop over nc chains)
        accept_vec = np.zeros(
            (self.n_chains,)
        )  # this vector will be used to compute accept_ratio of each chain
        unif_rvs = (
            Uniform()
            .rvs(nsamples=self.n_chains, random_state=self.random_state)
            .reshape((-1,))
        )
        for nc, (cand, log_p_cand, r_) in enumerate(
                zip(candidate, log_prob_candidate, log_ratios)
        ):
            accept = np.log(unif_rvs[nc]) <= r_
            if accept:
                current_state[nc, :] = cand
                current_log_pdf[nc] = log_p_cand
                accept_vec[nc] += 1.0
        # Update the acceptance rate
        self._update_acceptance_rate(accept_vec)
        return current_state, current_log_pdf
