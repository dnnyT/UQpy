import numpy as np
import scipy.stats as stats
from scipy.special import logsumexp
from scipy.integrate import simps, trapz
from UQpy.Distributions import *
from UQpy.SampleMethods.MCMC import *
from abc import ABC


class TemperingMCMC(ABC):
    """
    Parent class to parallel and sequential tempering MCMC algorithms.

    To sample from the target distribution :math:`p(x)`, a sequence of intermediate densities
    :math:`p(x, \beta) \propto q(x, \beta) p_{0}(x)` for values of the parameter :math:`\beta` between 0 and 1,
    where :math:`p_{0}` is a reference distribution (often set as the prior in a Bayesian setting).
    Setting :math:`\beta = 1` equates sampling from the target, while
    :math:`\beta \rightarrow 0` samples from the reference distribution.

    **Inputs:**

    **Methods:**
    """

    def __init__(self, pdf_intermediate=None, log_pdf_intermediate=None, args_pdf_intermediate=(),
                 distribution_reference=None, dimension=None, save_log_pdf=False, verbose=False,
                 random_state=None):

        # Check a few inputs
        self.dimension = dimension
        self.save_log_pdf = save_log_pdf
        if isinstance(random_state, int) or random_state is None:
            self.random_state = np.random.RandomState(random_state)
        elif not isinstance(self.random_state, np.random.RandomState):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')
        self.verbose = verbose

        # Initialize the prior and likelihood
        self.evaluate_log_intermediate = self._preprocess_intermediate(
            log_pdf_=log_pdf_intermediate, pdf_=pdf_intermediate, args=args_pdf_intermediate)
        if not (isinstance(distribution_reference, Distribution) or (distribution_reference is None)):
            raise TypeError('UQpy: if provided, input distribution_reference should be a UQpy.Distribution object.')
        # self.evaluate_log_reference = self._preprocess_reference(dist_=distribution_reference, args=())

        # Initialize the outputs
        self.samples = None
        self.intermediate_samples = None
        if self.save_log_pdf:
            self.log_pdf_values = None

    def run(self, nsamples):
        """ Run the tempering MCMC algorithms to generate nsamples from the target posterior """
        pass

    def evaluate_normalization_constant(self, **kwargs):
        """ Computes the normalization constant :math:`Z_{1}=\int{q_{1}(x) p_{0}(x)dx}` where p0 is the reference pdf
         and q1 is the intermediate density with :math:`\beta=1`, thus q1 p0 is the target pdf."""
        pass

    def _preprocess_reference(self, dist_, **kwargs):
        """
        Preprocess the target pdf inputs.

        Utility function (static method), that transforms the log_pdf, pdf, args inputs into a function that evaluates
        log_pdf_target(x) for a given x. If the target is given as a list of callables (marginal pdfs), the list of
        log margianals is also returned.

        **Inputs:**

        * dist_ (distribution object)

        **Output/Returns:**

        * evaluate_log_pdf (callable): Callable that computes the log of the target density function

        """
        # log_pdf is provided
        if dist_ is not None:
            if isinstance(dist_, Distribution):
                evaluate_log_pdf = (lambda x: dist_.log_pdf(x))
            else:
                raise TypeError('UQpy: A UQpy.Distribution object must be provided.')
        else:
            evaluate_log_pdf = None
        return evaluate_log_pdf

    @staticmethod
    def _preprocess_intermediate(log_pdf_, pdf_, args):
        """
        Preprocess the target pdf inputs.

        Utility function (static method), that transforms the log_pdf, pdf, args inputs into a function that evaluates
        log_pdf_target(x, beta) for a given x. If the target is given as a list of callables (marginal pdfs), the list of
        log margianals is also returned.

        **Inputs:**

        * log_pdf_ (callable): Log of the target density function from which to draw random samples. Either
          pdf_target or log_pdf_target must be provided.
        * pdf_ (callable): Target density function from which to draw random samples. Either pdf_target or
          log_pdf_target must be provided.
        * args (tuple): Positional arguments of the pdf target.

        **Output/Returns:**

        * evaluate_log_pdf (callable): Callable that computes the log of the target density function

        """
        # log_pdf is provided
        if log_pdf_ is not None:
            if callable(log_pdf_):
                if args is None:
                    args = ()
                evaluate_log_pdf = (lambda x, temper_param: log_pdf_(x, temper_param, *args))
            else:
                raise TypeError('UQpy: log_pdf_intermediate must be a callable')
        # pdf is provided
        elif pdf_ is not None:
            if callable(pdf_):
                if args is None:
                    args = ()
                evaluate_log_pdf = (lambda x, temper_param: np.log(
                    np.maximum(pdf_(x, temper_param, *args), 10 ** (-320) * np.ones((x.shape[0],)))))
            else:
                raise TypeError('UQpy: pdf_intermediate must be a callable')
        else:
            raise ValueError('UQpy: log_pdf_intermediate or pdf_intermediate must be provided')
        return evaluate_log_pdf

    @staticmethod
    def _target_generator(intermediate_logpdf_, reference_logpdf_, temper_param_):
        evaluate_log_pdf = (lambda x: (reference_logpdf_(x) + intermediate_logpdf_(x, temper_param_)))
        return evaluate_log_pdf


class ParallelTemperingMCMC(TemperingMCMC):
    """
    Parallel-Tempering MCMC

    This algorithms runs the chains sampling from various tempered distributions in parallel. Periodically during the
    run, the different temperatures swap members of their ensemble in a way that
    preserves detailed balance.The chains closer to the reference chain (hot chains) can sample from regions that have
    low probability under the target and thus allow a better exploration of the parameter space, while the cold chains
    can better explore the regions of high likelihood.

    **References**

    1. Parallel Tempering: Theory, Applications, and New Perspectives, Earl and Deem
    2. Adaptive Parallel Tempering MCMC
    3. emcee the MCMC Hammer python package

    **Inputs:**

    Many inputs are similar to MCMC algorithms. Additional inputs are:

    * **niter_between_sweeps**

    * **mcmc_class**

    **Methods:**

    """

    def __init__(self, niter_between_sweeps, pdf_intermediate=None, log_pdf_intermediate=None, args_pdf_intermediate=(),
                 distribution_reference=None, nburn=0, jump=1, dimension=None, seed=None,
                 save_log_pdf=False, nsamples=None, nsamples_per_chain=None, nchains=None, verbose=False,
                 random_state=None, temper_param_list=None, n_temper_params=None, mcmc_class=MH, **kwargs_mcmc):

        super().__init__(pdf_intermediate=pdf_intermediate, log_pdf_intermediate=log_pdf_intermediate,
                         args_pdf_intermediate=args_pdf_intermediate, distribution_reference=None, dimension=dimension,
                         save_log_pdf=save_log_pdf, verbose=verbose, random_state=random_state)
        self.distribution_reference = distribution_reference
        self.evaluate_log_reference = self._preprocess_reference(self.distribution_reference)

        # Initialize PT specific inputs: niter_between_sweeps and temperatures
        self.niter_between_sweeps = niter_between_sweeps
        if not (isinstance(self.niter_between_sweeps, int) and self.niter_between_sweeps >= 1):
            raise ValueError('UQpy: input niter_between_sweeps should be a strictly positive integer.')
        self.temper_param_list = temper_param_list
        self.n_temper_params = n_temper_params
        if self.temper_param_list is None:
            if self.n_temper_params is None:
                raise ValueError('UQpy: either input temper_param_list or n_temper_params should be provided.')
            elif not (isinstance(self.n_temper_params, int) and self.n_temper_params >= 2):
                raise ValueError('UQpy: input n_temper_params should be a integer >= 2.')
            else:
                self.temper_param_list = [1. / np.sqrt(2) ** i for i in range(self.n_temper_params-1, -1, -1)]
        elif (not isinstance(self.temper_param_list, (list, tuple))
              or not (all(isinstance(t, (int, float)) and (t > 0 and t <= 1.) for t in self.temper_param_list))
              #or float(self.temperatures[0]) != 1.
        ):
            raise ValueError(
                'UQpy: temper_param_list should be a list of floats in [0, 1], starting at 0. and increasing to 1.')
        else:
            self.n_temper_params = len(self.temper_param_list)

        # Initialize mcmc objects, need as many as number of temperatures
        if not issubclass(mcmc_class, MCMC):
            raise ValueError('UQpy: mcmc_class should be a subclass of MCMC.')
        if not all((isinstance(val, (list, tuple)) and len(val) == self.n_temper_params)
                   for val in kwargs_mcmc.values()):
            raise ValueError(
                'UQpy: additional kwargs arguments should be mcmc algorithm specific inputs, given as lists of length '
                'the number of temperatures.')
        # default value
        if isinstance(mcmc_class, MH) and len(kwargs_mcmc) == 0:
            from UQpy.Distributions import JointInd, Normal
            kwargs_mcmc = {'proposal_is_symmetric': [True, ] * self.n_temper_params,
                           'proposal': [JointInd([Normal(scale=1./np.sqrt(temper_param))] * dimension)
                                        for temper_param in self.temper_param_list]}

        # Initialize algorithm specific inputs: target pdfs
        self.thermodynamic_integration_results = None

        self.mcmc_samplers = []
        for i, temper_param in enumerate(self.temper_param_list):
            #log_pdf_target = self._target_generator(
            #    self.evaluate_log_intermediate, self.evaluate_log_reference, temper_param)
            log_pdf_target = (lambda x, temper_param=temper_param: self.evaluate_log_reference(
                x) + self.evaluate_log_intermediate(x, temper_param))
            self.mcmc_samplers.append(
                mcmc_class(log_pdf_target=log_pdf_target,
                           dimension=dimension, seed=seed, nburn=nburn, jump=jump, save_log_pdf=save_log_pdf,
                           concat_chains=True, verbose=verbose, random_state=self.random_state, nchains=nchains,
                            **dict([(key, val[i]) for key, val in kwargs_mcmc.items()])))

        # Samples connect to posterior samples, i.e. the chain with temperature 1.
        #self.samples = self.mcmc_samplers[0].samples
        #if self.save_log_pdf:
        #    self.log_pdf_values = self.mcmc_samplers[0].samples

        if self.verbose:
            print('\nUQpy: Initialization of ' + self.__class__.__name__ + ' algorithm complete.')

        # If nsamples is provided, run the algorithm
        if (nsamples is not None) or (nsamples_per_chain is not None):
            self.run(nsamples=nsamples, nsamples_per_chain=nsamples_per_chain)

    def run(self, nsamples=None, nsamples_per_chain=None):
        """
        Run the MCMC algorithm.

        This function samples from the MCMC chains and appends samples to existing ones (if any). This method leverages
        the ``run_iterations`` method that is specific to each algorithm.

        **Inputs:**

        * **nsamples** (`int`):
            Number of samples to generate.

        * **nsamples_per_chain** (`int`)
            Number of samples to generate per chain.

        Either `nsamples` or `nsamples_per_chain` must be provided (not both). Not that if `nsamples` is not a multiple
        of `nchains`, `nsamples` is set to the next largest integer that is a multiple of `nchains`.

        """
        # Initialize the runs: allocate space for the new samples and log pdf values
        final_ns, final_ns_per_chain, current_state_t, current_log_pdf_t = self.mcmc_samplers[0]._initialize_samples(
            nsamples=nsamples, nsamples_per_chain=nsamples_per_chain)
        current_state, current_log_pdf = [current_state_t.copy(), ], [current_log_pdf_t.copy(), ]
        for mcmc_sampler in self.mcmc_samplers[1:]:
            _, _, current_state_t, current_log_pdf_t = mcmc_sampler._initialize_samples(
                nsamples=nsamples, nsamples_per_chain=nsamples_per_chain)
            current_state.append(current_state_t.copy())
            current_log_pdf.append(current_log_pdf_t.copy())

        if self.verbose:
            print('UQpy: Running MCMC...')

        # Run nsims iterations of the MCMC algorithm, starting at current_state
        while self.mcmc_samplers[0].nsamples_per_chain < final_ns_per_chain:
            # update the total number of iterations
            # self.mcmc_samplers[0].niterations += 1

            # run one iteration of MCMC algorithms at various temperatures
            new_state, new_log_pdf = [], []
            for t, sampler in enumerate(self.mcmc_samplers):
                sampler.niterations += 1
                new_state_t, new_log_pdf_t = sampler.run_one_iteration(
                    current_state[t], current_log_pdf[t])
                new_state.append(new_state_t.copy())
                new_log_pdf.append(new_log_pdf_t.copy())

            # Do sweeps if necessary
            if self.mcmc_samplers[-1].niterations % self.niter_between_sweeps == 0:
                for i in range(self.n_temper_params - 1):
                    log_accept = (self.mcmc_samplers[i].evaluate_log_target(new_state[i + 1]) +
                                  self.mcmc_samplers[i + 1].evaluate_log_target(new_state[i]) -
                                  self.mcmc_samplers[i].evaluate_log_target(new_state[i]) -
                                  self.mcmc_samplers[i + 1].evaluate_log_target(new_state[i + 1]))
                    for nc, log_accept_chain in enumerate(log_accept):
                        if np.log(self.random_state.rand()) < log_accept_chain:
                            new_state[i][nc], new_state[i + 1][nc] = new_state[i + 1][nc], new_state[i][nc]
                            new_log_pdf[i][nc], new_log_pdf[i + 1][nc] = new_log_pdf[i + 1][nc], new_log_pdf[i][nc]

            # Update the chain, only if burn-in is over and the sample is not being jumped over
            # also increase the current number of samples and samples_per_chain
            if self.mcmc_samplers[-1].niterations > self.mcmc_samplers[-1].nburn and \
                    (self.mcmc_samplers[-1].niterations - self.mcmc_samplers[-1].nburn) % self.mcmc_samplers[-1].jump == 0:
                for t, sampler in enumerate(self.mcmc_samplers):
                    sampler.samples[sampler.nsamples_per_chain, :, :] = new_state[t].copy()
                    if self.save_log_pdf:
                        sampler.log_pdf_values[sampler.nsamples_per_chain, :] = new_log_pdf[t].copy()
                    sampler.nsamples_per_chain += 1
                    sampler.nsamples += sampler.nchains
                #self.nsamples_per_chain += 1
                #self.nsamples += self.nchains

        if self.verbose:
            print('UQpy: MCMC run successfully !')

        # Concatenate chains maybe
        if self.mcmc_samplers[-1].concat_chains:
            for t, mcmc_sampler in enumerate(self.mcmc_samplers):
                mcmc_sampler._concatenate_chains()

        # Samples connect to posterior samples, i.e. the chain with beta=1.
        self.intermediate_samples = [sampler.samples for sampler in self.mcmc_samplers]
        self.samples = self.mcmc_samplers[-1].samples
        if self.save_log_pdf:
            self.log_pdf_values = self.mcmc_samplers[-1].log_pdf_values

    def evaluate_normalization_constant(self, compute_potential, log_Z0=None, nsamples_from_p0=None):
        """
        Evaluate new log free energy as

        :math:`\log{Z_{1}} = \log{Z_{0}} + \int_{0}^{1} E_{x~p_{beta}} \left[ U_{\beta}(x) \right] d\beta`

        References (for the Bayesian case):
        * https://emcee.readthedocs.io/en/v2.2.1/user/pt/

        **Inputs:**

        * **compute_potential** (callable):
            Function that takes three inputs (`x`, `log_factor_tempered_values`, `beta`) and computes the potential
            :math:`U_{\beta}(x)`. `log_factor_tempered_values` are the values saved during sampling of
            :math:`\log{p_{\beta}(x)}` at saved samples x.

        * **log_Z0** (`float`):
            Value of :math:`\log{Z_{0}}`

        * **nsamples_from_p0** (`int`):
            N samples from the reference distribution p0. Then :math:`\log{Z_{0}}` is evaluate via MC sampling
            as :math:`\frac{1}{N} \sum{p_{\beta=0}(x)}`. Used only if input *log_Z0* is not provided.

        """
        if not self.save_log_pdf:
            raise NotImplementedError('UQpy: the evidence cannot be computed when save_log_pdf is set to False.')
        if log_Z0 is None and nsamples_from_p0 is None:
            raise ValueError('UQpy: input log_Z0 or nsamples_from_p0 should be provided.')
        # compute average of log_target for the target at various temperatures
        log_pdf_averages = []
        for i, (temper_param, sampler) in enumerate(zip(self.temper_param_list, self.mcmc_samplers)):
            log_factor_values = sampler.log_pdf_values - self.evaluate_log_reference(sampler.samples)
            potential_values = compute_potential(
                x=sampler.samples, temper_param=temper_param, log_intermediate_values=log_factor_values)
            log_pdf_averages.append(np.mean(potential_values))

        # use quadrature to integrate between 0 and 1
        temper_param_list_for_integration = np.copy(np.array(self.temper_param_list))
        log_pdf_averages = np.array(log_pdf_averages)
        #if self.temper_param_list[-1] != 1.:
            #log_pdf_averages = np.append(log_pdf_averages, log_pdf_averages[-1])
            #slope_linear = (log_pdf_averages[-1]-log_pdf_averages[-2]) / (
            #        betas_for_integration[-1] - betas_for_integration[-2])
            #log_pdf_averages = np.append(
            #    log_pdf_averages, log_pdf_averages[-1] + (1. - betas_for_integration[-1]) * slope_linear)
            #betas_for_integration = np.append(betas_for_integration, 1.)
        int_value = trapz(x=temper_param_list_for_integration, y=log_pdf_averages)
        if log_Z0 is None:
            samples_p0 = self.distribution_reference.rvs(nsamples=nsamples_from_p0)
            log_Z0 = np.log(1./nsamples_from_p0) + logsumexp(
                self.evaluate_log_intermediate(x=samples_p0, temper_param=self.temper_param_list[0]))

        self.thermodynamic_integration_results = {
            'log_Z0': log_Z0, 'temper_param_list': temper_param_list_for_integration,
            'expect_potentials': log_pdf_averages}

        return np.exp(int_value + log_Z0)


class SequentialTemperingMCMC(TemperingMCMC):
    """
    Sequential-Tempering MCMC

    This algorithms samples from a series of intermediate targets that are each tempered versions of the final/true
    target. In going from one intermediate distribution to the next, the existing samples are resampled according to
    some weights (similar to importance sampling). To ensure that there aren't a large number of duplicates, the
    resampling step is followed by a short (or even single-step) MCMC run that disperses the samples while remaining
    within the correct intermediate distribution. The final intermediate target is the required target distribution.

    **References**

    1. Ching and Chen, "Transitional Markov Chain Monte Carlo Method for Bayesian Model Updating,
       Model Class Selection, and Model Averaging", Journal of Engineering Mechanics/ASCE, 2007

    **Inputs:**

    Many inputs are similar to MCMC algorithms. Additional inputs are:

    * **mcmc_class**
    * **recalc_w**
    * **nburn_resample**
    * **nburn_mcmc**

    **Methods:**
    """

    def __init__(self, pdf_intermediate=None, log_pdf_intermediate=None, args_pdf_intermediate=(),
                 distribution_reference=None, dimension=None, seed=None, nsamples=None, recalc_w=False,
                 nburn_resample=0, nburn_mcmc=0, jump_mcmc=1, save_intermediate_samples=False, nchains=1,
                 percentage_resampling=100, mcmc_class=MH, proposal=None, proposal_is_symmetric=False,
                 save_log_pdf=False, verbose=False, random_state=None, **kwargs_mcmc):

        super().__init__(pdf_intermediate=pdf_intermediate, log_pdf_intermediate=log_pdf_intermediate,
                         args_pdf_intermediate=args_pdf_intermediate, distribution_reference=distribution_reference,
                         dimension=dimension, save_log_pdf=save_log_pdf, verbose=verbose, random_state=random_state)

        # Initialize inputs
        self.save_intermediate_samples = save_intermediate_samples
        self.recalc_w = recalc_w
        self.nburn_resample = nburn_resample
        self.nburn_mcmc = nburn_mcmc
        self.jump_mcmc = jump_mcmc
        self.nchains = nchains
        self.resample_frac = percentage_resampling / 100

        self.nspc = int(np.floor(((1 - self.resample_frac) * nsamples) / self.nchains))
        self.nresample = int(nsamples - (self.nspc * self.nchains))

        if not issubclass(mcmc_class, MCMC):
            raise ValueError('UQpy: mcmc_class should be a subclass of MCMC.')
        self.mcmc_class = mcmc_class

        self.random_state = random_state
        if isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')

        # Initialize input distributions
        self.evaluate_log_reference, self.seed = self._preprocess_reference(dist_=distribution_reference, args=(),
                                                                            seed_=seed, nsamples=nsamples,
                                                                            dimension=self.dimension)

        self.proposal = proposal
        self.proposal_is_symmetric = proposal_is_symmetric

        # Initialize flag that indicates whether default proposal is to be used (default proposal defined adaptively
        # during run)
        if self.proposal is None:
            self.proposal_given_flag = False
        else:
            self.proposal_given_flag = True

        # Initialize attributes
        self.temper_param_list = None
        self.evidence = None
        self.evidence_cov = None

        # Call the run function
        if nsamples is not None:
            if isinstance(nsamples, int) and nsamples > 0:
                self.run(nsamples=nsamples)
            else:
                raise ValueError('UQpy: "nsamples" must be an integer greater than 0')
        else:
            raise ValueError('UQpy: a value for "nsamples" must be specified ')

    def run(self, nsamples=None):

        if self.verbose:
            print('TMCMC Start')

        if self.samples is not None:
            raise RuntimeError('UQpy: run method cannot be called multiple times for the same object')

        pts = self.seed     # Generated Samples from prior for zero-th tempering level

        # Initializing other variables
        temper_param = 0.0   # Intermediate exponent
        temper_param_prev = temper_param
        self.temper_param_list = np.array(temper_param)
        pts_index = np.arange(nsamples)     # Array storing sample indices
        w = np.zeros(nsamples)              # Array storing plausibility weights
        wp = np.zeros(nsamples)             # Array storing plausibility weight probabilities
        exp_q0 = 0
        for i in range(nsamples):
            exp_q0 += np.exp(self.evaluate_log_intermediate(pts[i, :].reshape((1, -1)), 0.0))
        S = exp_q0/nsamples

        if self.save_intermediate_samples is True:
            self.intermediate_samples = []
            self.intermediate_samples += [pts.copy()]

        # Looping over all adaptively decided tempering levels
        while temper_param < 1:

            # Adaptively set the tempering exponent for the current level
            temper_param_prev = temper_param
            temper_param = self._find_temper_param(temper_param_prev, pts, self.evaluate_log_intermediate, nsamples)
            # d_exp = temper_param - temper_param_prev
            self.temper_param_list = np.append(self.temper_param_list, temper_param)

            if self.verbose:
                print('beta selected')

            # Calculate the plausibility weights
            for i in range(nsamples):
                w[i] = np.exp(self.evaluate_log_intermediate(pts[i, :].reshape((1, -1)), temper_param)
                              - self.evaluate_log_intermediate(pts[i, :].reshape((1, -1)), temper_param_prev))

            # Calculate normalizing constant for the plausibility weights (sum of the weights)
            w_sum = np.sum(w)
            # Calculate evidence from each tempering level
            S = S * (w_sum / nsamples)
            # Normalize plausibility weight probabilities
            wp = (w / w_sum)

            # Calculate covariance matrix for the default proposal
            cov_scale = 0.2
            w_th_sum = np.zeros(self.dimension)
            for i in range(nsamples):
                for j in range(self.dimension):
                    w_th_sum[j] += w[i] * pts[i, j]
            sig_mat = np.zeros((self.dimension, self.dimension))
            for i in range(nsamples):
                pts_deviation = np.zeros((self.dimension, 1))
                for j in range(self.dimension):
                    pts_deviation[j, 0] = pts[i, j] - (w_th_sum[j] / w_sum)
                sig_mat += (w[i] / w_sum) * np.dot(pts_deviation,
                                                   pts_deviation.T)  # Normalized by w_sum as per Betz et al
            sig_mat = cov_scale * cov_scale * sig_mat

            mcmc_log_pdf_target = self._target_generator(self.evaluate_log_intermediate,
                                                         self.evaluate_log_reference, temper_param)

            if self.verbose:
                print('Begin Resampling')
            # Resampling and MH-MCMC step
            for i in range(self.nresample):

                # Resampling from previous tempering level
                lead_index = int(np.random.choice(pts_index, p=wp))
                lead = pts[lead_index]

                # Defining the default proposal
                if self.proposal_given_flag is False:
                    self.proposal = MVNormal(lead, cov=sig_mat)

                # Single MH-MCMC step
                x = MH(dimension=self.dimension, log_pdf_target=mcmc_log_pdf_target, seed=lead, nsamples=1,
                       nchains=1, nburn=self.nburn_resample, proposal=self.proposal,
                       proposal_is_symmetric=self.proposal_is_symmetric)

                # Setting the generated sample in the array
                pts[i] = x.samples

                if self.recalc_w:
                    w[i] = np.exp(self.evaluate_log_intermediate(pts[i, :].reshape((1, -1)), temper_param)
                                  - self.evaluate_log_intermediate(pts[i, :].reshape((1, -1)), temper_param_prev))
                    wp[i] = w[i]/w_sum

            if self.verbose:
                print('Begin MCMC')
            mcmc_seed = self._mcmc_seed_generator(resampled_pts=pts[0:self.nresample, :], arr_length=self.nresample,
                                                  seed_length=self.nchains)
            y = self.mcmc_class(log_pdf_target=mcmc_log_pdf_target, seed=mcmc_seed, dimension=self.dimension,
                                nchains=self.nchains, nsamples_per_chain=self.nspc, nburn=self.nburn_mcmc,
                                jump=self.jump_mcmc, concat_chains=True)
            pts[self.nresample:, :] = y.samples

            if self.save_intermediate_samples is True:
                self.intermediate_samples += [pts.copy()]

            if self.verbose:
                print('Tempering level ended')

        # Setting the calculated values to the attributes
        self.samples = pts
        self.evidence = S

    def evaluate_normalization_constant(self):
        return self.evidence

    @staticmethod
    def _find_temper_param(temper_param_prev, samples, q_func, n, iter_lim=1000, iter_thresh=0.00001):
        """
        Find the tempering parameter for the next intermediate target using bisection search between 1.0 and the
        previous tempering parameter (taken to be 0.0 for the first level).

        **Inputs:**

        * **temper_param_prev** ('float'):
            The value of the previous tempering parameter

        * **samples** (`ndarray`):
            Generated samples from the previous intermediate target distribution

        * **q_func** (callable):
            The intermediate distribution (called 'self.evaluate_log_intermediate' in this code)

        * **n** ('int'):
            Number of samples

        * **iter_lim** ('int'):
            Number of iterations to run the bisection search algorithm for, to avoid infinite loops

        * **iter_thresh** ('float'):
            Threshold on the bisection interval, to avoid infinite loops
        """
        bot = temper_param_prev
        top = 1.0
        flag = 0  # Indicates when the tempering exponent has been found (flag = 1 => solution found)
        loop_counter = 0
        while flag == 0:
            loop_counter += 1
            q_scaled = np.zeros(n)
            temper_param_trial = ((bot + top) / 2)
            for i2 in range(0, n):
                q_scaled[i2] = np.exp(q_func(samples[i2, :].reshape((1, -1)), 1)
                                      - q_func(samples[i2, :].reshape((1, -1)), temper_param_prev))
            sigma_1 = np.std(q_scaled)
            mu_1 = np.mean(q_scaled)
            if sigma_1 < mu_1:
                flag = 1
                temper_param_trial = 1
                continue
            for i3 in range(0, n):
                q_scaled[i3] = np.exp(q_func(samples[i3, :].reshape((1, -1)), temper_param_trial)
                                      - q_func(samples[i3, :].reshape((1, -1)), temper_param_prev))
            sigma = np.std(q_scaled)
            mu = np.mean(q_scaled)
            if sigma < (0.9 * mu):
                bot = temper_param_trial
            elif sigma > (1.1 * mu):
                top = temper_param_trial
            else:
                flag = 1
            if loop_counter > iter_lim:
                flag = 2
                raise RuntimeError('UQpy: unable to find tempering exponent due to nonconvergence')
            if top - bot <= iter_thresh:
                flag = 3
                raise RuntimeError('UQpy: unable to find tempering exponent due to nonconvergence')
        return temper_param_trial

    def _preprocess_reference(self, dist_, seed_=None, nsamples=None, dimension=None):
        """
        Preprocess the target pdf inputs.

        Utility function (static method), that if given a distribution object, returns the log pdf of the target
        distribution of the first tempering level (the prior in a Bayesian setting), and generates the samples from this
        level. If instead the samples of the first level are passed, then the function passes these samples to the rest
        of the algorithm, and does a Kernel Density Approximation to estimate the log pdf of the target distribution for
        this level (as specified by the given sample points).

        **Inputs:**

        * seed_ ('ndarray'): The samples of the first tempering level
        * prior_ ('Distribution' object): Target distribution for the first tempering level
        * nsamples (int): Number of samples to be generated
        * dimension (int): The dimension  of the sample space

        **Output/Returns:**

        * evaluate_log_pdf (callable): Callable that computes the log of the target density function (the prior)
        """

        if dist_ is not None and seed_ is not None:
            raise ValueError('UQpy: both prior and seed values cannot be provided')
        elif dist_ is not None:
            if not(isinstance(dist_, Distribution)):
                raise TypeError('UQpy: A UQpy.Distribution object must be provided.')
            else:
                evaluate_log_pdf = (lambda x: dist_.log_pdf(x))
                seed_values = dist_.rvs(nsamples=nsamples)
        elif seed_ is not None:
            if seed_.shape[0] == nsamples and seed_.shape[1] == dimension:
                seed_values = seed_
                kernel = stats.gaussian_kde(seed_)
                evaluate_log_pdf = (lambda x: kernel.logpdf(x))
            else:
                raise TypeError('UQpy: the seed values should be a numpy array of size (nsamples, dimension)')
        else:
            raise ValueError('UQpy: either prior distribution or seed values must be provided')
        return evaluate_log_pdf, seed_values


    @staticmethod
    def _mcmc_seed_generator(resampled_pts, arr_length, seed_length):
        """
        Generates the seed from the resampled samples for the mcmc step

        Utility function (static method), that returns a selection of the resampled points (at any tempering level) to
        be used as the seed for the following mcmc exploration step.

        **Inputs:**

        * resampled_pts ('ndarray'): The resampled samples of the tempering level
        * arr_length (int): Length of resampled_pts
        * seed_length (int): Number of samples needed in the seed (same as nchains)

        **Output/Returns:**

        * evaluate_log_pdf (callable): Callable that computes the log of the target density function (the prior)
        """
        index_arr = np.arange(arr_length)
        seed_indices = np.random.choice(index_arr, size=seed_length, replace=False)
        mcmc_seed = resampled_pts[seed_indices, :]
        return mcmc_seed
