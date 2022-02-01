import numpy as np
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
                 log_pdf_reference=None, pdf_reference=None, dimension=None, save_log_pdf=False, verbose=False,
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
        print(type(log_pdf_intermediate))
        print(type(pdf_intermediate))
        print(type(args_pdf_intermediate))
        self.evaluate_log_intermediate = self._preprocess_intermediate(
            log_pdf_=log_pdf_intermediate, pdf_=pdf_intermediate, args=args_pdf_intermediate)
        self.evaluate_log_reference = self._preprocess_reference(
            log_pdf_=log_pdf_reference, pdf_=pdf_reference, args=())

        # Initialize the outputs
        self.samples = None
        if self.save_log_pdf:
            self.log_pdf_values = None

    def run(self, nsamples):
        """ Run the tempering MCMC algorithms to generate nsamples from the target posterior """
        pass

    def evaluate_normalization_constant(self, compute_potential, log_p0=None, samples_p0=None):
        """ Computes the normalization constant :math:`Z_{1}=\int{q_{1}(x) p_{0}(x)dx}` where p0 is the reference pdf
         and q1 is the intermediate density with :math:`\beta=1`, thus q1 p0 is the target pdf."""
        pass

    @staticmethod
    def _preprocess_reference(log_pdf_, pdf_, args):
        """
        Preprocess the target pdf inputs.

        Utility function (static method), that transforms the log_pdf, pdf, args inputs into a function that evaluates
        log_pdf_target(x) for a given x. If the target is given as a list of callables (marginal pdfs), the list of
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
                evaluate_log_pdf = (lambda x: log_pdf_(x, *args))
            else:
                raise TypeError('UQpy: log_pdf_reference must be a callable')
        # pdf is provided
        elif pdf_ is not None:
            if callable(pdf_):
                if args is None:
                    args = ()
                evaluate_log_pdf = (lambda x: np.log(np.maximum(pdf_(x, *args), 10 ** (-320) * np.ones((x.shape[0],)))))
            else:
                raise TypeError('UQpy: pdf_reference must be a callable')
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
                evaluate_log_pdf = (lambda x, beta: log_pdf_(x, beta, *args))
            else:
                raise TypeError('UQpy: log_pdf_intermediate must be a callable')
        # pdf is provided
        elif pdf_ is not None:
            if callable(pdf_):
                if args is None:
                    args = ()
                evaluate_log_pdf = (lambda x, beta: np.log(
                    np.maximum(pdf_(x, beta, *args), 10 ** (-320) * np.ones((x.shape[0],)))))
            else:
                raise TypeError('UQpy: pdf_intermediate must be a callable')
        else:
            raise ValueError('UQpy: log_pdf_intermediate or pdf_intermediate must be provided')
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
                 log_pdf_reference=None, pdf_reference=None, nburn=0, jump=1, dimension=None, seed=None,
                 save_log_pdf=False, nsamples=None, nsamples_per_chain=None, nchains=None, verbose=False,
                 random_state=None, betas=None, nbetas=None, mcmc_class=MH, **kwargs_mcmc):

        super().__init__(pdf_intermediate=pdf_intermediate, log_pdf_intermediate=log_pdf_intermediate,
                         args_pdf_intermediate=args_pdf_intermediate, log_pdf_reference=log_pdf_reference,
                         pdf_reference=pdf_reference, dimension=dimension, save_log_pdf=save_log_pdf, verbose=verbose,
                         random_state=random_state)

        # Initialize PT specific inputs: niter_between_sweeps and temperatures
        self.niter_between_sweeps = niter_between_sweeps
        if not (isinstance(self.niter_between_sweeps, int) and self.niter_between_sweeps >= 1):
            raise ValueError('UQpy: input niter_between_sweeps should be a strictly positive integer.')
        self.betas = betas
        self.nbetas = nbetas
        if self.betas is None:
            if self.nbetas is None:
                raise ValueError('UQpy: either input betas or nbetas should be provided.')
            elif not (isinstance(self.nbetas, int) and self.nbetas >= 2):
                raise ValueError('UQpy: input nbetas should be a integer >= 2.')
            else:
                self.betas = [1. / np.sqrt(2) ** i for i in range(self.nbetas-1, -1, -1)]
        elif (not isinstance(self.betas, (list, tuple))
              or not (all(isinstance(t, (int, float)) and (t > 0 and t <= 1.) for t in self.betas))
              #or float(self.temperatures[0]) != 1.
        ):
            raise ValueError('UQpy: betas should be a list of floats in [0, 1], starting at 0. and increasing to 1.')
        else:
            self.nbetas = len(self.betas)

        # Initialize mcmc objects, need as many as number of temperatures
        if not issubclass(mcmc_class, MCMC):
            raise ValueError('UQpy: mcmc_class should be a subclass of MCMC.')
        if not all((isinstance(val, (list, tuple)) and len(val) == self.nbetas)
                   for val in kwargs_mcmc.values()):
            raise ValueError(
                'UQpy: additional kwargs arguments should be mcmc algorithm specific inputs, given as lists of length '
                'the number of temperatures.')
        # default value
        if isinstance(mcmc_class, MH) and len(kwargs_mcmc) == 0:
            from UQpy.Distributions import JointInd, Normal
            kwargs_mcmc = {'proposal_is_symmetric': [True, ] * self.nbetas,
                           'proposal': [JointInd([Normal(scale=1./np.sqrt(beta))] * dimension) for beta in self.betas]}

        # Initialize algorithm specific inputs: target pdfs
        self.ti_results = None

        #list_of_targets = list(map(lambda beta: lambda x: self._preprocess_target(
        #    log_pdf_=log_factor_tempered, pdf_=factor_tempered, args=(temp,) + args_factor_tempered)[0](x) +
        #                                                           self.evaluate_log_reference(x),
        #                                    self.temperatures))

        self.mcmc_samplers = []
        for i, beta in enumerate(self.betas):
            self.mcmc_samplers.append(
                mcmc_class(log_pdf_target=
                           lambda x, beta=beta: self.evaluate_log_reference(x) + self.evaluate_log_intermediate(x, beta),
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
                for i in range(self.nbetas - 1):
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
        self.samples = self.mcmc_samplers[-1].samples
        if self.save_log_pdf:
            self.log_pdf_values = self.mcmc_samplers[-1].log_pdf_values

    def evaluate_normalization_constant(self, compute_potential, log_p0=None, samples_p0=None):
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

        * **log_p0** (`float`):
            Value of :math:`\log{Z_{0}}`

        * **samples_p0** (`int`):
            N samples from the reference distribution p0. Then :math:`\log{Z_{0}}` is evaluate via MC sampling
            as :math:`\frac{1}{N} \sum{p_{\beta=0}(x)}`. Used only if input *log_p0* is not provided.

        """
        if not self.save_log_pdf:
            raise NotImplementedError('UQpy: the evidence cannot be computed when save_log_pdf is set to False.')
        # compute average of log_target for the target at various temperatures
        log_pdf_averages = []
        for i, (beta, sampler) in enumerate(zip(self.betas, self.mcmc_samplers)):
            log_factor_values = sampler.log_pdf_values - self.evaluate_log_reference(sampler.samples)
            potential_values = compute_potential(
                x=sampler.samples, beta=beta, log_intermediate_values=log_factor_values)
            log_pdf_averages.append(np.mean(potential_values))

        # use quadrature to integrate between 0 and 1
        from scipy.integrate import simps, trapz
        betas_for_integration = np.copy(np.array(self.betas))
        log_pdf_averages = np.array(log_pdf_averages)
        #if self.betas[-1] != 1.:
            #log_pdf_averages = np.append(log_pdf_averages, log_pdf_averages[-1])
            #slope_linear = (log_pdf_averages[-1]-log_pdf_averages[-2]) / (
            #        betas_for_integration[-1] - betas_for_integration[-2])
            #log_pdf_averages = np.append(
            #    log_pdf_averages, log_pdf_averages[-1] + (1. - betas_for_integration[-1]) * slope_linear)
            #betas_for_integration = np.append(betas_for_integration, 1.)
        int_value = trapz(x=betas_for_integration, y=log_pdf_averages)
        if log_p0 is None:
            if samples_p0 is None:
                raise ValueError('UQpy: log_p0 or samples_p0 should be provided.')
            log_p0 = np.log(np.mean(np.exp(self.evaluate_log_intermediate(x=samples_p0, beta=self.betas[0]))))

        self.ti_results = {
            'log_p0': log_p0, 'betas': betas_for_integration, 'expect_potentials': log_pdf_averages}

        return int_value + log_p0