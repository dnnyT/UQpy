from scipy.optimize import minimize

from UQpy.utilities.optimization.baseclass.Optimizer import Optimizer
import logging


class MinimizeOptimizer(Optimizer):

    def __init__(self, method: str = None, bounds=None):
        super().__init__(bounds)
        self.logger = logging.getLogger(__name__)
        self.optimization = minimize
        self.method = method
        self.save_bounds(bounds)
        self.constraints = None

    def save_bounds(self, bounds):
        if self.method in ['Nelder-Mead', 'L-BFGS-B', 'TNC', 'SLSQP', 'Powell', 'trust-constr']:
            self._bounds = bounds
        else:
            self.logger.warning("The selected optimizer method does not support bounds and thus will be ignored.")

    def optimize(self, function, initial_guess, args=()):
        return minimize(function, initial_guess, args=args,
                        method=self.method, bounds=self._bounds,
                        constraints=self.constraints)

    def apply_constraints(self, constraints):
        if self.method in ['COBYLA', 'SLSQP', 'trust-constr']:
            self.constraints = constraints
        else:
            self.logger.warning("The selected optimizer method does not support constraints and thus will be ignored.")

    def update_bounds(self, bounds):
        self.save_bounds(bounds)

    def supports_jacobian(self):
        self.method in ['CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP', 'dogleg', 'trust-ncg',
                        'trust-krylov', 'trust-exact', 'trust-constr']
