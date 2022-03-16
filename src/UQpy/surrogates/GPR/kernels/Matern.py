from UQpy.surrogates.gpr.kernels.baseclass.Kernel import *
from scipy.spatial.distance import pdist, cdist, squareform


class Matern(Kernel):
    def __init__(self, nu=1.5):
        self.nu = nu

    def c(self, x, s, params):
        stack = Kernel.check_samples_and_return_stack(x, s)
        stack1 = cdist(x, s, metric='euclidean')
        print(stack, stack1)
        k = params[-1]**2 * np.exp(np.sum(-(1/(2*params[:-1]**2)) * (stack ** 2), axis=2))
        return k
