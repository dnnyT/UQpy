import numpy as np
import scipy.special as special
from beartype import beartype

from UQpy.distributions import Uniform
from UQpy.surrogates.polynomial_chaos.polynomials.baseclass.Polynomials import Polynomials
from scipy.special import eval_legendre


class Legendre(Polynomials):

    @beartype
    def __init__(self, degree: int, distributions):
        """
        Class of univariate polynomials appropriate for data generated from a uniform distribution.

        :param degree: Maximum degree of the polynomials.
        :param distributions: Distribution object of the generated samples.
        """
        super().__init__(distributions, degree)
        self.degree = degree
        self.pdf = self.distributions

    def evaluate(self, x):
        """
        Calculates the normalized Legendre polynomials evaluated at sample points.

        :param x: :class:`numpy.ndarray` containing the samples.
        :return: Α list of :class:`numpy.ndarray` with the design matrix and the
                    normalized polynomials.
        """
        x = np.array(x).flatten()

        # normalize data
        x_normed = Polynomials.standardize_uniform(x, self.distributions)

        # evaluate standard Legendre polynomial, i.e. orthogonal in [-1,1] with
        # PDF = 1 (NOT 1/2!!!)
        l = eval_legendre(self.degree, x_normed)

        # normalization constant
        st_lege_norm = np.sqrt(2 / (2 * self.degree + 1))

        # multiply by sqrt(2) to take into account the pdf 1/2
        l = np.sqrt(2) * l / st_lege_norm

        return l
    
    @staticmethod
    def legendre_triple_product (k,l,m):
    
        normk=1/((2*k)+1)
        norml=1/((2*l)+1)
        normm=1/((2*m)+1)
        norm=np.sqrt(normm/(normk*norml))


        return norm*(2*m+1)*Legendre.wigner_3j_PCE(k,l,m)**2
