from UQpy.optimization.baseclass.OptimizationMethod import OptimizationMethod
from UQpy.utilities.ValidationTypes import PositiveInteger
import numpy as np


class StochasticGradientDescent(OptimizationMethod):
    def __init__(self,
                 error_tolerance: float = 1e-3,
                 max_iterations: PositiveInteger = 1000,):
        """

        :param error_tolerance:
        :param max_iterations:
        """
        self.max_iterations = max_iterations
        self.error_tolerance = error_tolerance

    def optimize(self, data_points, distance):
        from UQpy.dimension_reduction.grassmann_manifold.Grassmann import Grassmann
        n_mat = len(data_points)

        mean_element = self.calculate_initial_point(data_points, distance, n_mat)
        counter_iteration = 0
        k = 1

        while counter_iteration < self.max_iterations:
            indices = np.arange(n_mat)
            np.random.shuffle(indices)

            melem = mean_element
            for i in range(len(indices)):
                alpha = 0.5 / k
                idx = indices[i]
                _gamma = self.calculate_gradient(data_points, idx, mean_element)

                step = 2 * alpha * _gamma[0]

                X = self.calculate_function(mean_element, step)

                mean_element = X[0]

                k += 1

            test_1 = np.linalg.norm(mean_element - melem, "fro")
            if test_1 < self.error_tolerance:
                break

            counter_iteration += 1

        return mean_element

    def calculate_function(self, mean_element, step):
        from UQpy.dimension_reduction.grassmann_manifold.Grassmann import Grassmann
        return Grassmann.exp_map(tangent_points=[step], reference_point=np.asarray(mean_element))

    def calculate_gradient(self, data_points, idx, mean_element):
        from UQpy.dimension_reduction.grassmann_manifold.Grassmann import Grassmann
        _gamma = Grassmann.log_map(manifold_points=[data_points[idx]],
                                   reference_point=np.asarray(mean_element), )
        return _gamma

    def calculate_initial_point(self, data_points, distance, points_number):
        from UQpy.dimension_reduction.grassmann_manifold.Grassmann import Grassmann
        # fmean = [Grassmann.frechet_variance(data_points[i]) for i in range(points_number)]
        fmean = [
            Grassmann.frechet_variance(
                reference_point=data_points[i],
                manifold_points=data_points,
                distance=distance,
            )
            for i in range(points_number)
        ]
        index_0 = fmean.index(min(fmean))
        mean_element = data_points[index_0].tolist()
        return mean_element

    def optimize(self, function, initial_guess, args=()):
        n_mat = len(initial_guess)


