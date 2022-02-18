import copy

from UQpy.optimization.baseclass.OptimizationMethod import OptimizationMethod
from UQpy.dimension_reduction.grassmann_manifold.GrassmannPoint import GrassmannPoint
from UQpy.utilities.ValidationTypes import PositiveInteger
import numpy as np


class GradientDescent(OptimizationMethod):
    def __init__(
        self,
        acceleration: bool = False,
        error_tolerance: float = 1e-3,
        max_iterations: PositiveInteger = 1000,
    ):
        """

        :param acceleration:
        :param error_tolerance:
        :param max_iterations:
        """
        self.max_iterations = max_iterations
        self.error_tolerance = error_tolerance
        self.acceleration = acceleration

    def optimize(self, data_points: list[GrassmannPoint], distance):
        # Number of points.
        points_number = len(data_points)
        alpha = 0.5 #learning rate?

        mean_element = self.calculate_initial_point(data_points, distance, points_number)

        avg_gamma = np.zeros([np.shape(data_points[0].data)[0], np.shape(data_points[0].data)[1]])

        counter_iteration = 0

        l = 0
        avg = []
        _gamma = []
        if self.acceleration:
            avg_gamma = self.calculate_gradient(avg_gamma, data_points, mean_element, points_number)
            avg.append(avg_gamma)

        # Main loop
        while counter_iteration <= self.max_iterations:
            avg_gamma = self.calculate_gradient(avg_gamma, data_points, mean_element, points_number)

            test_0 = np.linalg.norm(avg_gamma, "fro")
            if test_0 < self.error_tolerance and counter_iteration == 0:
                break

            # Nesterov: Accelerated Gradient Descent
            if self.acceleration:
                avg.append(avg_gamma)
                l0 = l
                l1 = 0.5 * (1 + np.sqrt(1 + 4 * l * l))
                ls = (1 - l0) / l1
                step = (1 - ls) * avg[counter_iteration + 1] + ls * avg[counter_iteration]
                l = copy.copy(l1)
            else:
                step = alpha * avg_gamma

            x = self.calculate_function(mean_element, step)

            test_1 = np.linalg.norm(x.data - mean_element.data, "fro")

            if test_1 < self.error_tolerance:
                break

            mean_element = x

            counter_iteration += 1

        return mean_element

    def calculate_function(self, mean_element, step):
        from UQpy.dimension_reduction.grassmann_manifold.Grassmann import Grassmann
        return Grassmann.exp_map(tangent_points=[step], reference_point=mean_element)[0]

    def calculate_initial_point(self, data_points, distance, points_number):
        from UQpy.dimension_reduction.grassmann_manifold.Grassmann import Grassmann
        fmean = [
            Grassmann.frechet_variance(
                reference_point=data_points[i],
                manifold_points=data_points,
                distance=distance,
            )
            for i in range(points_number)
        ]

        index_0 = fmean.index(min(fmean))
        return GrassmannPoint(data_points[index_0].data)

    def calculate_gradient(self, avg_gamma, data_points, mean_element, points_number):
        from UQpy.dimension_reduction.grassmann_manifold.Grassmann import Grassmann
        _gamma = Grassmann.log_map(manifold_points=data_points,
                                   reference_point=mean_element)
        avg_gamma.fill(0)
        for i in range(points_number):
            avg_gamma += _gamma[i] / points_number
        return avg_gamma
