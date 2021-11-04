import logging

import numpy as np
import scipy.stats as stats
from UQpy.utilities.strata.StratificationCriterion import StratificationCriterion
from UQpy.utilities.ValidationTypes import RandomStateType
from UQpy.utilities.strata.baseclass.Strata import Strata
from UQpy.sampling.SimplexSampling import SimplexSampling


class Delaunay(Strata):
    def calculate_strata_metrics(self, **kwargs):
        pass

    def __init__(
        self,
        seeds: np.ndarray = None,
        seeds_number: int = None,
        dimension: np.ndarray = None,
        stratification_criterion: StratificationCriterion = StratificationCriterion.RANDOM
    ):
        """
        Define a geometric decomposition of the n-dimensional unit hypercube into disjoint and space-filling
         Delaunay strata of n-dimensional simplexes. :class:`.Delaunay` is a child class of the :class:`.Strata` class.

        :param seeds: An array of dimension `N x n` specifying the seeds of all strata. The seeds of the strata are the
         coordinates of the vertices of the Delaunay cells. The user must provide `seeds` or `seeds_number` and
         `dimension`. Note that, if `seeds` does not include all corners of the unit hypercube, they are added.
        :param seeds_number: The number of seeds to randomly generate. Seeds are generated by random sampling on the
         unit hypercube. In addition, the class also adds seed points at all corners of the unit hypercube.
         The user must provide `seeds` or `seeds_number` and `dimension`
        :param dimension: The dimension of the unit hypercube in which to generate random seeds. Used only if
         `seeds_number` is provided. The user must provide `seeds` or `seeds_number` and `dimension`
        """
        super().__init__(seeds=seeds, stratification_criterion=stratification_criterion)

        self.stratification_criterion = stratification_criterion
        self.seeds_number = seeds_number
        self.dimension = dimension
        self.delaunay = None
        """Defines a Delaunay decomposition of the set of seed points and all corner points."""
        self.centroids = []
        """A list of the vertices for each Voronoi stratum on the unit hypercube."""
        self.logger = logging.getLogger(__name__)

        if self.seeds is not None:
            if self.seeds_number is not None or self.dimension is not None:
                print("UQpy: Ignoring 'seeds_number' and 'dimension' attributes because 'seeds' are provided")
            self.seeds_number, self.dimension = self.seeds.shape[0], self.seeds.shape[1]

    def stratify(self, random_state):
        import itertools
        from scipy.spatial import Delaunay

        self.logger.info("UQpy: Creating Delaunay stratification ...")

        initial_seeds = self.seeds
        if self.seeds is None:
            initial_seeds = stats.uniform.rvs(
                size=[self.seeds_number, self.dimension], random_state=random_state
            )

        # Modify seeds to include corner points of (0,1) space
        corners = list(
            itertools.product(*zip([0] * self.dimension, [1] * self.dimension))
        )
        initial_seeds = np.vstack([initial_seeds, corners])
        initial_seeds = np.unique([tuple(row) for row in initial_seeds], axis=0)

        self.delaunay = Delaunay(initial_seeds)
        self.centroids = np.zeros([0, self.dimension])
        self.volume = np.zeros([0])
        count = 0
        for sim in self.delaunay.simplices:  # extract simplices from Delaunay triangulation
            # pylint: disable=E1136
            cent, vol = self.compute_delaunay_centroid_volume(self.delaunay.points[sim])
            self.centroids = np.vstack([self.centroids, cent])
            self.volume = np.hstack([self.volume, np.array([vol])])
            count = count + 1

        self.logger.info("UQpy: Delaunay stratification created.")

    @staticmethod
    def compute_delaunay_centroid_volume(vertices):
        """
        This function computes the centroid and volume of a Delaunay simplex from its vertices.

        :param vertices: Coordinates of the vertices of the simplex.
        :return: Centroid and Volume of the Delaunay simplex.
        """
        from scipy.spatial import ConvexHull

        ch = ConvexHull(vertices)
        volume = ch.volume
        centroid = np.mean(vertices, axis=0)

        return centroid, volume

    def sample_strata(self, samples_per_stratum_number, random_state):
        samples_in_strata, weights = [], []
        count = 0
        for (
            simplex
        ) in self.delaunay.simplices:  # extract simplices from Delaunay triangulation
            samples_temp = SimplexSampling(
                nodes=self.delaunay.points[simplex],
                samples_number=int(samples_per_stratum_number[count]),
                random_state=random_state,
            )
            samples_in_strata.append(samples_temp.samples)
            self.extend_weights(samples_per_stratum_number, count, weights)
            count = count + 1
        return samples_in_strata, weights
