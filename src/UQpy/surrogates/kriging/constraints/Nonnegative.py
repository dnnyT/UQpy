from UQpy.surrogates.kriging.constraints.baseclass.Constraints import *


class Nonnegative(Constraints):
    def __init__(self, candidate_points, observed_error=0.01, z_value=2):
        self.candidate_points = candidate_points
        self.observed_error = observed_error
        self.z_value = z_value

    def constraints(self, x_train, y_train, predict_function):
        cons = []
        # for i in range(y_train.shape[0]):
        #     cons.append({'type': 'ineq',
        #                  'fun': lambda theta: self.observed_error -
        #                  abs(predict_function(x_train[i, :], correlation_model_parameters=theta) -
        #                      y_train[i])})

        for j in range(self.candidate_points.shape[0]):

            cons.append({'type': 'ineq', 'fun': self.constraints_candidate, 'args': (predict_function,
                                                                                     self.candidate_points[j, :],
                                                                                     self.z_value)})

        return cons

    @staticmethod
    def constraints_candidate(theta_, pred, cand_points, z_):
        tmp_predict, tmp_error = pred(cand_points, True, correlation_model_parameters=theta_)
        return tmp_predict - z_ * tmp_error
