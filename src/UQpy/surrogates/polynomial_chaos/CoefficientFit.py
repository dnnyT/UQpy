import numpy as np


def fit_lstsq(pce, sample_in, sample_out):
    """
    Computes the PCE coefficients with the numpy.linalg.lstsq solver.
    `sample_in` are realizations of the input parameters and `sample_out` the 
    corresponding observations regarding the model output.
    
    **Inputs**
    
    * **pce** (`PCE object`)
        Polynomial Chaos Expansion the coefficients of which will be evaluated
        by means of least squares regression.
        
    * **sample_in** (`ndarray`)
        Realizations of the input random parameters
        
    * **sample_out** (`ndarray`)
        Evaluations of the original model on sample_in
    """
    sample_in = np.array(sample_in)
    sample_out = np.array(sample_out)
    # update experimental design
    pce.exp_design_in = sample_in
    pce.exp_design_out = sample_out
    # compute and update design matrix
    D = pce.eval_basis(sample_in)
    pce.design_matrix = D
    # fit coefficients
    c, res, rank, sing = np.linalg.lstsq(D, sample_out, rcond=None)
    if c.ndim == 1:
        c = c.reshape(-1, 1)
    # update n_outputs and coefficients
    pce.n_outputs = np.shape(c)[1]
    pce.coefficients = c
    

def fit_lasso(pce, sample_in, sample_out, learning_rate=0.01, iterations=1000, 
              penalty=1):
    """
    Compute the PCE coefficients with the LASSO method.
    `sample_in` are realizations of the input parameters and `sample_out` the 
    corresponding observations regarding the model output.
    
    **Inputs**
    
    * **pce** (`PCE object`)
        Polynomial Chaos Expansion the coefficients of which will be evaluated
        by means of LASSO regression.
        
    * **sample_in** (`ndarray`)
        Realizations of the input random parameters
        
    * **sample_out** (`ndarray`)
        Evaluations of the original model on sample_in
    """
    sample_in = np.array(sample_in)
    sample_out = np.array(sample_out)
    # update experimental design
    pce.exp_design_in = sample_in
    pce.exp_design_out = sample_out
    # compute and update design matrix
    D = pce.eval_basis(sample_in)
    pce.design_matrix = D
    
    m, n = D.shape
    # in some 1D output problems the y array in python might be (n,) or 
    # (n,1)
    if sample_out.ndim == 1 or sample_out.shape[1] == 1:
        sample_out = sample_out.reshape(-1, 1)
        w = np.zeros(n).reshape(-1, 1)
        dw = np.zeros(n).reshape(-1, 1)
        b = 0
        for _ in range(iterations):
            predictions = (D.dot(w) + b)

            for i in range(n):
                if w[i] > 0:
                    dw[i] = (-(2 * (D.T[i, :]).dot(sample_out - predictions)) \
                             + penalty) / m
                else:
                    dw[i] = (-(2 * (D.T[i, :]).dot(sample_out - predictions)) \
                             - penalty) / m

            db = - 2 * np.sum(sample_out - predictions) / m

            w = w - learning_rate * dw
            b = b - learning_rate * db
    else:
        n_out_dim = sample_out.shape[1]
        w = np.zeros((n, n_out_dim))
        b = np.zeros(n_out_dim).reshape(1, -1)
        for _ in range(iterations):
            predictions = (D.dot(w) + b)

            dw = (-(2 * D.T.dot(sample_out - predictions)) - penalty) / m
            db = - 2 * np.sum((sample_out - predictions), 
                              axis=0).reshape(1, -1) / m

            w = w - learning_rate * dw
            b = b - learning_rate * db
    pce.bias = b
    pce.coefficients = w
    pce.n_outputs = np.shape(w)[1]
    

def fit_ridge(pce, sample_in, sample_out, learning_rate=0.01, iterations=1000,
             penalty=1):
    """
    Compute the PCE coefficients with ridge regression.
    `sample_in` are realizations of the input parameters and `sample_out` the 
    corresponding observations regarding the model output.
    
    **Inputs**
    
    * **pce** (`PCE object`)
        Polynomial Chaos Expansion the coefficients of which will be evaluated
        by means of ridge regression.
        
    * **sample_in** (`ndarray`)
        Realizations of the input random parameters
        
    * **sample_out** (`ndarray`)
        Evaluations of the original model on sample_in

    """
    sample_in = np.array(sample_in)
    sample_out = np.array(sample_out)
    # update experimental design
    pce.exp_design_in = sample_in
    pce.exp_design_out = sample_out
    # compute and update design matrix
    D = pce.eval_basis(sample_in)
    pce.design_matrix = D
    
    m, n = D.shape
    # in some 1D output problems the y array in python might be (n,) or 
    # (n,1)
    if sample_out.ndim == 1 or sample_out.shape[1] == 1:
        sample_out = sample_out.reshape(-1, 1)
        w = np.zeros(n).reshape(-1, 1)
        b = 0
        for _ in range(iterations):
            predictions = (D.dot(w) + b).reshape(-1, 1)

            dw = (-(2 * D.T.dot(sample_out - predictions)) + \
                  (2 * penalty * w)) / m
            db = - 2 * np.sum(sample_out - predictions) / m

            w = w - learning_rate * dw
            b = b - learning_rate * db
    else:
        n_out_dim = sample_out.shape[1]
        w = np.zeros((n, n_out_dim))
        b = np.zeros(n_out_dim).reshape(1, -1)
        for _ in range(iterations):
            predictions = (D.dot(w) + b)

            dw = (-(2 * D.T.dot(sample_out - predictions)) + \
                  (2 * penalty * w)) / m
            db = - 2 * np.sum((sample_out - predictions), 
                              axis=0).reshape(1, -1) / m

            w = w - learning_rate * dw
            b = b - learning_rate * db
    pce.bias = b
    pce.coefficients = w
    pce.n_outputs = np.shape(w)[1]