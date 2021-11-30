from UQpy.surrogates.polynomial_chaos.MultiIndexSets import td_multiindex_set, tp_multiindex_set
from UQpy.surrogates.polynomial_chaos.ChaosPolynomials import ChaosPolynomial1d, ChaosPolynomialNd

def construct_arbitrary_basis(pce, midx_set):
    """
    Create polynomial basis for a given multiindex set.
    
    **Inputs**
    
    * **midx_set** (`ndarray`):
        n_polys x n_inputs ndarray with the multiindices of the PCE basis
        
    * **pce** (`PolyChaosExp object`):
        Polynomial chaos expansion for which the multiindex set will be 
        generated.
        
    **Output**
    
    * **poly_basis** (`list`)
        List with the basis polynomials (ChaosPolynomial1d or 
        ChaosPolynomialNd objects)
    """
    # populate polynomial basis
    if pce.n_inputs == 1:
        poly_basis = [ChaosPolynomial1d(pce.dist, idx) for idx in midx_set]
    else:
        poly_basis = [ChaosPolynomialNd(pce.dist, idx) for idx in midx_set]
    # update attributes of PolyChaosExp object
    pce.midx_set = midx_set
    pce.n_polys = len(midx_set)
    pce.poly_basis = poly_basis


def construct_tp_basis(pce, max_degree):
    """

    
    **Inputs**:
        
    * **max_degree** (`int`):

        
    * **pce** (`PolyChaosExp object`):
        Polynomial chaos expansion for which the multiindex set will be 
        generated.
        
    **Output**
    
    **poly_basis** (`list`)
        List with the basis polynomials (ChaosPolynomial1d or 
        ChaosPolynomialNd objects)
    """
    midx_set = tp_multiindex_set(pce.n_inputs, max_degree)
    construct_arbitrary_basis(pce, midx_set)
    
    
def construct_td_basis(pce, max_degree):
    """

    
    **Inputs**:
        
    * **max_degree** (`int`):

        
    * **pce** (`PolyChaosExp object`):
        Polynomial chaos expansion for which the multiindex set will be 
        generated.
        
    **Output**
    
    **poly_basis** (`list`)
        List with the basis polynomials (ChaosPolynomial1d or 
        ChaosPolynomialNd objects)
    """
    midx_set = td_multiindex_set(pce.n_inputs, max_degree) 
    construct_arbitrary_basis(pce, midx_set)
    