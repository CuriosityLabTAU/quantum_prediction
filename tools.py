from qutip import *
import numpy as np

def ndarray2Qobj(psi_vec, typ = 'vec', norm=False):
    '''
    Taking np.array vector of psi's (rho) coefficients and returning it in a Qobj.
    This we can work with QuTiP
    :param psi_vec: np.array - vector or 2d array.
    :return: psi: Qobj
    '''
    psi = Qobj(psi_vec)
    if norm:
        psi = psi.unit() # normalization
    n = psi_vec.shape
    nq = int(np.log2(n[0])) # num of qubits

    d1 = (2*np.ones(nq)).astype(int).tolist()
    if typ == 'vec': # psi - vector
        d2 = np.ones(nq).astype(int).tolist()
    elif typ == 'dm': # rho - density matrix
        d2 = d1[:]

    dimensions = [d1, d2] # for the Qobj
    psi.dims = dimensions
    return psi