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


def unasked_qubits(asked_qubits):
    '''
    :param asked_qubits: Which qubits were asked in the question
    :return: not_asked - Which qubits weren't asked in the question.
             ordr: order of the qubits in the question.
    '''
    qubits = np.array([0, 1, 2, 3])
    not_asked = np.delete(qubits, asked_qubits - 1)
    ordr = np.append(asked_qubits - 1, not_asked)
    return not_asked+1, ordr