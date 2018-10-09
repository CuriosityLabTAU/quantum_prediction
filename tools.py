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
    asked_qubits = np.array(asked_qubits)
    qubits = np.array([0, 1, 2, 3])
    not_asked = np.delete(qubits, asked_qubits - 1)
    ordr = np.append(asked_qubits - 1, not_asked)
    return not_asked+1, ordr


def create_Nqubit_state(N):
    '''Create index matrix for all possible combinations of N sized psi.'''
    m, n = 2**N, N
    a = np.zeros([m, n])
    A = np.ones(N)
    for i in range(0, m):
        for j in range(0,N):
            if i % 2**j == 0:
                A[-j-1] = flipa(A[-j-1]) # flipping the qubit value 0/1.1
        a[i, :] = A
    return a


def flipa(a):
    '''Flip 0/1'''
    if a == 0:
        a = 1
    elif a == 1:
        a = 0
    return a

def rho_mat(psi):
    '''Create generic rho matrix for a give Psi,
    what product there is in every place in the matrix.'''
    nr,nc = psi.shape
    rho = np.zeros([2,nc,nr,nr])
    for i in range(0,nr):
        for j in range(0, nr):
            rho[:,:,i,j] = psi[i],psi[j] # Putting the combinations |psi1><psi0| in the right place in the matrix.
    return rho

def reorganize_operator(qubits_order, operator_mat):
    '''Reorganizing matrix from specific order to the right order [1,2,3,4].'''
    N = len(qubits_order)
    psi = create_Nqubit_state(N)
    rho_scramb = rho_mat(psi)
    rho_org = np.copy(rho_scramb)
    nps, nq, nr, nc = rho_scramb.shape
    re_rho = np.zeros([nr, nc])
    # Scrambling rho according to given qubits order
    for i in range(0, nr):
        temp = np.zeros(nq)
        for j in range(0, nc):
            for k in range(0, nps):
                for l in range(0, nq):
                    # finding the index of the cell from the scrambled rho in the organized rho.
                    temp[qubits_order[l]] = rho_scramb[k, l, i, j]
                rho_scramb[k, :, i, j] = temp

    # Reorganizing Rho matrix with the real values (Not just indices).
    for i in range(0, nr):
        for j in range(0, nc):
            for k in range(0, nr):
                for l in range(0, nc):
                    if np.sum(rho_org[0, :, k, l] == rho_scramb[0, :, i, j]) == nq & np.sum(
                                    rho_org[1, :, k, l] == rho_scramb[1, :, i, j]) == nq:
                        re_rho[k, l] = operator_mat[i, j]
    return re_rho
