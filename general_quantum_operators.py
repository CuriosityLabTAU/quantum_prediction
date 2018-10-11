import numpy as np
from scipy.linalg import expm

def rmse(pred_, real_):
    return np.sqrt(np.square(np.subtract(pred_, real_)).mean())


def zero_H(n_qubits=2):
    dim_ = 2 ** n_qubits
    H_ = np.zeros([dim_, dim_])
    return H_


def param_H(h_):
    the_param = np.squeeze(h_)
    if h_ is None:
        H_ = np.eye(2)
    else:
        H_ = 1.0 / np.sqrt(1 + the_param * the_param) * np.matrix([[1, the_param], [the_param, -1]])
    return H_


def param_Hmix(g_):
    H_ = (np.squeeze(g_) / np.sqrt(2)) * np.matrix([[1, 1], [1, -1]])
    return H_


def U_from_H(H_):
    U_ = expm(-1j * np.pi / 2.0 * H_)
    return U_


def Projection(q, n_qubits=2):
    dim_ = 2 ** n_qubits
    d_ = np.zeros([dim_])
    for i in range(dim_):
        i_repr = np.binary_repr(i, width=n_qubits)
        if i_repr[q] == '1':
            d_[i] = 1
    P_ = np.diag(d_)
    return P_


def MultiProjection(q_str, all_q, n_qubits=2):
    P_ = None
    if q_str == '0':
        P_ = Projection(all_q[0], n_qubits)
    elif q_str == '1':
        P_ = Projection(all_q[1], n_qubits)
    elif q_str == 'C':
        P_ = np.dot(Projection(all_q[0], n_qubits), Projection(all_q[1], n_qubits))
    elif q_str == 'D':
        P_ = Projection(all_q[0], n_qubits) + Projection(all_q[1], n_qubits) - \
             np.dot(Projection(all_q[0], n_qubits), Projection(all_q[1], n_qubits))
    return P_



def uniform_psi(n_qubits=2):
    dim_ = 2 ** n_qubits
    psi_ = np.ones([dim_,1]) / np.sqrt(dim_)
    return psi_


def norm_psi(psi):
    p_ = np.dot(np.conjugate(np.transpose(psi)), psi).real
    return p_


def get_psi(H_, psi_0):
    psi_ = np.dot(U_from_H(H_), psi_0)
    return psi_


def get_prob_single_q(psi_0, H_, q, n_qubits=2):
    psi_ = get_psi(psi_0, H_)
    proj_psi = np.dot(Projection(q, n_qubits), psi_)
    p_ = norm_psi(proj_psi).real
    return p_


def get_general_p(full_h, all_q, all_P, psi_0, n_qubits=4):
    H_ = compose_H(full_h, all_q, n_qubits)
    psi_dyn = get_psi(H_, psi_0)
    P_ = MultiProjection(all_P, all_q, n_qubits)
    psi_final = np.dot(P_, psi_dyn)
    p_ = norm_psi(psi_final)
    return p_

def compose_H(full_h, all_q, n_qubits=4):
    # full_h = [h_a, h_b, h_mix]
    # all_q = [q1, q2]
    H_ = zero_H(n_qubits)

    for q in range(n_qubits):
        if q == 0:
            if q == all_q[0]:
                H_ = param_H(full_h[0])
            elif q ==  all_q[1]:
                H_ = param_H(full_h[1])
            else:
                H_ = np.eye(2)
        else:
            if q == all_q[0]:
                H_ = np.kron(H_, param_H(full_h[0]))
            elif q == all_q[1]:
                H_ = np.kron(H_, param_H(full_h[1]))
            else:
                H_ = np.kron(H_, np.eye(2))

    if full_h[2] == None:
        Hmix_ = np.zeros([2 ** n_qubits, 2 ** n_qubits])
    else:
        Hmix_ = param_Hmix(full_h[2])
        mix = np.zeros([4, 4])
        mix[0, 0] = Hmix_[0, 0]
        mix[0, -1] = Hmix_[0, 1]
        mix[-1, 0] = Hmix_[1, 0]
        mix[-1, -1] = Hmix_[1, 1]

        for q in range(n_qubits - 2):
            mix = np.kron(mix, np.eye(2))

        current_order = all_q + list(set(np.arange(n_qubits)) - set(all_q))
        Hmix_ = reorganize_operator(current_order, mix)

    H_total = H_ + Hmix_

    return H_total


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


def grandH_from_x(x_):
    # TODO: add all off-diagonal
    H_ = np.kron(np.kron(np.kron(param_H(x_[0]), np.eye(2)), np.eye(2)), np.eye(2))
    H_ += np.kron(np.kron(np.kron(np.eye(2), param_H(x_[1])), np.eye(2)), np.eye(2))
    H_ += np.kron(np.kron(np.kron(np.eye(2), np.eye(2)), param_H(x_[2])), np.eye(2))
    H_ += np.kron(np.kron(np.kron(np.eye(2), np.eye(2)), np.eye(2)), param_H(x_[3]))

    # total
    mix_total = param_Hmix(x_[4])
    H_[0, 0] += mix_total[0, 0]
    H_[0, -1] += mix_total[0, 1]
    H_[-1, 0] += mix_total[1, 0]
    H_[-1, -1] += mix_total[1, 1]

    return H_

