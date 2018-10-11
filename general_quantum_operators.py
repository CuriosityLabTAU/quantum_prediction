import numpy as np


def rmse(pred_, real_):
    return np.sqrt(np.square(np.subtract(pred_, real_)).mean())


def zero_H(n_qubits=2):
    dim_ = 2 ** n_qubits
    H_ = np.zeros([dim_, dim_])
    return H_


def param_H(h_):
    the_param = np.squeeze(h_)
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


def compose_H(full_h, all_q, n_qubits=4):
    # full_h = [h_a, h_b, h_mix]
    # all_q = [q1, q2]
    H_ = zero_H(n_qubits)

    Hmix_ = param_Hmix(full_h[2])
    mix = np.zeros([4, 4])
    mix[0, 0] = Hmix_[0, 0]
    mix[0, -1] = Hmix_[0, 1]
    mix[-1, 0] = Hmix_[1, 0]
    mix[-1, -1] = Hmix_[1, 1]

    so = np.kron(np.kron(mix, np.eye(2)), np.eye(2))

    o = reorganize_operator([0, 3, 1, 2], so)

    return H_


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
