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

    return H_
