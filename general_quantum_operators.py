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
        # H_ = 1.0 / np.sqrt(1 + the_param * the_param) * np.matrix([[1, the_param], [the_param, -1]])
        H_ = 1.0 / np.sqrt(1 + the_param * the_param) * np.matrix([[1, the_param], [the_param, 1]])
    return H_


def param_Hmix_old(g_):
    H_ = (np.squeeze(g_) / np.sqrt(2)) * np.matrix([[1, 1], [1, -1]])
    return H_

def param_Hmix(g_, h_type):
    if h_type == 0:
        H_ = (np.squeeze(g_) / np.sqrt(2)) * np.matrix([[1, 1], [1, -1]])
    elif h_type == 1:
        H_ = (np.squeeze(g_) / np.sqrt(2)) * np.matrix([[1, 0], [0, -1]]) + np.matrix([[0, 1], [1, 0]])
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
    elif q_str == 'C': # conjunction
        P_ = np.dot(Projection(all_q[0], n_qubits), Projection(all_q[1], n_qubits))
    elif q_str == 'D': # disjunction
        P_ = Projection(all_q[0], n_qubits) + Projection(all_q[1], n_qubits) - \
             np.dot(Projection(all_q[0], n_qubits), Projection(all_q[1], n_qubits))
    return P_



def uniform_psi(n_qubits=2):
    # dim_ = 2 ** n_qubits
    # psi_ = np.ones([dim_,1]) / np.sqrt(dim_)

    plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
    psi_ = np.kron(plus, plus)
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


def get_general_p(full_h, all_q, all_P, psi_0, n_qubits=4, h_mix_type = 0):
    H_ = compose_H(full_h, all_q, n_qubits, h_mix_type)
    psi_dyn = get_psi(H_, psi_0)
    P_ = MultiProjection(all_P, all_q, n_qubits)
    psi_final = np.dot(P_, psi_dyn)
    p_ = norm_psi(psi_final)
    return p_

def compose_H(full_h, all_q, n_qubits=4, h_mix_type = 0):
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
        Hmix_ = param_Hmix(full_h[2], h_mix_type)
        mix = np.zeros([4, 4])

        # # h_mix_type=0 [h,0,0,h;0,0,0,0;0,0,0, 0;h,0,0,-h]
        # mix[0, 0] = Hmix_[0, 0]
        # mix[0, -1] = Hmix_[0, 1]
        # mix[-1, 0] = Hmix_[1, 0]
        # mix[-1, -1] = Hmix_[1, 1]

        # [1,0,0,0;0,1,h,0;0,h,1,0;0,0,0,1]
        mix[0, 0] = 1
        mix[1, 1] = 1
        mix[2, 2] = 1
        mix[3, 3] = 1
        mix[1, 2] = Hmix_[0, 0]
        mix[2, 1] = Hmix_[0, 0]

        for q in range(n_qubits - 2):
            mix = np.kron(mix, np.eye(2))

        current_order = all_q + list(set(np.arange(n_qubits)) - set(all_q))
        Hmix_ = reorganize_operator(current_order, mix)

    H_total = H_ + Hmix_

    return H_total


def flipa(a):
    '''Flip 0/1'''
    if a == 0:
        a = 1
    elif a == 1:
        a = 0
    return a

def flipa(a):
    '''Flip 0/1'''
    if a == 0:
        a = 1
    elif a == 1:
        a = 0
    return a

def create_Nqubit_state(N):
    '''Create index matrix for all possible combinations of N sized psi.'''
    m, n = 2**N, N
    a = []
    A = np.ones(N, dtype = 'int')
    for i in range(0, m):
        for j in range(0,N):
            if i % 2**j == 0:
                A[-j-1] = flipa(A[-j-1]) # flipping the qubit value 0/1.1
        a.append(''.join(str(e) for e in list(A)))
    return a


def rho_mat(psi):
    '''Create generic rho matrix for a give Psi,
    what product there is in every place in the matrix.'''
    nr = psi.__len__()
    rho = []
    rho = list(rho)
    for i in range(0,nr):
        row = []
        for j in range(0, nr):
            row.append(psi[i]+psi[j])
        rho.append(row)
    return rho

def reorganize_operator(qubits_order, operator_mat):
    '''Reorganizing matrix from specific order to the right order [1,2,3,4].
    e.g: reorganize_operator([1,3,4,2], O) '''
    N = len(qubits_order)
    psi = create_Nqubit_state(N)
    rho_scramb = rho_mat(psi)
    rho_org = np.copy(rho_scramb)
    nq = qubits_order.__len__()
    nr = nq ** 2
    re_rho = np.zeros([nr, nr])
    nps = [0,1]
    # Scrambling rho according to given qubits order
    for i in range(0, nr):
        for j in range(0, nr):
            t = []
            for k in nps:
                temp = np.zeros(nq, dtype='int')
                for l in range(0,nq):
                    # finding the index of the cell from the scrambled rho in the organized rho.
                    temp[qubits_order[l]] = rho_scramb[i][j][l+nq*k]
                t.append(''.join(str(e) for e in list(temp)))
            # print(t)
            rho_scramb[i][j] =''.join(t)

    # Reorganizing Rho matrix with the real values (Not just indices).
    for i in range(0, nr):
        for j in range(0, nr):
            for k in range(0, nr):
                for l in range(0, nr):
                    if rho_org[k][l] == rho_scramb[i][j]:
                        re_rho[k, l] = operator_mat[i, j]
    return re_rho


def grandH_from_x_old(x_):
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


def find_where2multiple_h_param(num_of_qubits = 4, qubits = [1, 3], combo = [1, 0]):
    '''
    Find where the given qubits are equal to the combo. (qubits [0,1] == |1,0,x,x><x,x,x,x| or qubits [0,1] == |x,x,x,x><1,0,x,x| )
    For now works only on 2 qubits.
    :param num_of_qubits: How many qubits in total in the state. (e.g. 4 qubits).
    :param qubits: Which qubits we are working on. (e.g. the first and the third would be: [0,2])
    :param combo: What is the state of each qubit.
    :return: matrix that has True where the values of the two qubits together correspond to the combo.
    '''
    N = num_of_qubits
    psi = create_Nqubit_state(N)
    rho_scramb = rho_mat(psi)
    rho_org = np.copy(rho_scramb)
    h_multipication_place = np.zeros(rho_org.shape)
    nq = N # number of qubits
    nr = nq ** 2 # number of rows in rho
      # current_qubits
    for i in range(0, nr):
        for j in range(0, nr):
            # if (int(rho_org[i, j][cq[0]]) == c[0] and int(rho_org[i, j][cq[1]]) == c[1]) or \
            #         (int(rho_org[i, j][nq + cq[0]]) == c[0] and int(rho_org[i, j][nq + cq[1]]) == c[1]):
            if (int(rho_org[i, j][qubits[0]]) == combo[0] and int(rho_org[i, j][qubits[1]]) == combo[1]):
                h_multipication_place[i, j] = 1
            elif (int(rho_org[i, j][nq + qubits[0]]) == combo[0] and int(rho_org[i, j][nq + qubits[1]]) == combo[1]):
                h_multipication_place[i, j] = 1
    return h_multipication_place == 1


def create_H_from_x(x):
    # todo: TORR - normalization
    param = np.squeeze(x) / np.sqrt(2)

    H_x = param * np.matrix([[1, 0, 0, 1],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [1, 0, 0, -1]])

    return H_x


def grandH_from_x(x_, qubits = [1, 3]):
    H_ = np.kron(np.kron(np.kron(param_H(x_[0]), np.eye(2)), np.eye(2)), np.eye(2))
    H_ += np.kron(np.kron(np.kron(np.eye(2), param_H(x_[1])), np.eye(2)), np.eye(2))
    H_ += np.kron(np.kron(np.kron(np.eye(2), np.eye(2)), param_H(x_[2])), np.eye(2))
    H_ += np.kron(np.kron(np.kron(np.eye(2), np.eye(2)), np.eye(2)), param_H(x_[3]))

    ij = [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]] # qubits pairs
    for i, r in enumerate(range(4,10)):
        H_ij = np.kron(np.kron(create_H_from_x(x_[r]), np.eye(2)), np.eye(2))
        current_order = ij[i] + list(set(np.arange(4)) - set(ij[i]))
        H_ij = reorganize_operator(current_order, H_ij)
        H_ += H_ij

    # indices_00 = find_where2multiple_h_param(N = 4, c = [0,0], cq = qubits)
    # indices_01 = find_where2multiple_h_param(N = 4, c = [0,1], cq = qubits)
    # indices_10 = find_where2multiple_h_param(N = 4, c = [1,0], cq = qubits)
    # indices_11 = find_where2multiple_h_param(N = 4, c = [1,1], cq = qubits)
    #
    # if len(x_) < 5:
    #     h = (np.squeeze(x_[4]) / np.sqrt(2))
    #     h_00 = h.copy()
    #     h_01 = h.copy()
    #     h_10 = h.copy()
    #     h_11 = h.copy()
    # else:
    #     h_00 = (np.squeeze(x_[4]) / np.sqrt(2))
    #     h_01 = (np.squeeze(x_[5]) / np.sqrt(2))
    #     h_10 = (np.squeeze(x_[6]) / np.sqrt(2))
    #     h_11 = (np.squeeze(x_[7]) / np.sqrt(2))
    #
    # H_[indices_00] = h_00 * H_
    # H_[indices_01] = h_01 * H_
    # H_[indices_10] = h_10 * H_
    # H_[indices_11] = -1 * h_11 * H_

    return H_

