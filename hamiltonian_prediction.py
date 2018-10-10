from scipy.optimize import minimize
import numpy as np
from scipy.linalg import expm
import pandas as pd


def zero_H(n_qubits=2):
    dim_ = 2 ** n_qubits
    H_ = np.zeros([dim_, dim_])
    return H_


def param_H(h_):
    H_ = np.matrix([[1, h_], [h_, -1]])
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


def compose_H(H1_, H2_, Hmix_, n_qubits=2):
    dim_ = 2 ** n_qubits
    # TODO: add off diagonal
    H_ = np.kron(H1_, np.eye(2)) + np.kron(np.eye(2), H2_) + np.kron(np.array([[0,1], [1,0]]), Hmix_)
    #
    # H_ = np.zeros([dim_, dim_])
    # H_[:H1_.shape[0], :H1_.shape[1]] = H1_
    # H_[H1_.shape[0]:, H1_.shape[1]:] = H2_
    # H_[H1_.shape[0]:, :H1_.shape[1]] = Hmix_
    # H_[:H1_.shape[0], H1_.shape[1]:] = np.transpose(Hmix_)

    return H_


def uniform_psi(n_qubits=2):
    dim_ = 2 ** n_qubits
    psi_ = np.ones([dim_,1]) / np.sqrt(dim_)
    return psi_


def norm_psi(psi):
    p_ = np.dot(np.conjugate(np.transpose(psi)), psi).real
    return p_


def get_psi(psi_0, H_):
    psi_ = np.dot(U_from_H(H_), psi_0)
    return psi_


def get_prob(psi_0, H_, q, n_qubits=2):
    psi_ = get_psi(psi_0, H_)
    proj_psi = np.dot(Projection(q, n_qubits), psi_)
    p_ = norm_psi(proj_psi).real
    return p_


def prob_from_h(h_, psi_0, q, n_qubits=2):
    H_ = zero_H(n_qubits)
    if q == 0:
        H_ = compose_H(param_H(h_), zero_H(1), zero_H(1))
    elif q == 1:
        H_ = compose_H(zero_H(1), param_H(h_), zero_H(1))
    p_ = get_prob(psi_0, H_, q, n_qubits)
    return p_


def fun_to_minimize(h_, real_p_, psi_0, q, n_qubits=2):
    p_ = prob_from_h(h_, psi_0, q, n_qubits=2)
    err_ = (p_ - real_p_) ** 2
    return err_


def get_prob_mix(psi_0, H_, n_qubits=2):
    psi_ = np.dot(Projection(0, n_qubits), np.dot(Projection(1, n_qubits), np.dot(U_from_H(H_), psi_0)))
    p_ = norm_psi(psi_).real
    return p_


def prob_from_hmix(hmix_, h_a, h_b, psi_0, n_qubits=2):
    H_ = compose_H(param_H(h_a), param_H(h_b), param_Hmix(hmix_))
    p_ = get_prob_mix(psi_0, H_, n_qubits)
    return p_


def fun_to_minimize_mix(h_, real_p_, h_a, h_b, psi_0, n_qubits=2):
    p_ = prob_from_hmix(h_, h_a, h_b, psi_0, n_qubits=2)
    err_ = (p_ - real_p_) ** 2
    return err_


def grandH_from_x(x_):
    # TODO: add all off-diagonal
    H_ = np.kron(np.kron(np.kron(param_H(x_[0]), np.eye(2)), np.eye(2)), np.eye(2))
    H_ += np.kron(np.kron(np.kron(np.eye(2), param_H(x_[1])), np.eye(2)), np.eye(2))
    H_ += np.kron(np.kron(np.kron(np.eye(2), np.eye(2)), param_H(x_[2])), np.eye(2))
    H_ += np.kron(np.kron(np.kron(np.eye(2), np.eye(2)), np.eye(2)), param_H(x_[3]))
    return H_

def get_p_from_grandH(grand_U, data):
    sub_U_psi = np.dot(grand_U, data['01']['psi'])

    # if A-D
    P_Ha_2qubits = np.dot(Projection(0), U_from_H(compose_H(param_H(data[0]['h_a']), zero_H(1), zero_H(1))))
    P_Ha_4qubits = np.kron(np.kron(P_Ha_2qubits, np.eye(2)), np.eye(2))
    p_a_calc = norm_psi(np.dot(P_Ha_4qubits, sub_U_psi))

    P_Hb_2qubits = np.dot(Projection(1), U_from_H(compose_H(zero_H(1), param_H(data[1]['h_b']), zero_H(1))))
    P_Hb_4qubits = np.kron(np.kron(np.eye(2), np.eye(2)), P_Hb_2qubits)
    p_b_calc = norm_psi(np.dot(P_Hb_4qubits, sub_U_psi))
    return p_a_calc, p_b_calc


def fun_to_minimize_grandH(x_, all_data):
    grand_U = U_from_H(grandH_from_x(x_))

    err_ = 0.0
    for data in all_data.values():
        p_a = data['2']['p_a']
        p_b = data['2']['p_b']

        p_a_calc, p_b_calc = get_p_from_grandH(grand_U, data)

        err_ += (p_a - p_a_calc) ** 2
        err_ += (p_b - p_b_calc) ** 2

    err_ = np.sqrt(err_ / (2*len(all_data.values())))
    return err_


def main():
    df = pd.read_csv('data/new_dataframe.csv', index_col=0)

    # go over all individuals
    user_same_q_list = {}
    for qn in df[(df.qn == 2.)].pos.unique():
        user_same_q_temp = df[(df.pos == 2.) & (df.qn == qn)]['userID'] #
        # user_same_q_list.append(user_same_q_temp)
        user_same_q_list[qn] = user_same_q_temp.unique()

    all_data = {}
    for ui, u_id in enumerate(user_same_q_list[2]):
        # select only from one group that has the same third question
        print('calculating states for user #:',  ui)
        # go over questions 1 & 2
        psi_0 = uniform_psi()
        sub_data = {}
        for p_id in range(2):

            d = df[(df['userID'] == u_id) & (df['pos'] == p_id)]
            p = {
                'A': d['p1'].values,
                'B': d['p2'].values,
                'A_B': d['p12'].values
            }
            # find h_a
            res_temp = minimize(fun_to_minimize, 0.0, args=(p['A'], psi_0, 0),
                                method='SLSQP', bounds=None, options={'disp': False})
            # print(res_temp.fun)
            h_a = res_temp.x
            p_a = prob_from_h(h_a, psi_0, 0)
            # print(p_a, p['A'])

            # find h_b
            res_temp = minimize(fun_to_minimize, 0.0, args=(p['B'], psi_0, 1),
                                method='SLSQP', bounds=None, options={'disp': False})
            # print(res_temp.fun)
            h_b = res_temp.x  #find h_mix
            p_b = prob_from_h(h_b, psi_0, 1)
            # print(p_b, p['B'])

            # find h_ab
            res_temp = minimize(fun_to_minimize_mix, 0.0, args=(p['A_B'], h_a, h_b, psi_0, 1),
                                method='SLSQP', bounds=None, options={'disp': False})
            # print(res_temp.fun)
            h_ab = res_temp.x
            p_ab = prob_from_hmix(h_ab, h_a, h_b, psi_0)
            # print(p_ab, p['A_B'])

            total_H = compose_H(param_H(h_a), param_H(h_b), param_Hmix(h_ab))
            psi_final = get_psi(psi_0, total_H)
            sub_data[p_id] = {
                'h_a': h_a,
                'h_b': h_b,
                'h_ab': h_ab,
                'psi': psi_final
            }

        sub_data['01'] = {
            'psi': np.kron(sub_data[0]['psi'], sub_data[1]['psi'])
        }

        d = df[(df['userID'] == u_id) & (df['pos'] == 2)]
        p = {
            'A': d['p1'].values,
            'B': d['p2'].values,
            'A_B': d['p12'].values
        }
        sub_data['2'] = {
            'p_a': p['A'],
            'p_b': p['B']
        }

        all_data[u_id] = sub_data

    print(len(all_data.keys()))

    # to find U_3
    n_user = len(all_data.keys())
    n_train = int(0.9 * n_user)
    user_rand_order = np.random.permutation(np.arange(n_user))
    user_rand_order = np.array(all_data.keys())[user_rand_order]
    user_train = user_rand_order[:n_train].tolist()
    user_test = user_rand_order[n_train:].tolist()

    train_data = {}
    for i_t, i_train in enumerate(user_train):
        train_data[i_t] = all_data[i_train]

    test_data = {}
    for i_t, i_test in enumerate(user_test):
        test_data[i_t] = all_data[i_test]


    res_temp = minimize(fun_to_minimize_grandH, np.zeros([4]), args=(train_data),
                        method='SLSQP', bounds=None, options={'disp': False})
    print('train error: ', res_temp.fun)
    print(res_temp.x)

    # given U, calculate p_a, p_b for all 100%
    grand_U = U_from_H(grandH_from_x(res_temp.x))
    grand_I = np.eye(16)

    test_err_U = 0.0
    test_err_I = 0.0
    for u_id, data in all_data.items():
        p_a_calc, p_b_calc = get_p_from_grandH(grand_U, data)
        data['2']['p_a_calc'] = p_a_calc
        data['2']['p_b_calc'] = p_b_calc

        p_a_calc, p_b_calc = get_p_from_grandH(grand_I, data)
        data['2']['p_a_calc_I'] = p_a_calc
        data['2']['p_b_calc_I'] = p_b_calc

        test_err_U += (data['2']['p_a'] - data['2']['p_a_calc']) ** 2
        test_err_U += (data['2']['p_b'] - data['2']['p_b_calc']) ** 2

        test_err_I += (data['2']['p_a'] - data['2']['p_a_calc_I']) ** 2
        test_err_I += (data['2']['p_b'] - data['2']['p_b_calc_I']) ** 2

    test_err_U = np.sqrt(test_err_U / (2 * len(all_data.keys())))
    test_err_I = np.sqrt(test_err_I / (2 * len(all_data.keys())))

    print('test error: ', test_err_U)
    print('test error I : ', test_err_I)

main()
