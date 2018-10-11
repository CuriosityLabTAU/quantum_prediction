from copy import deepcopy
from scipy.optimize import minimize
import numpy as np
from scipy.linalg import expm
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import wilcoxon
from general_quantum_operators import *


def compose_H(H1_, H2_, Hmix_, n_qubits=2):
    dim_ = 2 ** n_qubits
    H_ = np.kron(H1_, np.eye(2)) + np.kron(np.eye(2), H2_)

    mix = np.zeros([dim_, dim_])
    mix[0, 0] = Hmix_[0, 0]
    mix[0, -1] = Hmix_[0, 1]
    mix[-1, 0] = Hmix_[1, 0]
    mix[-1, -1] = Hmix_[1, 1]

    H_ += mix

    return H_





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
    err_ = rmse(p_, real_p_)
    return err_


def get_prob_mix(psi_0, H_, n_qubits=2, the_qubits=[]):

    psi_ = np.dot(Projection(0, n_qubits), np.dot(Projection(1, n_qubits), np.dot(U_from_H(H_), psi_0)))
    p_ = norm_psi(psi_).real
    return p_


def prob_from_hmix(hmix_, h_a, h_b, psi_0, n_qubits=2):
    H_ = compose_H(param_H(h_a), param_H(h_b), param_Hmix(hmix_))
    p_ = get_prob_mix(psi_0, H_, n_qubits)
    return p_


def fun_to_minimize_mix(h_, real_p_, h_a, h_b, psi_0, n_qubits=2):
    p_ = prob_from_hmix(h_, h_a, h_b, psi_0, n_qubits=2)
    err_ = rmse(p_, real_p_)
    return err_


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

    err_ = []
    for data in all_data.values():
        p_a = data['2']['p_a']
        p_b = data['2']['p_b']

        p_a_calc, p_b_calc = get_p_from_grandH(grand_U, data)
        err_.append((p_a_calc - p_a) ** 2)
        err_.append((p_b_calc - p_b) ** 2)

    return np.sqrt(np.mean(err_))


# finding H_AD
def fun_to_minimize_HAD(hmix_, real_p_, h_a, h_b, U, psi_0, n_qubits=4):
    U_psi = np.dot(U, psi_0)                        # 16 x 1
    H_ = compose_H(param_H(h_a), param_H(h_b), param_Hmix(hmix_))   # TODO: we need this - 16x16
    psi_ = np.dot(Projection(0, n_qubits), np.dot(Projection(3, n_qubits), np.dot(U_from_H(H_), U_psi)))
    p_ = norm_psi(psi_).real
    err_ = rmse(p_, real_p_)
    return err_


def general_minimize(f, args_, x_0):
    min_err = 100.0
    best_result = None
    for i in range(100):
        x_0_rand = np.random.random(x_0.shape) * 2.0 - 1.0
        res_temp = minimize(f, x_0_rand, args=args_, method='SLSQP', bounds=None, options={'disp': False})
        if res_temp.fun < min_err:
            min_err = res_temp.fun
            best_result = deepcopy(res_temp)

    return best_result


def calculate_all_data():
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

            sub_data[p_id] = {}

            # find h_a
            res_temp = general_minimize(fun_to_minimize, args_=(p['A'], psi_0, 0), x_0=np.array([0.0]))
            # res_temp = minimize(fun_to_minimize, 0.0, args=(p['A'], psi_0, 0),
            #                     method='SLSQP', bounds=None, options={'disp': False})
            # print(res_temp.fun)
            h_a = res_temp.x
            p_a = prob_from_h(h_a, psi_0, 0)
            sub_data[p_id]['p_a'] = p['A']
            sub_data[p_id]['p_a_h'] = p_a
            sub_data[p_id]['p_a_err'] = res_temp.fun
            # print(p_a, p['A'])

            # find h_b
            res_temp = general_minimize(fun_to_minimize, args_=(p['B'], psi_0, 1), x_0=np.array([0.0]))
            # res_temp = minimize(fun_to_minimize, 0.0, args=(p['B'], psi_0, 1),
            #                     method='SLSQP', bounds=None, options={'disp': False})
            # print(res_temp.fun)
            h_b = res_temp.x  #find h_mix
            p_b = prob_from_h(h_b, psi_0, 1)
            sub_data[p_id]['p_b'] = p['B']
            sub_data[p_id]['p_b_h'] = p_b
            sub_data[p_id]['p_b_err'] = res_temp.fun
            # print(p_b, p['B'])

            # find
            res_temp = general_minimize(fun_to_minimize_mix, args_=(p['A_B'], h_a, h_b, psi_0, 1), x_0=np.array([0.0]))
            # res_temp = minimize(fun_to_minimize_mix, 0.0, args=(p['A_B'], h_a, h_b, psi_0, 1),
            #                     method='SLSQP', bounds=None, options={'disp': False})
            # print(res_temp.fun)
            h_ab = res_temp.x
            p_ab = prob_from_hmix(h_ab, h_a, h_b, psi_0)
            sub_data[p_id]['p_ab'] = p['A_B']
            sub_data[p_id]['p_ab_h'] = p_ab
            sub_data[p_id]['p_ab_err'] = res_temp.fun
            # print(p_ab, p['A_B'])

            total_H = compose_H(param_H(h_a), param_H(h_b), param_Hmix(h_ab))
            psi_final = get_psi(psi_0, total_H)
            sub_data[p_id]['h_a'] = h_a
            sub_data[p_id]['h_b'] = h_b
            sub_data[p_id]['h_ab'] = h_ab
            sub_data[p_id]['psi'] = psi_final

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

    pickle.dump(all_data, 'data/all_data.pkl')

    all_data_list = []
    for k, v in all_data.items():
        sub_data_list = []
        col_names = []
        for q in [0, 1]:
            for p_s in ['a', 'b', 'ab']:
                for t_s in ['', '_h', '_err']:
                    col_names.append('%s_%s%s' % (q, p_s, t_s))
                    sub_data_list.append(np.squeeze(v[q]['p_%s%s' % (p_s, t_s)]))
        q = '2'
        for p_s in ['a', 'b']:
            col_names.append('q%s_p%s' % (q, p_s))
            sub_data_list.append(np.squeeze(v[q]['p_%s' % (p_s)]))
        all_data_list.append(sub_data_list)
    df_all_data = pd.DataFrame(data=all_data_list, columns=col_names)
    df_all_data.to_csv('data/all_data_df.csv')
    return all_data


def calculate_errors():
    all_data_in = pickle.load('data/all_data.pkl')
    # first filter bad q0, q1:
    filter_threshold = 0.05
    all_data = {}
    for k, v in all_data_in.items():
        if v[0]['p_a_err'] < filter_threshold and v[0]['p_b_err'] < filter_threshold and v[0]['p_ab_err'] < filter_threshold:
            if v[1]['p_a_err'] < filter_threshold and v[1]['p_b_err'] < filter_threshold and v[1]['p_ab_err'] < filter_threshold:
                all_data[k] = deepcopy(v)

    print('Before filter: ', len(all_data_in.keys()), 'After filter: ', len(all_data))

    # to find U_3, and H_AD
    n_user = len(all_data.keys())
    n_train = int(0.8 * n_user)
    user_rand_order = np.random.permutation(np.arange(n_user))
    user_rand_order = np.array(all_data.keys())[user_rand_order]
    user_train = user_rand_order[:n_train].tolist()
    user_test = user_rand_order[n_train:].tolist()

    # for U
    train_data = {}
    train_p = np.zeros([len(user_train), 2])
    for i_t, i_train in enumerate(user_train):
        train_data[i_t] = all_data[i_train]
        train_p[i_t, :] = all_data[i_train]['2']['p_a'], all_data[i_train]['2']['p_b']

    test_data = {}
    for i_t, i_test in enumerate(user_test):
        test_data[i_t] = all_data[i_test]
    res_temp = minimize(fun_to_minimize_grandH, np.zeros([5]), args=(train_data),
                        method='SLSQP', bounds=None, options={'disp': False})
    print('train error: ', res_temp.fun)
    print(res_temp.x)

    # given U, calculate p_a, p_b for all 100%
    grand_U = U_from_H(grandH_from_x(res_temp.x))
    grand_I = np.eye(16)


    # for H_AD
    reg_data = np.zeros([len(user_train), 7])
    for i_t, i_train in enumerate(user_train):
        reg_data[i_t, 0:3] = all_data[i_train][0]['h_a'], all_data[i_train][0]['h_b'], all_data[i_train][0]['h_ab']
        reg_data[i_t, 3:6] = all_data[i_train][1]['h_a'], all_data[i_train][1]['h_b'], all_data[i_train][1]['h_ab']
        reg_data[i_t, 6] = all_data[i_train][]

    test_err_U = []
    test_err_I = []
    test_err_mean_train = []
    test_err_uniform = []

    for u_id, data in all_data.items():
        p_a_calc, p_b_calc = get_p_from_grandH(grand_U, data)
        data['2']['p_a_calc'] = p_a_calc
        data['2']['p_b_calc'] = p_b_calc

        p_a_calc, p_b_calc = get_p_from_grandH(grand_I, data)
        data['2']['p_a_calc_I'] = p_a_calc
        data['2']['p_b_calc_I'] = p_b_calc

        test_err_U.append((data['2']['p_a'] - data['2']['p_a_calc']) ** 2)
        test_err_U.append((data['2']['p_b'] - data['2']['p_b_calc']) ** 2)

        test_err_I.append((data['2']['p_a'] - data['2']['p_a_calc_I']) ** 2)
        test_err_I.append((data['2']['p_b'] - data['2']['p_b_calc_I']) ** 2)

        test_err_mean_train.append((np.mean(train_p[:, 0]) - data['2']['p_a']) ** 2)
        test_err_mean_train.append((np.mean(train_p[:, 1]) - data['2']['p_b']) ** 2)

        test_err_uniform.append((0.5 - data['2']['p_a']) ** 2)
        test_err_uniform.append((0.5 - data['2']['p_b']) ** 2)

    print('test error: ', np.sqrt(np.mean(test_err_U)), np.sqrt(np.std(test_err_U)))
    print('test error I : ', np.sqrt(np.mean(test_err_I)), np.sqrt(np.std(test_err_I)))
    print('test error mean train : ', np.sqrt(np.mean(test_err_mean_train)), np.sqrt(np.std(test_err_mean_train)))
    print('test error uniform : ', np.sqrt(np.mean(test_err_uniform)), np.sqrt(np.std(test_err_uniform)))


    test_df = pd.DataFrame(data=np.squeeze(np.array([test_err_U, test_err_I, test_err_mean_train, test_err_uniform]).T),
                           columns=['U', 'I', 'Mean', 'Uniform'])
    test_df.to_csv('data/test_errors_df.csv')


def show_results():
    test_df = pd.read_csv('data/test_errors_df.csv', index_col=0)
    print(test_df.describe())
    for t0 in test_df.columns:
        for t1 in test_df.columns:
            if t0 != t1:
                print(t0, t1, wilcoxon(test_df[t0], test_df[t1]))

    print('=== Hypotheses =====')
    print('U - Mean (given training, quantum helps): ', test_df['U'].mean(), test_df['Mean'].mean(),
          wilcoxon(test_df['U'], test_df['Mean']))

    print('I - Uniform (no training, quantum helps): ', test_df['I'].mean(), test_df['Uniform'].mean(),
          wilcoxon(test_df['I'], test_df['Uniform']))

    print('U - I (given quantum, training helps): ', test_df['U'].mean(), test_df['I'].mean(),
          wilcoxon(test_df['U'], test_df['I']))

    test_df.boxplot()
    plt.show()

# all_data_ = calculate_all_data()
calculate_errors(all_data_)
show_results()
