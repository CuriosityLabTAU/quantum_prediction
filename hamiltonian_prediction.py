import pandas as pd
import matplotlib.pyplot as plt
import pickle
from minimization_functions import *
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import wilcoxon
from general_quantum_operators import *


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
        psi_0 = uniform_psi(n_qubits=4)
        sub_data = {}
        for p_id in range(2):

            d = df[(df['userID'] == u_id) & (df['pos'] == p_id)]
            p = {
                'A': d['p1'].values,
                'B': d['p2'].values,
                'A_B': d['p12'].values
            }
            all_q = [int(d['q1'].values[0] - 1), int(d['q2'].values[0] - 1)]

            sub_data[p_id] = {}

            # find h_a
            full_h = ['x', None, None]
            all_P = '0'
            res_temp = general_minimize(fun_to_minimize,  args_=(p['A'], psi_0, full_h, all_q, all_P, 4), x_0=np.array([0.0]))
            h_a = res_temp.x[0]

            full_h = [h_a, None, None]
            p_a = get_general_p(full_h, all_q, all_P, psi_0, n_qubits=4)
            sub_data[p_id]['p_a'] = p['A']
            sub_data[p_id]['p_a_h'] = p_a
            sub_data[p_id]['p_a_err'] = res_temp.fun
            # print(p_a, p['A'])

            # find h_b
            full_h = [None, 'x', None]
            all_P = '1'
            res_temp = general_minimize(fun_to_minimize, args_=(p['B'], psi_0, full_h, all_q, all_P, 4),
                                        x_0=np.array([0.0]))
            h_b = res_temp.x[0]
            full_h = [None, h_b, None]
            p_b = get_general_p(full_h, all_q, all_P, psi_0, n_qubits=4)
            sub_data[p_id]['p_b'] = p['B']
            sub_data[p_id]['p_b_h'] = p_b
            sub_data[p_id]['p_b_err'] = res_temp.fun
            # print(p_b, p['B'])

            # find
            full_h = [h_a, h_b, 'x']
            all_P = 'C'
            res_temp = general_minimize(fun_to_minimize, args_=(p['A_B'], psi_0, full_h, all_q, all_P, 4),
                                        x_0=np.array([0.0]))
            # print(res_temp.fun)
            h_ab = res_temp.x[0]
            full_h = [h_a, h_b, h_ab]
            p_ab = get_general_p(full_h, all_q, all_P, psi_0, n_qubits=4)
            sub_data[p_id]['p_ab'] = p['A_B']
            sub_data[p_id]['p_ab_h'] = p_ab
            sub_data[p_id]['p_ab_err'] = res_temp.fun
            # print(p_ab, p['A_B'])

            full_h = [h_a, h_b, h_ab]
            total_H = compose_H(full_h, all_q, n_qubits=4)
            psi_final = get_psi(total_H, psi_0)
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
        # reg_data[i_t, 6] = all_data[i_train][]

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

all_data_ = calculate_all_data()
# calculate_errors(all_data_)
# show_results()
