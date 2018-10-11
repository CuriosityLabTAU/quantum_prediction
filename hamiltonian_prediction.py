import pandas as pd
import matplotlib.pyplot as plt
import pickle
from minimization_functions import *
from statsmodels.formula.api import ols

from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import wilcoxon
from general_quantum_operators import *


h_names_gen = ['0', '1', '2', '3', '01', '23']

def sub_q_p(df, u_id, p_id):
    d = df[(df['userID'] == u_id) & (df['pos'] == p_id)]
    p = {
        'A': d['p1'].values,
        'B': d['p2'].values,
        'A_B': d['p12'].values
    }
    return p, d


def get_question_H(psi_0, all_q, p_real, h_a_and_b=None):
    sub_q_data = {}
    if h_a_and_b is None:
        # find h_a
        full_h = ['x', None, None]
        all_P = '0'
        res_temp = general_minimize(fun_to_minimize, args_=(p_real['A'], psi_0, full_h, all_q, all_P, 4), x_0=np.array([0.0]))
        h_a = res_temp.x[0]

        full_h = [h_a, None, None]
        p_a = get_general_p(full_h, all_q, all_P, psi_0, n_qubits=4)
        sub_q_data['p_a'] = p_real['A']
        sub_q_data['p_a_h'] = p_a
        sub_q_data['p_a_err'] = res_temp.fun
        # print(p_a, p['A'])

        # find h_b
        full_h = [None, 'x', None]
        all_P = '1'
        res_temp = general_minimize(fun_to_minimize, args_=(p_real['B'], psi_0, full_h, all_q, all_P, 4),
                                    x_0=np.array([0.0]))
        h_b = res_temp.x[0]
        full_h = [None, h_b, None]
        p_b = get_general_p(full_h, all_q, all_P, psi_0, n_qubits=4)
        sub_q_data['p_b'] = p_real['B']
        sub_q_data['p_b_h'] = p_b
        sub_q_data['p_b_err'] = res_temp.fun
        # print(p_b, p['B'])
    else:
        h_a = h_a_and_b[0]
        h_b = h_a_and_b[1]

    # find h_ab
    full_h = [h_a, h_b, 'x']
    all_P = 'C'
    res_temp = general_minimize(fun_to_minimize, args_=(p_real['A_B'], psi_0, full_h, all_q, all_P, 4),
                                x_0=np.array([0.0]))
    # print(res_temp.fun)
    h_ab = res_temp.x[0]
    full_h = [h_a, h_b, h_ab]
    p_ab = get_general_p(full_h, all_q, all_P, psi_0, n_qubits=4)
    sub_q_data['p_ab'] = p_real['A_B']
    sub_q_data['p_ab_h'] = p_ab
    sub_q_data['p_ab_err'] = res_temp.fun
    # print(p_ab, p['A_B'])

    full_h = [h_a, h_b, h_ab]
    total_H = compose_H(full_h, all_q, n_qubits=4)
    psi_final = get_psi(total_H, psi_0)
    sub_q_data['h_a'] = h_a
    sub_q_data['h_b'] = h_b
    sub_q_data['h_ab'] = h_ab
    sub_q_data['psi'] = psi_final

    return sub_q_data


def calculate_all_data(use_U=True, with_mixing=True, use_neutral=False):
    df = pd.read_csv('data/new_dataframe.csv', index_col=0)

    # go over all individuals
    user_same_q_list = {}
    all_q_data = {}
    q_info = {}
    for qn in df[(df.qn == 2.)].pos.unique():
        user_same_q_temp = df[(df.pos == 2.) & (df.qn == qn)]['userID'] #
        # user_same_q_list.append(user_same_q_temp)
        user_same_q_list[qn] = user_same_q_temp.unique()
        all_q_data[qn] = {}
        first_user = user_same_q_temp.values[0]
        q_info[qn] = {
            'q1': df[(df.pos == 2.) & (df.userID == first_user)]['q1'].values,
            'q2': df[(df.pos == 2.) & (df.userID == first_user)]['q2'].values,
            'fal':df[(df.pos == 2.) & (df.userID == first_user)]['fal'].values
        }

    # first two question, all subjects
    all_data = {}
    for ui, u_id in enumerate(df['userID'].unique()):
        if ui > 20:
            break

        # select only from one group that has the same third question
        print('calculating states for user #:',  ui)
        # go over questions 1 & 2
        psi_0 = uniform_psi(n_qubits=4)
        sub_data = {
            'h_q': {}
        }
        for p_id in range(2):
            p_real, d = sub_q_p(df, u_id, p_id)
            all_q = [int(d['q1'].values[0] - 1), int(d['q2'].values[0] - 1)]

            sub_data[p_id] = get_question_H(psi_0, all_q, p_real)

            if use_neutral:
                psi_0 = uniform_psi(n_qubits=4)
            else:
                psi_0 = sub_data[p_id]['psi']

            sub_data['h_q'][str(all_q[0])] = sub_data[p_id]['h_a']
            sub_data['h_q'][str(all_q[1])] = sub_data[p_id]['h_b']
            sub_data['h_q'][str(all_q[0])+str(all_q[1])] = sub_data[p_id]['h_ab']

        all_data[u_id] = sub_data

    # third question
    for qn, user_list in user_same_q_list.items():
        # go over all 4 types of questions

        for k, v in all_data.items():
            if k in user_list:
                all_q_data[qn][k] = deepcopy(v)

        all_q = [q_info[qn]['q1']-1, q_info[qn]['q2']-1]
        h_names = ['0', '1', '2', '3', '01', '23', str(all_q[0]) + str(all_q[1])]

        # find U for each question
        res_temp = general_minimize(fun_to_minimize_grandH, args_=(all_q, all_q_data), x_0=np.zeros([5]))
        q_info[qn]['U'] = U_from_H(grandH_from_x(res_temp.x))

        # calculate H_AB
        H_dict = {}
        for u_id in user_list:
            if use_neutral:
                psi_0 = uniform_psi(n_qubits=4)
            else:
                psi_0 = np.dot(q_info[qn]['U'], all_data[u_id][1]['psi'])
            p_real, d = sub_q_p(df, u_id, 2)
            sub_data_q = get_question_H(psi_0, all_q, p_real,
                                        [all_data[u_id]['h_q'][str(all_q[0])], all_data[u_id]['h_q'][str(all_q[1])]])
            all_data[u_id]['h_q'][str(all_q[0])+str(all_q[1])] = sub_data_q['h_ab']
            H_dict[u_id] = []
            for hs in h_names:
                H_dict[u_id].append(all_data[u_id]['h_q'][hs])
        df_H = pd.DataFrame.from_dict(data=H_dict, orient='index', columns=h_names)

        formula = h_names[-1] + '~' + h_names[0]
        for h_i in range(1, len(h_names)-1):
            formula += '+' + h_names[h_i]
        est = ols(formula=formula, data=df_H).fit()
        q_info[qn]['H_ols'] = est

    pickle.dump(all_data, 'data/all_data.pkl')
    pickle.dump(q_info, 'data/q_info.pkl')


def generate_predictions(use_U=True, with_mixing=True, use_neutral=False):
    all_data = pickle.load('data/all_data.pkl')
    q_info = pickle.load('data/q_info.pkl')
    df = pd.read_csv('data/new_dataframe.csv', index_col=0)

    pred_df_dict = {}
    # go over all individuals
    for u_id, data in all_data.items():
        pred_df_dict[u_id] = []
        pred_df_col_names = []

        # go over question 3-5
        for p_id in range(3,6):
            p_real, d = sub_q_p(df, u_id, p_id)
            all_q = [int(d['q1'].values[0] - 1), int(d['q2'].values[0] - 1)]
            qn = int(d['qn'].values[0])

            # use question U to generate psi_0
            if use_neutral:
                psi_0 = uniform_psi(n_qubits=4)
            else:
                psi_0 = np.dot(q_info[qn]['U'], data[p_id - 1]['psi'])

            # use question H to generate h_ab
            all_h = []
            for hs in h_names_gen:
                all_h.append(data['h_q'][hs])
            h_ab = q_info[qn]['H_ols'].predict(np.array(all_h))

            full_h = [data['h_q'][str(all_q[0])], data['h_q'][str(all_q[1])], h_ab]
            pred_p_a = get_general_p(full_h, all_q, '0', psi_0, n_qubits=4)
            pred_p_b = get_general_p(full_h, all_q, '1', psi_0, n_qubits=4)
            if q_info[qb]['fal'] == 1:
                pred_p_ab = get_general_p(full_h, all_q, 'C', psi_0, n_qubits=4)
            else:
                pred_p_ab = get_general_p(full_h, all_q, 'D', psi_0, n_qubits=4)

            total_H = compose_H(full_h, all_q, n_qubits=4)
            psi_final = get_psi(total_H, psi_0)
            data[p_id]['psi'] = psi_final

            pred_df_col_names.append('q%d_pred_pa' % p_id)
            pred_df_dict[u_id].append(pred_p_a)
            pred_df_col_names.append('q%d_real_pa' % p_id)
            pred_df_dict[u_id].append(p_real['A'])

            pred_df_col_names.append('q%d_pred_pb' % p_id)
            pred_df_dict[u_id].append(pred_p_b)
            pred_df_col_names.append('q%d_real_pb' % p_id)
            pred_df_dict[u_id].append(p_real['B'])

            pred_df_col_names.append('q%d_pred_pab' % p_id)
            pred_df_dict[u_id].append(pred_p_ab)
            pred_df_col_names.append('q%d_real_pab' % p_id)
            pred_df_dict[u_id].append(p_real['A_B'])

    pred_df = pd.DataFrame.from_dict(data=pred_df_dict, orient='index', columns=pred_df_col_names)
    pred_df.to_csv('data/pred_df.csv')


def all_data_to_csv(all_data):
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
