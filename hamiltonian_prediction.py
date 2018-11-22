import pandas as pd
import matplotlib.pyplot as plt
import pickle
from minimization_functions import *
from statsmodels.formula.api import ols
import time

from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import wilcoxon
from general_quantum_operators import *

import os.path



h_names_gen = ['0', '1', '2', '3', '01', '23']
h_names_letter = ['A', 'B', 'C', 'D', 'AB', 'CD', 'pred']

def sub_q_p(df, u_id, p_id):
    d = df[(df['userID'] == u_id) & (df['pos'] == p_id)]
    p = {
        'A': d['p1'].values,
        'B': d['p2'].values,
        'A_B': d['p12'].values
    }
    return p, d


def get_question_H(psi_0, all_q, p_real, h_a_and_b=None, with_mixing=True, h_mix_type = 0):
    sub_q_data = {}
    if h_a_and_b is None:
        # find h_a
        full_h = ['x', None, None]
        all_P = '0'
        res_temp = general_minimize(fun_to_minimize, args_=(p_real['A'], psi_0, full_h, all_q, all_P, 4, h_mix_type),
                                    x_0=np.array([0.0]))
        h_a = res_temp.x[0]

        full_h = [h_a, None, None]
        p_a = get_general_p(full_h, all_q, all_P, psi_0, n_qubits=4, h_mix_type = h_mix_type)
        sub_q_data['p_a'] = p_real['A']
        sub_q_data['p_a_h'] = p_a
        sub_q_data['p_a_err'] = res_temp.fun
        # print(p_a, p['A'])

        # find h_b
        full_h = [None, 'x', None]
        all_P = '1'
        res_temp = general_minimize(fun_to_minimize, args_=(p_real['B'], psi_0, full_h, all_q, all_P, 4, h_mix_type),
                                    x_0=np.array([0.0]))
        h_b = res_temp.x[0]

        full_h = [None, h_b, None]
        p_b = get_general_p(full_h, all_q, all_P, psi_0, n_qubits=4, h_mix_type = h_mix_type)
        sub_q_data['p_b'] = p_real['B']
        sub_q_data['p_b_h'] = p_b
        sub_q_data['p_b_err'] = res_temp.fun
        # print(p_b, p['B'])
    else:
        h_a = h_a_and_b[0]
        h_b = h_a_and_b[1]

    if with_mixing:
        # find h_ab
        full_h = [h_a, h_b, 'x']
        all_P = 'C'
        res_temp = general_minimize(fun_to_minimize, args_=(p_real['A_B'], psi_0, full_h, all_q, all_P, 4, h_mix_type),
                                    x_0=np.array([0.0]))
        # print(res_temp.fun)
        h_ab = res_temp.x[0]
    else:
        h_ab = 0.0

    full_h = [h_a, h_b, h_ab]
    p_ab = get_general_p(full_h, all_q, all_P, psi_0, n_qubits=4, h_mix_type = h_mix_type)
    sub_q_data['p_ab'] = p_real['A_B']
    sub_q_data['p_ab_h'] = p_ab
    sub_q_data['p_ab_err'] = np.sqrt((p_real['A_B'] - p_ab) ** 2)
    # print(p_ab, p['A_B'])

    full_h = [h_a, h_b, h_ab]
    total_H = compose_H(full_h, all_q, n_qubits=4, h_mix_type = h_mix_type)
    psi_final = get_psi(total_H, psi_0)
    sub_q_data['h_a'] = h_a
    sub_q_data['h_b'] = h_b
    sub_q_data['h_ab'] = h_ab
    sub_q_data['psi'] = psi_final

    return sub_q_data


def calculations_before_question3(h_mix_type):
    df = pd.read_csv('data/new_dataframe.csv', index_col=0)
    # df = df[df['user'].isin([0., 7., 8., 17.])] # Uncomment this when testing the code.

    # go over all individuals
    user_same_q_list = {}
    all_q_data = {}
    q_info = {}
    for qn in df[(df.pos == 2.)].qn.unique():
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
    t0 = 0 ; t1 = 0
    all_data = {}
    for ui, u_id in enumerate(df['userID'].unique()):

        # select only from one group that has the same third question
        t0 = time.time()
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
        t1 = time.time()

        print('Calculated states for user: {}/{},\ttime elapsed = {}'.format(ui, df['userID'].unique().__len__(), np.round(t1-t0,2)))

    fname2save = 'data/all_data_before3{}.pkl'.format(int(h_mix_type))
    pickle.dump([all_data,user_same_q_list, all_q_data, q_info], open(fname2save, 'wb'))


def calculate_all_data(use_U=True, with_mixing=True, use_neutral=False, h_mix_type = 0):
    df = pd.read_csv('data/new_dataframe.csv', index_col=0)
    # df = df[df['user'].isin([0., 7., 8., 17.])]
    fname2read = 'data/all_data_before3{}.pkl'.format(h_mix_type)
    all_data, user_same_q_list, all_q_data, q_info = pickle.load(open(fname2read, 'rb'))

    # todo: change once I run everything again and comment all the loop below!!!
    # # create list of users with the same qn in pos
    # user_same_q_list = {}
    # all_q_data = {}
    # q_info = {}
    # for qn in df[(df.qn == 2.)].pos.unique():
    #     user_same_q_temp = df[(df.pos == 2.) & (df.qn == qn)]['userID'] #
    #     # user_same_q_list.append(user_same_q_temp)
    #     user_same_q_list[qn] = user_same_q_temp.unique()
    #     all_q_data[qn] = {}
    #     first_user = user_same_q_temp.values[0]
    #     q_info[qn] = {
    #         'q1': df[(df.pos == 2.) & (df.userID == first_user)]['q1'].values,
    #         'q2': df[(df.pos == 2.) & (df.userID == first_user)]['q2'].values,
    #         'fal':df[(df.pos == 2.) & (df.userID == first_user)]['fal'].values
    #     }


    # third question
    print('third question')
    for qn, user_list in user_same_q_list.items():
        # go over all 4 types of questions

        for k, v in all_data.items():
            if k in user_list:
                p_real, d = sub_q_p(df, k, 2)
                all_q_data[qn][k] = deepcopy(v)
                all_q_data[qn][k][2] = {
                    'p_a': p_real['A'],
                    'p_b': p_real['B'],
                    'p_ab': p_real['A_B']
                }

        if len(all_q_data[qn]) > 0:

            all_q = [int(q_info[qn]['q1'][0])-1, int(q_info[qn]['q2'][0])-1]
            h_names = ['0', '1', '2', '3', '01', '23', str(all_q[0]) + str(all_q[1])]

            # find U for each question
            if use_U:
                res_temp = general_minimize(fun_to_minimize_grandH, args_=(all_q, all_q_data[qn], h_mix_type), x_0=np.zeros([10]))
                q_info[qn]['U'] = U_from_H(grandH_from_x(res_temp.x))
            else:
                q_info[qn]['U'] = np.eye(16)

            # calculate H_AB
            H_dict = {}
            for u_id in user_list:
                if u_id in all_data:
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
            df_H = pd.DataFrame.from_dict(data=H_dict, orient='index')
            df_H.columns = ['A', 'B', 'C', 'D', 'AB', 'CD', 'pred']

            formula = df_H.columns[-1] + '~' + df_H.columns[0]
            for h_i in range(1, len(df_H.columns)-1):
                formula += '+' + df_H.columns[h_i]
            est = ols(formula=formula, data=df_H).fit()
            q_info[qn]['H_ols'] = est

    print('before saving pkl')
    control_str = '_U_%s_mixing_%s_neutral_%s_mix_type_%d' % (use_U, with_mixing, use_neutral, h_mix_type)
    pickle.dump(all_data, open('data/all_data%s.pkl' % control_str, 'wb'))
    pickle.dump(q_info, open('data/q_info%s.pkl' %control_str, 'wb'))


def generate_predictions(use_U=True, with_mixing=True, use_neutral=False, h_mix_type = 0):
    control_str = '_U_%s_mixing_%s_neutral_%s_mix_type_%d' % (use_U, with_mixing, use_neutral, h_mix_type)
    all_data = pickle.load(open('data/all_data%s.pkl' % control_str, 'rb'))
    q_info = pickle.load(open('data/q_info%s.pkl' % control_str, 'rb'))
    df = pd.read_csv('data/new_dataframe.csv', index_col=0)
    # df = df[df['user'].isin([0., 7., 8., 17.])]

    pred_df_dict = {}
    # go over all individuals
    for u_id, data in all_data.items():
        # print('Generating prediction for ', u_id)
        pred_df_dict[u_id] = []
        pred_df_col_names = []

        # go over question 3-5
        for p_id in range(2, 6):
            p_real, d = sub_q_p(df, u_id, p_id)
            all_q = [int(d['q1'].values[0] - 1), int(d['q2'].values[0] - 1)]
            qn = int(d['qn'].values[0])

            # use question U to generate psi_0
            if use_neutral:
                psi_0 = uniform_psi(n_qubits=4)
            else:
                psi_0 = np.dot(q_info[qn]['U'], data[p_id - 1]['psi'])

            # use question H to generate h_ab
            if with_mixing:
                all_h = {'one': []}
                for hs in h_names_gen:
                    all_h['one'].append(data['h_q'][hs])
                df_H = pd.DataFrame.from_dict(data=all_h, orient='index')
                df_H.columns = ['A', 'B', 'C', 'D', 'AB', 'CD']
                h_ab = q_info[qn]['H_ols'].predict(df_H).values[0]
            else:
                h_ab = 0.0

            full_h = [data['h_q'][str(all_q[0])], data['h_q'][str(all_q[1])], h_ab]
            pred_p_a = get_general_p(full_h, all_q, '0', psi_0, n_qubits=4, h_mix_type = h_mix_type)
            pred_p_b = get_general_p(full_h, all_q, '1', psi_0, n_qubits=4, h_mix_type = h_mix_type)
            if q_info[qn]['fal'] == 1:
                pred_p_ab = get_general_p(full_h, all_q, 'C', psi_0, n_qubits=4, h_mix_type = h_mix_type)
            else:
                pred_p_ab = get_general_p(full_h, all_q, 'D', psi_0, n_qubits=4, h_mix_type = h_mix_type)

            total_H = compose_H(full_h, all_q, n_qubits=4, h_mix_type = h_mix_type)
            psi_final = get_psi(total_H, psi_0)
            data[p_id] = {
                'psi': psi_final
            }

            pred_df_col_names.append('q%d_pred_pa' % p_id)
            pred_df_dict[u_id].append(pred_p_a[0][0])
            pred_df_col_names.append('q%d_real_pa' % p_id)
            pred_df_dict[u_id].append(p_real['A'][0])

            pred_df_col_names.append('q%d_pred_pb' % p_id)
            pred_df_dict[u_id].append(pred_p_b[0][0])
            pred_df_col_names.append('q%d_real_pb' % p_id)
            pred_df_dict[u_id].append(p_real['B'][0])

            pred_df_col_names.append('q%d_pred_pab' % p_id)
            pred_df_dict[u_id].append(pred_p_ab[0][0])
            pred_df_col_names.append('q%d_real_pab' % p_id)
            pred_df_dict[u_id].append(p_real['A_B'][0])

    pred_df = pd.DataFrame.from_dict(data=pred_df_dict, orient='index')
    pred_df.columns = pred_df_col_names
    pred_df.to_csv('data/pred_df%s.csv' % control_str)


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
    res_temp = minimize(fun_to_minimize_grandH, np.zeros([10]), args=(train_data, h_mix_type),
                        method='SLSQP', bounds=None, options={'disp': False}) # todo: ***** missing all_q????? *******
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


h_type = [0]
use_U_l = [True, False]
use_neutral_l = [False, True]
with_mixing_l = [True, False]

print('Are you sure that:\n'
      '1) You changed (un)comment all the necessary lines?\n'
      '2) That all the parameters changed?\n'
      '\ta) num_of_minimizations.\n'
      '\tb) h_mix_type, use_U_l, use_neutral_l, with_mixing_l.')

# s = input('\n\n ############ Press any key to continue ############\n')

# Loop to run all controls, except the uniform or the mean
for h_mix_type in h_type:
    for use_U in use_U_l:
        for use_neutral in use_neutral_l:
            for with_mixing in with_mixing_l:

                print('Running:\tUse_U = {} |\tUse_Neutral = {} |\tWith_Mixing = {} |\th_mix_type = {}'.format(use_U,use_neutral,with_mixing, h_mix_type))

                if (use_U == False) & (use_neutral == False) & (with_mixing == True):
                    if not os.path.isfile('./data/all_data_before30.pkl') or not os.path.isfile('./data/all_data_before31.pkl'): # run once for every h_mix_type
                        calculations_before_question3(h_mix_type)

                calculate_all_data(use_U=use_U, use_neutral=use_neutral, with_mixing=with_mixing, h_mix_type=h_mix_type)

                generate_predictions(use_U=use_U, use_neutral=use_neutral, with_mixing=with_mixing, h_mix_type=h_mix_type)
