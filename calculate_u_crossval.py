import random
import numpy as np
import pandas as pd

from hamiltonian_prediction import *
from sklearn.model_selection import train_test_split

# psi_dyn = np.dot(U, psi_0)

def get_general_p(U, all_q, which_prob, psi, n_qubits=4):
    '''calculate probability based on U and psi'''
    P_ = MultiProjection(which_prob, all_q, n_qubits)
    psi_final = np.dot(P_, psi)
    p_ = norm_psi(psi_final)
    return p_

def sub_sample_data(all_data, data_qn, df, users):
    '''return data'''
    for k, v in all_data.items():
        if k in users:
            p_real, d = sub_q_p(df, k, 2)
            data_qn[k] = deepcopy(v)
            data_qn[k][2] = {
                'p_a': p_real['A'],
                'p_b': p_real['B'],
                'p_ab': p_real['A_B']
            }

    return  data_qn


def calculate_all_data_cross_val(use_U=True, with_mixing=True, use_neutral=False, h_mix_type = 0):
    '''cross validation only for the third question'''

    control_str = '_U_%s_mixing_%s_neutral_%s_mix_type_%d' % (use_U, with_mixing, use_neutral, h_mix_type)

    ### load data
    df = pd.read_csv('data/new_dataframe.csv', index_col=0)
    # df = df[df['user'].isin([0., 7., 8., 17.])]
    fname2read = './data/all_data_before3_N{}_M{}_h{}.pkl'.format(str(use_neutral)[0], str(with_mixing)[0], int(h_mix_type))
    all_data, user_same_q_list, all_q_data, q_info = pickle.load(open(fname2read, 'rb'))

    # third question
    ### creating a dataframe to save all the predictions error --> for specific question group by 'qn' --> agg('mean')
    prediction_errors = pd.DataFrame()

    ### Run on all users that have the same third question.
    for qn, user_list in user_same_q_list.items():
        # go over all 4 types of questions

        ### split the users to test and train
        train_users, test_users, _, __ = train_test_split(user_list,user_list,test_size=.2)
        train_q_data_qn = {}
        test_q_data_qn = {}

        train_q_data_qn = sub_sample_data(all_data, train_q_data_qn, df, train_users)
        test_q_data_qn  = sub_sample_data(all_data, test_q_data_qn , df, test_users)


        if len(train_q_data_qn) > 0:
            ### question qubits (-1) because if the range inside of some function
            all_q = [int(q_info[qn]['q1'][0])-1, int(q_info[qn]['q2'][0])-1]
            h_names = ['0', '1', '2', '3', '01', '23', str(all_q[0]) + str(all_q[1])]

            # find U for each question
            if use_U:
                start = time.clock()
                print('calculating U for %d' % qn)
                res_temp = general_minimize(fun_to_minimize_grandH, args_=(all_q, train_q_data_qn, h_mix_type), x_0=np.zeros([10]), U = True)
                end = time.clock()
                print('question %d, U optimization took %.2f s' % (qn, end - start))

                q_info[qn]['U'] = U_from_H(grandH_from_x(res_temp.x))
            else:
                q_info[qn]['U'] = np.eye(16)

        ### predict on test users
        U = q_info[qn]['U']
        for tu in test_q_data_qn:
            temp = {}
            temp['id'] = tu
            temp['qn'] = qn
            temp['U'] = use_U

            ### psi after the 2nd question
            psi_0 = tu[1]['psi']

            ### propogate psi with the U of the 3rd question
            psi_dyn = np.dot(U, psi_0)

            ### real probabilities
            temp['p_a_real'] = [tu[2]['p_a'][0]]
            temp['p_a_real'] = [tu[2]['p_b'][0]]
            temp['p_ab_real'] = [tu[2]['p_ab'][0]]

            ### predicted probabilities
            temp['p_a_pred'] = [get_general_p(U, all_q, '1', psi_dyn, n_qubits=4)]
            temp['p_b_pred'] = [get_general_p(U, all_q, '2', psi_dyn, n_qubits=4)]
            if q_info[qn] == 1:
                temp['p_ab_pred'] = [get_general_p(U, all_q, 'C', psi_dyn, n_qubits=4)]
            if q_info[qn] == 2:
                temp['p_ab_pred'] = [get_general_p(U, all_q, 'D', psi_dyn, n_qubits=4)]
            temp = pd.DataFrame(temp)

            prediction_errors = pd.concat([prediction_errors,temp], axis = 0) # todo: check that this is working.

    prediction_errors.set_index('id', inplace=True)
    prediction_errors.to_csv('data/calc_U/cross_val_prediction_errors_%s.csv' % control_str)#index=False)

            #   ### calculate h_ij...
            # start = time.clock()
            # print('building df_H, calculating h_ij for question 2')
            #
            # # calculate H_AB
            # H_dict = {}
            # full_user_list = []
            # for u_id in user_list:
            #     if u_id in all_data:
            #         if use_neutral:
            #             psi_0 = uniform_psi(n_qubits=4)
            #         else:
            #             psi_0 = np.dot(q_info[qn]['U'], all_data[u_id][1]['psi'])
            #         p_real, d = sub_q_p(df, u_id, 2)
            #         sub_data_q = get_question_H(psi_0, all_q, p_real,
            #                                     [all_data[u_id]['h_q'][str(all_q[0])], all_data[u_id]['h_q'][str(all_q[1])]],
            #                                     with_mixing, h_mix_type, fallacy_type = q_info[qn]['fal'][0])
            #         all_data[u_id]['h_q'][str(all_q[0])+str(all_q[1])] = sub_data_q['h_ab']
            #         all_data[u_id]['h_q'][str(all_q[0])] = sub_data_q['h_a']
            #         all_data[u_id]['h_q'][str(all_q[1])] = sub_data_q['h_b']
            #         H_dict[u_id] = []
            #         for hs in h_names:
            #             H_dict[u_id].append(all_data[u_id]['h_q'][hs])
            #
            # df_H = pd.DataFrame.from_dict(data=H_dict, orient='index')
            # df_H.columns = ['A', 'B', 'C', 'D', 'AB', 'CD', 'pred']
            # end = time.clock()
            # print('question %d, building df_H time %.2f s' % (qn, end - start))
            #
            # start = time.clock()
            # mtd = 'lr' #'ANN'
            # print('calculating h_ij' + mtd)
            # est = pred_h_ij(df_H, method = mtd)
            # end = time.clock()
            # print('question %d, h_ij prediction took %.2f s' % (qn, end - start))
            #
            # q_info[qn]['H_ols'] = est
            #
            # df_H.index = user_list
            # if 'df_H_all' in locals():
            #     df_H_all = df_H_all.append(df_H)
            # else:
            #     df_H_all = df_H.copy()
            # # df_H_all = df_H_all.reset_index(drop=True)

    print('before saving pkl')
    pickle.dump(all_data, open('data/calc_U/all_data%s.pkl' % control_str, 'wb'))
    pickle.dump(q_info, open('data/calc_U/q_info%s.pkl' %control_str, 'wb'))

    df_H_all.to_csv('data/calc_U/df_H%s.csv' % control_str)


h_type = [0]
use_U_l = [True, False]
use_neutral_l = [False]
with_mixing_l = [True]
comb = product(h_type, use_U_l, use_neutral_l, with_mixing_l)

for h_mix_type, use_U, use_neutral, with_mixing in comb:

    print('Running:\tUse_U = {} |\tUse_Neutral = {} |\tWith_Mixing = {} |\th_mix_type = {}'.format(use_U,use_neutral,with_mixing, h_mix_type))

    control_str = 'pred_df_U_%s_mixing_%s_neutral_%s_mix_type_%d.csv' % (use_U, with_mixing, use_neutral, h_mix_type)
    # if os.path.isfile('./data/' + control_str):
    #     print('Already calculated everything for this combination')
    #     continue

    calculate_all_data_cross_val(use_U=use_U, use_neutral=use_neutral, with_mixing=with_mixing, h_mix_type=h_mix_type)