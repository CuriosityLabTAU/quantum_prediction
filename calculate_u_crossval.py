import random
import numpy as np
import pandas as pd

from hamiltonian_prediction import *
from sklearn.model_selection import train_test_split

# psi_dyn = np.dot(U, psi_0)

qubits_dict = {1:'a', 2:'b', 3:'c', 4:'d'}

def get_general_p_without_h_trial(all_q, which_prob, psi, n_qubits=4, is_normalized = False):
    '''calculate probability based on U and psi'''
    P_ = MultiProjection(which_prob, all_q, n_qubits)
    psi_final = np.dot(P_, psi)
    p_ = np.dot(np.conjugate(np.transpose(psi_final)), psi).real / np.dot(np.conjugate(np.transpose(psi_final)), psi_final).real
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
                print('calculating U for %d on train data' % qn)
                res_temp = general_minimize(fun_to_minimize_grandH, args_=(all_q, train_q_data_qn, h_mix_type), x_0=np.zeros([10]), U = True)
                end = time.clock()
                print('question %d, U optimization took %.2f s' % (qn, end - start))

                q_info[qn]['U'] = U_from_H(grandH_from_x(res_temp.x))
            else:
                q_info[qn]['U'] = np.eye(16)

        ### predict on test users --> with NO {H_ab}
        print('calculating errors on test data')
        U = q_info[qn]['U']
        for u_id, tu in test_q_data_qn.items():
            temp = {}
            temp['id'] = [u_id]
            temp['qn'] = [qn]

            temp['q1'] = [q_info[qn]['q1'][0]]
            temp['q2'] = [q_info[qn]['q2'][0]]

            q1 = 'p_' + qubits_dict[temp['q1'][0]]
            q2 = 'p_' + qubits_dict[temp['q2'][0]]

            temp['U'] = [use_U]

            ### psi after the 2nd question
            psi_0 = tu[1]['psi']

            ### propogate psi with the U of the 3rd question
            psi_dyn = np.dot(U, psi_0)

            ### probabilities from the 1st and 2nd question
            temp['p_a'] = [tu[0]['p_a'][0]]
            temp['p_b'] = [tu[0]['p_b'][0]]
            temp['p_c'] = [tu[1]['p_a'][0]]
            temp['p_d'] = [tu[1]['p_b'][0]]

            ### probs of the current question
            temp['p_a_pre'] = temp[q1]
            temp['p_b_pre'] = temp[q2]

            ### real probabilities
            temp['p_a_real'] = [tu[2]['p_a'][0]]
            temp['p_b_real'] = [tu[2]['p_b'][0]]
            temp['p_ab_real'] = [tu[2]['p_ab'][0]]

            ### predicted probabilities
            # full_h = [tu['h_q'][str(int(temp['q1'][0]) - 1)], tu['h_q'][str(int(temp['q2'][0]) - 1)], None]
            h_a = [tu['h_q'][str(int(temp['q1'][0]) - 1)], None, None]
            h_b = [None, tu['h_q'][str(int(temp['q2'][0]) - 1)], None]
            temp['p_a_pred_U'] = [get_general_p(h_a, all_q, '0', psi_dyn, n_qubits=4, is_normalized=True).flatten()[0]]
            temp['p_b_pred_U'] = [get_general_p(h_b, all_q, '1', psi_dyn, n_qubits=4, is_normalized=True).flatten()[0]]

            temp['p_a_pred_I'] = [get_general_p(h_a, all_q, '0', psi_0, n_qubits=4, is_normalized=True).flatten()[0]]
            temp['p_b_pred_I'] = [get_general_p(h_b, all_q, '1', psi_0, n_qubits=4, is_normalized=True).flatten()[0]]

            # ### joint probabilities
            # if q_info[qn]['fal'][0] == 1:
            #     temp['p_ab_pred'] = [get_general_p_without_h(all_q, 'C', psi_dyn, n_qubits=4)]
            # if q_info[qn]['fal'][0] == 2:
            #     temp['p_ab_pred'] = [get_general_p_without_h(all_q, 'D', psi_dyn, n_qubits=4)]
            # temp = pd.DataFrame(temp)

            ### calculate the error from the previous probabilities with NO U.
            temp['p_a_err_real_pre']  = [np.abs(temp['p_a_real'][0] - temp['p_a_pre'][0])]
            temp['p_b_err_real_pre']  = [np.abs(temp['p_b_real'][0] - temp['p_b_pre'][0])]

            ### calculate the error from the propogated state with U
            temp['p_a_err_real_U'] = [np.abs(temp['p_a_real'][0] - temp['p_a_pred_U'][0])]
            temp['p_b_err_real_U'] = [np.abs(temp['p_b_real'][0] - temp['p_b_pred_U'][0])]

            ### calculate the error from the full 4 qubits state with I
            temp['p_a_err_real_I'] = [np.abs(temp['p_a_real'][0] - temp['p_a_pred_I'][0])]
            temp['p_b_err_real_I'] = [np.abs(temp['p_b_real'][0] - temp['p_b_pred_I'][0])]

            prediction_errors = pd.concat([prediction_errors,pd.DataFrame(temp)], axis = 0)

    prediction_errors.set_index('id', inplace=True)
    prediction_errors.to_csv('data/calc_U/cross_val_prediction_errors_%s.csv' % control_str)#index=False)

    print('before saving pkl')
    pickle.dump(all_data, open('data/calc_U/all_data%s.pkl' % control_str, 'wb'))
    pickle.dump(q_info, open('data/calc_U/q_info%s.pkl' %control_str, 'wb'))

    # df_H_all.to_csv('data/calc_U/df_H%s.csv' % control_str)

def main():
    h_type = [0]
    use_U_l = [True]
    use_neutral_l = [False]
    with_mixing_l = [True]
    comb = product(h_type, use_U_l, use_neutral_l, with_mixing_l)

    calcU = True
    # calcU = False

    if calcU:

        for h_mix_type, use_U, use_neutral, with_mixing in comb:

            print('Running:\tUse_U = {} |\tUse_Neutral = {} |\tWith_Mixing = {} |\th_mix_type = {}'.format(use_U,use_neutral,with_mixing, h_mix_type))

            control_str = 'pred_df_U_%s_mixing_%s_neutral_%s_mix_type_%d.csv' % (use_U, with_mixing, use_neutral, h_mix_type)
            # if os.path.isfile('./data/' + control_str):
            #     print('Already calculated everything for this combination')
            #     continue

            calculate_all_data_cross_val(use_U=use_U, use_neutral=use_neutral, with_mixing=with_mixing, h_mix_type=h_mix_type)
    else:
        for h_mix_type, use_U, use_neutral, with_mixing in comb:
            control_str = '_U_%s_mixing_%s_neutral_%s_mix_type_%d' % (use_U, with_mixing, use_neutral, h_mix_type)
            prediction_errors = pd.read_csv('data/calc_U/cross_val_prediction_errors_%s.csv' % control_str)

            ### list of the columns of the errors
            err_cl = list(prediction_errors.columns[prediction_errors.columns.str.contains('err')])

            ### group the errors by question
            grouped = prediction_errors[err_cl + ['qn']].groupby('qn')

            ### boxplot of the results
            fig, ax = plt.subplots(1,1, figsize = (8,6))
            grouped.boxplot(rot = 45, ax = ax)
            plt.tight_layout(pad = 1)
            print()


if __name__ == '__main__':
    main()
