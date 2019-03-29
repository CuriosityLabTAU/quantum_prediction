import random
import numpy as np
import pandas as pd

from hamiltonian_prediction import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import seaborn as sns
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


def calculate_all_data_cross_val(use_U=True, with_mixing=True, use_neutral=False, h_mix_type = 0, i = None):
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

        ### split the users to test and train - 10 times
        train_users, test_users, _, __ = train_test_split(user_list,user_list,test_size=.2)
        train_q_data_qn = {}
        test_q_data_qn = {}

        train_q_data_qn = sub_sample_data(all_data, train_q_data_qn, df, train_users)
        test_q_data_qn  = sub_sample_data(all_data, test_q_data_qn , df, test_users)

        if len(train_q_data_qn) > 0:
            ### question qubits (-1) because if the range inside of some function
            all_q = [int(q_info[qn]['q1'][0])-1, int(q_info[qn]['q2'][0])-1]
            h_names = ['0', '1', '2', '3', '01', '23', str(all_q[0]) + str(all_q[1])]

            # find U for each question #
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
            temp['p_a_pred_U'] = [get_general_p(h_a, all_q, '0', psi_dyn, n_qubits=4).flatten()[0]]
            temp['p_b_pred_U'] = [get_general_p(h_b, all_q, '1', psi_dyn, n_qubits=4).flatten()[0]]

            temp['p_a_pred_I'] = [get_general_p(h_a, all_q, '0', psi_0, n_qubits=4).flatten()[0]]
            temp['p_b_pred_I'] = [get_general_p(h_b, all_q, '1', psi_0, n_qubits=4).flatten()[0]]

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
    prediction_errors.to_csv('data/calc_U/cross_val_prediction_errors_%s_%d.csv' % (control_str, i))#index=False)

    print('before saving pkl')
    pickle.dump(all_data, open('data/calc_U/all_data%s_%d.pkl' % (control_str, i), 'wb'))
    pickle.dump(q_info, open('data/calc_U/q_info%s_%d.pkl' % (control_str, i), 'wb'))

    # df_H_all.to_csv('data/calc_U/df_H%s.csv' % control_str)

def calculate_all_data_cross_val_kfold(use_U=True, with_mixing=True, use_neutral=False, h_mix_type = 0, i = None):
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

        ### split the users to test and train using kfold - each user will be one time in test
        kf = KFold(n_splits= 10) # todo: num_of_repeats
        kf.get_n_splits(user_list)

        for train_index, test_index in kf.split(user_list):
            train_users, test_users = user_list[train_index], user_list[test_index]
            train_q_data_qn = {}
            test_q_data_qn = {}

            train_q_data_qn = sub_sample_data(all_data, train_q_data_qn, df, train_users)
            test_q_data_qn = sub_sample_data(all_data, test_q_data_qn, df, test_users)

            p_a_80 = []
            p_b_80 = []
            for u_id, tu in test_q_data_qn.items():
                p_a_80.append(tu[2]['p_a'][0])
                p_b_80.append(tu[2]['p_b'][0])
            p_a_80 = np.array(p_a_80).mean()
            p_b_80 = np.array(p_b_80).mean()

            if len(train_q_data_qn) > 0:
                ### question qubits (-1) because if the range inside of some function
                all_q = [int(q_info[qn]['q1'][0])-1, int(q_info[qn]['q2'][0])-1]
                h_names = ['0', '1', '2', '3', '01', '23', str(all_q[0]) + str(all_q[1])]

                # find U for each question #
                start = time.clock()
                print('calculating U for %d on train data' % qn)
                res_temp = general_minimize(fun_to_minimize_grandH, args_=(all_q, train_q_data_qn, h_mix_type), x_0=np.zeros([10]), U = True)
                end = time.clock()
                print('question %d, U optimization took %.2f s' % (qn, end - start))

                q_info[qn]['U'] = U_from_H(grandH_from_x(res_temp.x))


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

                ### probs of the current question taken from previous questions
                temp['p_a_pre'] = temp[q1]
                temp['p_b_pre'] = temp[q2]

                ### real probabilities
                temp['p_a_real'] = [tu[2]['p_a'][0]]
                temp['p_b_real'] = [tu[2]['p_b'][0]]
                temp['p_ab_real'] = [tu[2]['p_ab'][0]]

                ### predicted probabilities with U
                # full_h = [tu['h_q'][str(int(temp['q1'][0]) - 1)], tu['h_q'][str(int(temp['q2'][0]) - 1)], None]
                h_a = [tu['h_q'][str(int(temp['q1'][0]) - 1)], None, None]
                h_b = [None, tu['h_q'][str(int(temp['q2'][0]) - 1)], None]
                temp['p_a_pred_U'] = [get_general_p(h_a, all_q, '0', psi_dyn, n_qubits=4).flatten()[0]]
                temp['p_b_pred_U'] = [get_general_p(h_b, all_q, '1', psi_dyn, n_qubits=4).flatten()[0]]

                ### predicted probabilities with I
                temp['p_a_pred_I'] = [get_general_p(h_a, all_q, '0', psi_0, n_qubits=4).flatten()[0]]
                temp['p_b_pred_I'] = [get_general_p(h_b, all_q, '1', psi_0, n_qubits=4).flatten()[0]]

                ### calculate the error from the previous probabilities with NO U.
                temp['p_a_err_real_pre']  = [np.abs(temp['p_a_real'][0] - temp['p_a_pre'][0])]
                temp['p_b_err_real_pre']  = [np.abs(temp['p_b_real'][0] - temp['p_b_pre'][0])]

                ### calculate the error from the propogated state with U
                temp['p_a_err_real_U'] = [np.abs(temp['p_a_real'][0] - temp['p_a_pred_U'][0])]
                temp['p_b_err_real_U'] = [np.abs(temp['p_b_real'][0] - temp['p_b_pred_U'][0])]

                ### calculate the error from the full 4 qubits state with I
                temp['p_a_err_real_I'] = [np.abs(temp['p_a_real'][0] - temp['p_a_pred_I'][0])]
                temp['p_b_err_real_I'] = [np.abs(temp['p_b_real'][0] - temp['p_b_pred_I'][0])]

                ### calculate the error from the mean of 80 precent
                temp['p_a_err_real_mean80'] = [np.abs(temp['p_a_real'][0] - p_a_80)]
                temp['p_b_err_real_mean80'] = [np.abs(temp['p_b_real'][0] - p_b_80)]

                prediction_errors = pd.concat([prediction_errors,pd.DataFrame(temp)], axis = 0)

    prediction_errors.set_index('id', inplace=True)
    prediction_errors.to_csv('data/calc_U/cross_val_prediction_errors_%s_%d.csv' % (control_str, i))#index=False)

    print('before saving pkl')
    pickle.dump(all_data, open('data/calc_U/all_data%s_%d.pkl' % (control_str, i), 'wb'))
    pickle.dump(q_info, open('data/calc_U/q_info%s_%d.pkl' % (control_str, i), 'wb'))

    # df_H_all.to_csv('data/calc_U/df_H%s.csv' % control_str)


def plot_errors(df):
    '''Boxplot of the errors per question type.'''
    ### list of the columns of the errors
    err_cl = list(df.columns[df.columns.str.contains('err')])

    ### take only the columns with the errors and question number
    df1 = df[err_cl + ['qn']]

    ### melt the data frame to: qn, err_type, err_value
    df2 = pd.melt(df1, id_vars=['qn'], value_vars=err_cl, var_name='err_type', value_name='err_value')

    ### boxplot of err to qn by err_type
    g = sns.factorplot(x="qn", hue='err_type', y="err_value", data=df2, kind="box", size=4, aspect=2)
    g.fig.suptitle('prediction error as function of which question was third\n by probability by prediction type (U, I, previous)')
    g.savefig('data/calc_U/err_boxplot_per_qn_per_prob.png')

    g1 = sns.factorplot(x="err_type", y="err_value", data=df2, kind="box", size=4, aspect=2)
    g1.set_xticklabels(rotation=45)
    g1.savefig('data/calc_U/err_boxplot_across_all_questions.png')

    df3 = df2.copy()
    df3['prob'] = 0
    df3['prob'][df3['err_type'].str.contains('p_b')] = 1 # p_a --> 0
    df3['err_type'][df3['err_type'].str.contains('_I')] = 'I'
    df3['err_type'][df3['err_type'].str.contains('_U')] = 'U'
    df3['err_type'][df3['err_type'].str.contains('_pre')] = 'pre'
    df3['err'] = pd.Categorical(df3['err_type'], categories=df3['err_type'].unique()).codes
    df3.to_csv('data/calc_U/00pred_err_per_prediction_type.csv')#index=False)

    ### group per probability a/b.
    gg = df3.groupby(['prob'])
    ### group per probability a/b per question.
    gg1 = df3.groupby(['prob','qn'])
    ### run on all combinations of prob and q.
    for p, q in product((0,1),(2,3,4,5)):
        gg.get_group(p).to_csv('data/calc_U/00pred_err_per_prob_%d.csv' % p)
        gg1.get_group((p, q)).to_csv('data/calc_U/00pred_err_per_prob_%d_per_qn_%d.csv' %(p,q))

    g2 = sns.factorplot(x="err_type", y="err_value", data=df3, kind="box", size=4, aspect=2)
    g2.set_xticklabels(rotation=45)
    g2.savefig('data/calc_U/err_boxplot_across_all_questions_and_probs.png')

    df4 = df3.pivot(columns='err_type', values='err_value')
    df5 = pd.DataFrame()
    df5['U'] = df4['U'].dropna().reset_index(drop=True)
    df5['I'] = df4['I'].dropna().reset_index(drop=True)
    df5['pre'] = df4['pre'].dropna().reset_index(drop=True)

    df5.to_csv('data/calc_U/00cross_val_prediction_errors_per_prediction_type4anova.csv')#index=False)('data/calc_U/cross_val_prediction_errors_per_prediction_type.csv')#index=False)

def average_results(h_mix_type, use_U, use_neutral, with_mixing, num_of_repeats):
    '''Combine all the data from num_of_repeats cross validations to one dataframe'''
    control_str = '_U_%s_mixing_%s_neutral_%s_mix_type_%d' % (use_U, with_mixing, use_neutral, h_mix_type)
    df_mean = pd.DataFrame()

    ### adding one dataframe at a time.
    for i in range(num_of_repeats):
        prediction_errors = pd.read_csv('data/calc_U/cross_val_prediction_errors_%s_%d.csv' % (control_str, i))
        df_mean = pd.concat((df_mean,prediction_errors), axis = 0)

    df_mean.to_csv('data/calc_U/00cross_val_prediction_errors_sum.csv')#index=False)

    for i, g in df_mean.groupby('qn'):
        g.to_csv('data/calc_U/00cross_val_prediction_errors_qn_%d.csv' % (i))#index=False)

    return df_mean

def main():
    h_type = [0]
    use_U_l = [True]
    use_neutral_l = [False]
    with_mixing_l = [True]
    comb = product(h_type, use_U_l, use_neutral_l, with_mixing_l)

    calcU = True
    # calcU = False

    ### How many times to repeat the cross validation
    num_of_repeats = 10
    if calcU:

        for h_mix_type, use_U, use_neutral, with_mixing in comb:

            print('Running:\tUse_U = {} |\tUse_Neutral = {} |\tWith_Mixing = {} |\th_mix_type = {}'.format(use_U,use_neutral,with_mixing, h_mix_type))

            control_str = 'pred_df_U_%s_mixing_%s_neutral_%s_mix_type_%d.csv' % (use_U, with_mixing, use_neutral, h_mix_type)
            # if os.path.isfile('./data/' + control_str):
            #     print('Already calculated everything for this combination')
            #     continue

            # for i in range(num_of_repeats):
            #     print('Performing cross validation number %d / %d'%(i, num_of_repeats))
            #     calculate_all_data_cross_val(use_U=use_U, use_neutral=use_neutral, with_mixing=with_mixing, h_mix_type=h_mix_type, i = i)

            calculate_all_data_cross_val_kfold(use_U=use_U, use_neutral=use_neutral, with_mixing=with_mixing, h_mix_type=h_mix_type)
    else:
        i = 0
        for h_mix_type, use_U, use_neutral, with_mixing in comb:
            # control_str = '_U_%s_mixing_%s_neutral_%s_mix_type_%d' % (use_U, with_mixing, use_neutral, h_mix_type)
            # prediction_errors = pd.read_csv('data/calc_U/cross_val_prediction_errors_%s_%d.csv' % (control_str,i))

            prediction_errors = average_results(h_mix_type, use_U, use_neutral, with_mixing, num_of_repeats)

            plot_errors(prediction_errors)

            plt.show()

if __name__ == '__main__':
    main()
