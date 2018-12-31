import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
from itertools import product

sns.set_context('notebook')

def ttest_or_mannwhitney(y1,y2):
    '''
    Check if y1 and y2 stand the assumptions for ttest and if not preform mannwhitney
    :param y1: 1st sample
    :param y2: 2nd sample
    :return: s, pvalue, ttest - True/False
    '''
    ttest = False

    # assumptions for t-test
    # https://pythonfordatascience.org/independent-t-test-python/#t_test-assumptions
    ns1, np1 = stats.shapiro(y1)  # test normality of the data
    ns2, np2 = stats.shapiro(y2)  # test noramlity of the data
    ls, lp = stats.levene(y1, y2)  # test that the variance behave the same
    if (lp > .05) & (np1 > .05) & (np2 > .05):
        ttest = True
        s, p = stats.ttest_ind(y1, y2)
    else:
        s, p = stats.mannwhitneyu(y1, y2)

    return s, p, ttest

def load_data(umn = ['True', 'True', 'False', 0], path = 'data/'):
    df = pd.read_csv('data/new_dataframe.csv', index_col = 0)

    temp = (path + 'pred_df_U_%s_mixing_%s_neutral_%s_mix_type_%d.csv') % (umn[0],umn[1],umn[2], umn[3])

    pred_df = pd.read_csv(temp)
    pred_df = pred_df.rename(columns = {'Unnamed: 0': 'userID'})
    return df, pred_df

def calculate_err(df):
    '''
    Calculate error of all the probabilities in the dataframe.
    And add them to the dataframe
    '''
    ps = [2,3,4,5]
    pr = ['real', 'pred']
    probs = ['a', 'b', 'ab']
    for pos in ps:
        for p in probs:
            prob = ('p%s') % (p)
            preal = ('q%s_%s_%s') % (pos, pr[0], prob)
            ppred = ('q%s_%s_%s') % (pos, pr[1], prob)

            pred_err = np.sqrt((df[preal] - df[ppred])**2)
            s = ('q%s_%s_err') % (pos, prob)
            df[s] = pred_err

            s1 = ('q%s_%s_mean_err') % (pos, prob)
            s2 = ('q%s_%s_uniform_err') % (pos, prob)
            df[s1] = np.sqrt((df[preal].mean() - df[preal]) ** 2)
            df[s2] = np.sqrt((0.5 - df[preal]) ** 2)

    return df

def find_users_with_same_order(df):
    '''Find all the users that had the same order
     of questions and return it as a dictionary.
     e.g: user_same_q{'q5_p2'} = [list of the users that had question5 in position 2]
    '''
    user_same_q = {}

    # creating a dictionary of the users that had in the same position the same question
    for qn in df[(df['qn'] == 2.)]['pos'].unique():
        for p in df[(df['pos'] == 2.)]['qn'].unique():
            user_same_q_temp = df[(df.pos == p) & (df.qn == qn)]['userID']
            user_same_q[('q%d_p%d') % (qn,p)] = user_same_q_temp
    return user_same_q

def resahpe_all_pred(df):
    '''Take the mean and uniform errors and put the under UMN'''
    cnames = df.columns[df.columns.str.contains('err')].tolist() + ['userID', 'UMNh']
    df1 = df[cnames]
    drop_names = df.columns[(df.columns.str.contains('mean')) | df.columns.str.contains('uniform')].tolist()
    df1 = df1.drop(drop_names, axis=1)

    for c in ['mean', 'uniform']:

        # a = df[df['UMN'] == '000']
        mnames = df.columns[df.columns.str.contains(c)]
        b = df[df['UMNh'] == '0001'][mnames.tolist()+['userID','UMNh']]
        d = dict(zip(b[mnames].columns, mnames.str.replace('_'+c,'')))
        b = b.rename(columns = d)
        b['UMNh'] = c
        df1 = df1.append(b)

    return df1


def predicitions_stats(df, pred_df):
    pred_df = calculate_err(pred_df)
    user_same_q = find_users_with_same_order(df)

    # are the predictions statistically differ from the real probabilities.
    predictions = pd.DataFrame.from_dict({'combination':[], 'prob': [], 's':[], 'pvalue':[]})
    for k,v in user_same_q.items():
        # predictions[k] = {}
        pos = k.split('_')[1][1]
        cpdf = pred_df[pred_df['userID'].isin(v)] # dataframe of the users with the same question number and position

        probs = ['a', 'b', 'ab']
        for p in probs:
            prob = ('p%s') % (p)
            cerr = ('q%s_%s_err') % (pos, prob)
            pred_err = cpdf[cerr]
            s,pv = stats.wilcoxon(pred_err)
            predictions = predictions.append(pd.DataFrame.from_dict({'combination':[k], 'prob': [prob], 's':[s], 'pvalue':[pv]}))

    predictions = predictions.reset_index(drop=True)
    return predictions


def plot_err(df):
    '''plot all the erros'''

    ps = [2, 3, 4, 5]
    probs = ['a', 'b', 'ab']
    for pos in ps:
        for p in probs:
            prob = ('p%s') % (p)
            cerr = ('q%s_%s_err') % (pos, prob)
            fig, cax = plt.subplots(1,1)
            sns.boxplot(y = cerr, x = 'UMNh', data = df, ax=cax)


def plot_err_avg(df):
    '''Plotting the average of the probabilities errors'''
    probs = ['a', 'b', 'ab']
    df1 = pd.DataFrame()

    df1['UMNh'] = pd.melt(df, id_vars=['UMNh'], value_vars=list(df.columns[df.columns.str.contains('pab_')]))['UMNh']

    for p in probs:
        prob = ('p%s_') % (p)
        df1[prob] = pd.melt(df, id_vars=['UMNh'], value_vars=list(df.columns[df.columns.str.contains(prob)]))['value']
        ######### plot average per probability
        fig, cax = plt.subplots(1, 1)
        sns.boxplot(y=prob, x='UMNh', data=df1, ax = cax)

    ######## plot average per UMNh combination
    fig, cax = plt.subplots(1, 1)
    df2 = pd.melt(df1, id_vars=['UMNh'], value_vars=list(df1.columns[df1.columns.str.contains('p')]))
    sns.boxplot(y = 'value', x = 'UMNh', data = df2, ax = cax)
    cax.set_ylabel('Total average error')


def stats_all(all_pred_err_df, df):
    '''dataframe contains all the errors data.
    Return all the wilcoxon for all questions, positions, and UMN'''
    user_same_q = find_users_with_same_order(df)
    umn_combinations = all_pred_err_df['UMNh'].unique()

    predictions = pd.DataFrame.from_dict({'q':[], 'pos':[], 'umn1':[], 'umn2':[], 'mean1':[], 'mean2':[], 'prob': [], 's':[], 'pvalue':[]})
    predictions_all = pd.DataFrame.from_dict({'umn1':[], 'umn2':[], 'mean1':[], 'mean2':[], 's':[], 'pvalue':[]})
    for k,v in user_same_q.items(): # k = qn_pn
        # predictions[k] = {}
        pos = k.split('_')[1][1]
        qn = k.split('_')[0][1]
        cpdf = all_pred_err_df[all_pred_err_df['userID'].isin(v)]

        probs = ['a', 'b', 'ab']
        for p in probs:
            prob = ('p%s') % (p)
            cerr = ('q%s_%s_err') % (pos, prob)
            for umn1 in umn_combinations:
                for umn2 in umn_combinations:
                    pred_err1 = cpdf[cpdf['UMNh']==umn1][cerr]
                    pred_err2 = cpdf[cpdf['UMNh']==umn2][cerr]
                    # todo: add std columns
                    s, pv = stats.wilcoxon(pred_err1, pred_err2)
                    temp_dict = {'q': [qn],
                                 'pos': [pos],
                                 'umn1': [umn1],
                                 'umn2': [umn2],
                                 'mean1': [pred_err1.mean()],
                                 'mean2': [pred_err2.mean()],
                                 'prob': [prob],
                                 's': [s],
                                 'pvalue': [pv]}

                    predictions = predictions.append(pd.DataFrame.from_dict(temp_dict))

    # difference between UMN combination in total
    # (not per users with the same questions order)
    for umn1 in umn_combinations:
        for umn2 in umn_combinations:
            pred_err1 = all_pred_err_df[all_pred_err_df['UMNh'] == umn1].drop(['UMNh', 'userID'], axis = 1)
            pred_err2 = all_pred_err_df[all_pred_err_df['UMNh'] == umn2].drop(['UMNh', 'userID'], axis = 1)
            s, pv = stats.wilcoxon(pred_err1.mean(), pred_err2.mean())
            temp_dict_all = {'umn1': [umn1],
                             'umn2': [umn2],
                             'mean1': [pred_err1.mean().mean()],
                             'mean2': [pred_err2.mean().mean()],
                             'std1': [pred_err1.mean().std()],
                             'std2': [pred_err2.mean().std()],
                             's': [s],
                             'pvalue': [pv]}
            predictions_all = predictions_all.append(pd.DataFrame.from_dict(temp_dict_all))


    predictions = predictions.reset_index(drop=True)
    predictions_all = predictions_all.reset_index(drop=True)
    return predictions, predictions_all

def create_all_data(path):
    '''load and build the '''
    h_type = [0,1,2,3,4]
    use_U_l = [True, False]
    use_neutral_l = [False, True]
    with_mixing_l = [True, False]

    comb = product(h_type, use_U_l, use_neutral_l, with_mixing_l)
    # Loop to run all controls, except the uniform or the mean
    for h, u, n, m in comb:
        if not os.path.isfile(path +'pred_df_U_%s_mixing_%s_neutral_%s_mix_type_%d.csv' % (u,m,n,h)):
            continue
        df, pred_df = load_data([u,m,n,h], path)
        temp_all_pred_df = pred_df.copy()
        temp_all_pred_df['U'] = u
        temp_all_pred_df['mix'] = m
        temp_all_pred_df['neutral'] = n
        temp_all_pred_df['htype'] = h

        if 'all_pred_df' in locals():
            all_pred_df = all_pred_df.append(temp_all_pred_df)
        else:
            all_pred_df = temp_all_pred_df.copy()

    all_pred_df = all_pred_df.reset_index(drop=True)

    temp_c = all_pred_df[['U', 'mix', 'neutral', 'htype']].astype(int).astype(str)
    all_pred_df['UMNh'] = temp_c['U'] + temp_c['mix'] + temp_c['neutral'] + temp_c['htype']
    all_pred_df = calculate_err(all_pred_df)
    all_pred_df1 = resahpe_all_pred(all_pred_df) # contains only the errors, add mean and uniform

    return all_pred_df, all_pred_df1, df, pred_df

def prob_dist(all_pred_df):
    '''To see the distibution of all the probabilities'''
    a = all_pred_df.mean(axis=0)
    a = a.drop(a.keys()[a.keys().str.contains('err')])
    a = a.drop(['userID', 'mix', 'neutral', 'htype', 'UMNh','U'])

    a   = a.loc[a.index[~a.index.str.contains('pred')]]
    # a = pd.DataFrame(a, columns = ['real_prob'])
    # a = a.assign(prob = np.nan)
    # a.loc[a.index[a.index.str.contains('_pa') * ~a.index.str.contains('_pab')], 'prob'] = 'a'
    # a.loc[a.index[a.index.str.contains('_pb') * ~a.index.str.contains('_pab')], 'prob'] = 'b'
    # a.loc[a.index[a.index.str.contains('_pab')], 'prob'] = 'ab'
    #
    # targets = [a.loc[a['prob'] == val] for val in a['prob'].unique()]
    #
    # for target in targets:
    #     sns.distplot(target[['real_prob']], kde=False)

    # sns.plt.show()
    c = {}
    b = all_pred_df[all_pred_df.columns[all_pred_df.columns.str.contains('real')]]
    c['pa_real']  = b.loc[:, b.columns[b.columns.str.contains('_pa') * ~b.columns.str.contains('_pab')]].values.flatten()
    c['pb_real']  = b.loc[:, b.columns[b.columns.str.contains('_pb') * ~b.columns.str.contains('_pab')]].values.flatten()
    c['pab_real'] = b.loc[:, b.columns[b.columns.str.contains('_pab')]].values.flatten()
    c = pd.DataFrame.from_dict(c)

    c.hist()

def main():
    # prepare_data, infer_plots = True, False
    prepare_data, infer_plots = False, True
    if prepare_data:
        # data_path = 'data_all/'
        data_path = 'data/'
        all_pred_df, all_pred_err_df, df, pred_df = create_all_data(data_path) # load all the data with combination (UMN) column
        stats_sig_all_user, stats_sig_combined =  stats_all(all_pred_err_df, df) # calculate all wilcoxon between all the UMN combination per question number and position.

        all_pred_df.to_csv('analysis/all_predictions.csv', index=False)
        all_pred_err_df.to_csv('analysis/all_predictions_errros.csv', index=False)
        stats_sig_all_user.to_csv('analysis/all_predictions_stats_user.csv', index=False)
        stats_sig_combined.to_csv('analysis/all_predictions_stats_combined.csv', index=False)

        print('Finished evaluating the predictions')
    else:
        all_pred_df        = pd.read_csv('analysis/all_predictions.csv')
        all_pred_err_df    = pd.read_csv('analysis/all_predictions_errros.csv')
        stats_sig_all_user = pd.read_csv('analysis/all_predictions_stats_user.csv')
        stats_sig_combined = pd.read_csv('analysis/all_predictions_stats_combined.csv')

    if infer_plots:
        plot_err(all_pred_err_df)
        # prob_dist(all_pred_df)
        # plot_err_avg(all_pred_err_df)
        print('Plotting ended')
        plt.show()

if __name__ == '__main__':
    main()

