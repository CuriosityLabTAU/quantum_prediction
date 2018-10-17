import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

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

def load_data(umn = ['True', 'True', 'False']):
    df = pd.read_csv('data/new_dataframe.csv', index_col = 0)

    temp = ('data/pred_df_U_%s_mixing_%s_neutral_%s.csv') % (umn[0],umn[1],umn[2])

    pred_df = pd.read_csv(temp)
    pred_df = pred_df.rename(columns = {'Unnamed: 0': 'userID'})
    return df, pred_df

def predicitions_stats(df, pred_df):
    user_same_q = {}

    # creating a dictionary of the users that had in the same position the same question
    for qn in df[(df['qn'] == 2.)]['pos'].unique():
        for p in df[(df['pos'] == 2.)]['qn'].unique():
            user_same_q_temp = df[(df.pos == p) & (df.qn == qn)]['userID']
            user_same_q[('q%d_p%d') % (qn,p)] = user_same_q_temp
            # print(qn,p,user_same_q_temp.__len__())

    # are the predictions statistically differ from the real probabilities.
    predictions = pd.DataFrame.from_dict({'combination':[], 'prob': [], 's':[], 'pvalue':[]})
    for k,v in user_same_q.items():
        # predictions[k] = {}
        pos = k.split('_')[1][1]
        cpdf = pred_df[pred_df['userID'].isin(v)] # dataframe of the users with the same question number and position
        pr = ['real', 'pred']
        probs = ['a', 'b', 'ab']
        for p in probs:
            prob = ('p%s') % (p)
            preal = ('q%s_%s_%s') %(pos, pr[0], prob)
            ppred = ('q%s_%s_%s') %(pos, pr[1], prob)
            # s,pv, ttest = ttest_or_mannwhitney(cpdf[preal], cpdf[ppred])
            # predictions[k][prob] = {'s':s, 'pvalue':pv, 'is_ttest': ttest}

            s,pv = stats.wilcoxon(cpdf[preal] - cpdf[ppred])
            predictions = predictions.append(pd.DataFrame.from_dict({'combination':[k], 'prob': [prob], 's':[s], 'pvalue':[pv]}))

    predictions = predictions.reset_index(drop=True)
    return predictions

df, pred_df = load_data()
predictions = predicitions_stats(df, pred_df)
predictions.to_csv('/data/predictions_stats.csv')

print('finished evaluating the predictions')
