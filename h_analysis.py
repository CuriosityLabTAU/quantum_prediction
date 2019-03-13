import pandas as pd
import seaborn as sns
from scipy import stats

def calculate_corr_with_pvalues(df, method = 'pearsonr'):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')

    rho = df.corr()

    for r in df.columns:
        for c in df.columns:
            if method == 'pearsonr':
                pvalues[r][c] = round(stats.pearsonr(df[r], df[c])[1], 4)
            elif method == 'spearman':
                pvalues[r][c] = round(stats.spearmanr(df[r], df[c])[1], 4)
            elif method == 'reg':
                slope, intercept, rho[r][c], pvalues[r][c], std_err = stats.linregress(x=df[r],y= df[c])

    rho = rho.round(2)
    pval = pvalues
    # create three masks
    r1 = rho.applymap(lambda x: '{}*'.format(x))
    r2 = rho.applymap(lambda x: '{}**'.format(x))
    r3 = rho.applymap(lambda x: '{}***'.format(x))
    # apply them where appropriate
    rho = rho.mask(pval <= 0.05, r1)
    rho = rho.mask(pval <= 0.01, r2)
    rho = rho.mask(pval <= 0.001, r3)

    return pvalues, rho

df_H = pd.read_csv('data/df_H_U_True_mixing_True_neutral_False_mix_type_0.csv', index_col=0)
df_H.sort_index(inplace=True)

df = pd.read_csv('data/new_dataframe.csv', index_col=0)

### taking the data only until the 2nd question
df = df[df['pos'] <3 ]

### reindexing the rows to have the names as the users IDs like df_H
df = df.set_index('userID', drop = True)

### wanted columns
# cnames = {u'q1', u'q2', u'p1', u'p2', u'p12', u'fal', u'irr'}

### run on the first 2 questions
for p in range(2):
    cnames = {'p1': 'q%s_real_pa' % p, 'p2': 'q%s_real_pb' % p, 'p12': 'q%s_real_pab' % p, 'fal':'fal%s'%p, 'irr': 'irr%s'%p}
    ### choose the current question.
    d = df[df['pos'] == p]

    ### choose only the columns that I'm intrested in.
    d = d[cnames.keys()]

    d = d.rename(index=str, columns=cnames)
    d.sort_index(inplace=True)
    # d.index = df_H.index

    df_H = pd.concat((df_H,d), axis = 1)
print(df_H.columns)
print()