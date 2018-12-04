import pandas as pd
import numpy as np
from scipy import stats

import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
import itertools
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_context('paper')

df = pd.read_csv('data_all/new_dataframe.csv', index_col=0)
all_pred_df = pd.read_csv('analysis/all_predictions.csv')
all_pred_err_df = pd.read_csv('analysis/all_predictions_errros.csv')
stats_sig_all_user = pd.read_csv('analysis/all_predictions_stats_user.csv')
stats_sig_combined = pd.read_csv('analysis/all_predictions_stats_combined.csv')

print()


### remove question 2
df = all_pred_err_df.drop(all_pred_err_df.columns[all_pred_err_df.columns.str.contains('2')], axis = 1)
df = df.drop('userID', axis = 1)
### Change the shape of the dataframe
df_anova = pd.melt(df, id_vars=['UMNh'])
### get rid of uniform and mean
df_anova = df_anova[(df_anova['UMNh'] != 'mean') * (df_anova['UMNh'] != 'uniform')]
### split the question and prob strin to different columns
df_anova[['q', 'prob']] = df_anova['variable'].str.split('_', n = 2, expand = True)[[0,1]]
df_anova['q'] = df_anova['q'].str.split('q', expand = True)[1]
df_anova['prob'] = df_anova['prob'].str.split('p', expand=True)[1]

df_anova = df_anova.assign(u = np.nan, m = np.nan, n = np.nan, h = np.nan)

### split umnh to u, m, n ,h
for umnh in df_anova['UMNh'].unique():
    x = df_anova[df_anova['UMNh'] == umnh]
    y = y = np.tile(np.array(list(umnh), dtype = int), (x.shape[0],1))
    df_anova.loc[df_anova['UMNh'] == umnh, ['u', 'm', 'n', 'h']] = y

### get rid of unwanted columns
df_anova = df_anova.drop(['UMNh', 'variable'], axis = 1)
df_anova['err'] = df_anova.pop('value')

df_anova.loc[:,df_anova.columns[df_anova.columns != 'prob']] = df_anova.loc[:,df_anova.columns[df_anova.columns != 'prob']].astype('float32')
# df_anova.loc[:,df_anova.columns[df_anova.columns == 'prob']] = df_anova.loc[:,df_anova.columns[df_anova.columns == 'prob']].astype('category')
df_anova.loc[:,df_anova.columns[df_anova.columns == 'prob']] = df_anova.loc[:,df_anova.columns[df_anova.columns == 'prob']].astype('str')

### perform anova
no_interaction = 'err ~ C(u) + C(m) + C(n) + C(h) + C(q) + C(prob)'
with_interactions = 'err ~ C(u) * C(m) * C(n) * C(h) * C(q) * C(prob)'
formula = 'err ~ u + m + n + h + q + prob'

model = ols(formula, data = df_anova).fit()
aov_table = sm.stats.anova_lm(model, typ=2) # Type 2 ANOVA DataFrame
### effect size (R^2) (R_squared)
esq_sm = aov_table['sum_sq'][:-1]/(aov_table['sum_sq'][:-1]+aov_table['sum_sq'][-1])

aov_table.loc[:,'r_sq'] = esq_sm

print(aov_table)

# for name, row in aov_table.iterrows():
#     if row['PR(>F)'] < 5e-2:
#         mc = MultiComparison(df_anova[name], df_anova['err'])
#         mc_results = mc.tukeyhsd()
#         print('post_hoc')
#         print(mc_results)

reg_model = ols(formula, data = df_anova).fit()

params = ['u', 'm', 'n', 'h', 'q']
combinations = []

### create all possible combinations between u, m, n, h, q
for r in range(1, len(params) + 1):
    combinations += list(itertools.combinations(params, r))

### create all possible formulas from the combinations
formulas = []
for comb in combinations:
    formula = 'err' + '~'
    for c in comb:
         formula += '+' + c
    formulas += [formula]

### making dataframes for the different probabilities
df_a = df_anova[df_anova['prob'] == 'a']
df_b = df_anova[df_anova['prob'] == 'b']
df_ab = df_anova[df_anova['prob'] == 'ab']
dfs = {'a':df_a, 'b':df_b, 'ab':df_ab}

### calculating the regression model for each combinations for each probability
for formula in formulas:
    for prob, current_df in dfs.items():
        reg_model = ols(formula, data = current_df).fit()
        # reg_model.summary()
        temp = pd.DataFrame.from_dict({'prob': [prob], 'formula': [formula], 'R_sq': [reg_model.rsquared], 'p_value': [reg_model.f_pvalue]})
        if 'df_reg_models' not in locals():
            df_reg_models = temp.copy()
        else:
            df_reg_models = df_reg_models.append(temp)

df_reg_models = df_reg_models.reset_index(drop=True)
df_reg_models = df_reg_models.sort_values(by = 'R_sq')

df_anova.to_csv('analysis/anova_df.csv')
aov_table.to_csv('analysis/anova_results.csv')
df_reg_models.to_csv('analysis/reg_results.csv')

for prob, current_df in dfs.items():
    fig, ax = plt.subplots(1,1)
    cdf = df_reg_models[df_reg_models['prob'] == prob]
    g = sns.barplot(x = 'formula', y = 'R_sq', data = cdf, ax=ax)
    g.set_xticklabels(g.get_xticklabels(), rotation=30)
    g.set_title(r"$R^2$"" of {} as function of combination".format(prob))
    fig.savefig('r_sq_p{}.png'.format(prob), dpi = 300)

plt.show()