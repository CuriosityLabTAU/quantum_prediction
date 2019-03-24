import hamiltonian_prediction as hp

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import datetime

import time

print('======= Started running at: %s =======' % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

s = 0.01 # start
e = 1 # end
ns = 5  # number of elements

### for probaiblities range
ap = np.linspace(s,e, ns)

### for {h} range
ah = np.linspace(-e,e, 2 * ns + 1)
### create all the combinations
# b = product(ap,ap,ah,ah,ah)
b = product(ah,ah,ah)

### --> to np.array
c = np.array(list(b))
# c = np.array([[-0.75, -0.75, -0.75]])

### --> to pandas dataframe
# df = pd.DataFrame(data = c, columns = ['pa','pb','ha','hb','hab'])
df = pd.DataFrame(data = c, columns = ['ha','hb','hab'])


### init psi
psi0 = hp.uniform_psi(4, 'uniform')

t0 = time.time()
# hp.get_general_p([df['ha'][0], df['hb'][0], df['hab'][0]], [0, 1], 'C', psi0)
# print('total running time will be at max: %.1f hours' % ((time.time() - t0)))# * (10**2 * 21**3 / 3600)))

### calculate the proability of the conjunction based on all {hi} and the possible probs.
df['pa'] = df.apply(lambda x: hp.get_general_p([x['ha'], None, None], [0,1], '0', psi0)[0][0], axis = 1)
# df['pa_full'] = df.apply(lambda x: hp.get_general_p([x['ha'], x['hb'], x['hab']], [0,1], '0', psi0)[0][0], axis = 1)
# df['pb'] = df.apply(lambda x: hp.get_general_p([None, x['hb'], None], [0,1], '1', psi0)[0][0], axis = 1)
# df['pb_full'] = df.apply(lambda x: hp.get_general_p([x['ha'], x['hb'], x['hab']], [0,1], '1', psi0)[0][0], axis = 1)
df['pab_c'] = df.apply(lambda x: hp.get_general_p([None, None, x['hab']], [0,1], 'C', psi0)[0][0], axis = 1)
# df['pab_c_full'] = df.apply(lambda x: hp.get_general_p([x['ha'], x['hb'], x['hab']], [0,1], 'C', psi0)[0][0], axis = 1)
# df['pab_d'] = df.apply(lambda x: hp.get_general_p([None, None, x['hab']], [0,1], 'D', psi0)[0][0], axis = 1)
# df['pab_full_d'] = df.apply(lambda x: hp.get_general_p([x['ha'], x['hb'], x['hab']], [0,1], 'D', psi0)[0][0], axis = 1)


# df['irr'] = df.apply(lambda x: x['pab_c'] - x[['pa','pb']].min(), axis = 1)
# df['irr_full'] = df.apply(lambda x: x['pab_c_full'] - x[['pa_full','pb_full']].min(), axis = 1)

df.to_csv('p_h_analysis_v2.csv')
print('======= Finished running at: %s =======' % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

df.plot(y='pab_c', x='hab')
df.plot(y='pa', x='ha')
plt.show()