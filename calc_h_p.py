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
ns = 10  # number of elements

### for probaiblities range
ap = np.linspace(s,e, ns)

### for {h} range
ah = np.linspace(-e,e, 2 * ns + 1)
### create all the combinations
b = product(ap,ap,ah,ah,ah)

### --> to np.array
c = np.array(list(b))

### --> to pandas dataframe
df = pd.DataFrame(data = c, columns = ['pa','pb','ha','hb','hab'])


### init psi
dim_ = 16
psi0 = np.ones([dim_,1]) / np.sqrt(dim_)


t0 = time.time()
hp.get_general_p([df['ha'][0], df['hb'][0], df['hab'][0]], [0, 1], 'C', psi0)
print('total running time is about ~ %.1f hours' % ((time.time() - t0) * (10**2 * 21**3 / 3600)))

### calculate the proability of the conjunction based on all {hi} and the possible probs.
df['pab'] = df.apply(lambda x: hp.get_general_p([x['ha'], x['hb'], x['hab']], [0,1], 'C', psi0)[0][0], axis = 1)

df['irr'] = df.apply(lambda x: x['pab'] - x[['pa','pb']].min(), axis = 1)

df.to_csv('p_h_analysis.csv')
print('======= Finished running at: %s =======' % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))