import pandas as pd
import numpy as np
from qutip import *
from tools import *

df = pd.read_csv('data/new_dataframe.csv', sep='\t')
print(df.head())

# def quantum_model(p1, p2, p12) --> a_ij
def quantum_coefficients(df):
    '''
    Calculating the coefficents based on the probabilities. (p1, p2, p12) --> a_ij
    :param df: dataframe with all the data.
    :return: df + a_ij
    '''
    # todo: sometimes a00 is nan because ( df.a10 ** 2 + df.a01 ** 2 + df.a11 ** 2) > 1

    df = df.assign(**{'a10': np.nan, 'a01': np.nan, 'a11': np.nan, 'a00': np.nan})

    # Conjunction
    df.loc[df.fal == 1., 'a10'] = np.sqrt(2 * df[df.fal == 1.].p1) - np.sqrt(df[df.fal == 1.].p12)
    df.loc[df.fal == 1., 'a01'] = np.sqrt(2 * df[df.fal == 1.].p2) - np.sqrt(df[df.fal == 1.].p12)
    df.loc[df.fal == 1., 'a11'] = np.sqrt(df[df.fal == 1.].p12)

    # Disjunction
    df.loc[df.fal == 2., 'a10'] = np.sqrt(3 * df[df.fal == 2.].p12) - np.sqrt(2 * df[df.fal == 2.].p2)
    df.loc[df.fal == 2., 'a01'] = np.sqrt(2 * df[df.fal == 2.].p2) - np.sqrt(2 * df[df.fal == 2.].p1)
    df.loc[df.fal == 2., 'a11'] = np.sqrt(2 * df[df.fal == 2.].p1) + np.sqrt(2 * df[df.fal == 2.].p2) - np.sqrt(
        3 * df[df.fal == 2.].p12)

    # a_00 from normalization
    df.loc[:, 'a00'] = np.sqrt(1 - df.a10 ** 2 - df.a01 ** 2 - df.a11 ** 2)
    return df

# def join 2+2=4 qubits (a_ij, a_kl) --> a_ijkl
def join(psi_ij, psi_kl):
    '''
    Taking to psi and return the thier tensor product. (a_ij, a_kl) --> a_ijkl
    :param psi_ij: psi in the form: np.array([a00, a01, a10, a11])
    :param psi_kl: psi in the form: np.array([a00, a01, a10, a11])
    :return:
    '''
    # a00, a01, a10, a11 =
    # temp_psi_ij = np.array([a00, a01, a10, a11])

    # Converting the psies to a QuTip object
    psi_ij = ndarray2Qobj(psi_ij)
    psi_kl = ndarray2Qobj(psi_kl)

    psi_ijkl = tensor(psi_ij, psi_kl)
    return psi_ijkl

# def trace (a_ijkl, q1, q2) [q1=2, q2=3] --> rho_il,il
# tracing psi_ijkl --> rho_il,il.
# note: psi_ijkl - Qobj from QuTip
#       In ptrace from QuTip the [i,l] you insert is the [i,l] you left with after the partial trace.
rho_il = psi_ijkl.ptrace([i,l])

# def perform_unitary (a_ijkl, U_ijkl,ijkl) --> a'_ijkl
# U_ijkl, a_ijkl - Qobj fron QuTip
a1_ijkl = U_ijkl * a_ijkl

# def new_two_qubit (a_ijkl, U_ijkl,ijkl, q1, q2) --> Tr_q1,q2 U*a --> rho_il,il [perform_unitary, trace]
(U_ijkl * a_ijkl).ptrace([i,l])

# def overlap (rho_il,il, a_il) --> p = <psi_il | rho_il | psi_il>
# rho, psi - Qobj fron QuTip
p = psi_il * rho_il * psi_il

def get_unitary(x): # --> U_ijkl,ijkl
    # x is a 256x1 np array
    M = np.reshape(x, int(np.sqrt(x.shape[0])), int(np.sqrt(x.shape[0])))
    [U, ~] = np.linalg.qr(M)
    return U


# def fun_to_minimize (x=U_ijkl,ijkl, param=a_ijkl, q1, q2, a_il) --> min (p - 1)^2
# given a U (this is the x to check):
#   a_ijkl - matrix 16 x N(users)
#   a_il - matrix 4 x N(users)
# for each user, compute error (p-1)^2
# return sum of errors over users
# in (Unitary), out (sum of errors over users)

# def find_optimal_u (x0=I) --> optimal U
# run over many U's and take the minimal error
