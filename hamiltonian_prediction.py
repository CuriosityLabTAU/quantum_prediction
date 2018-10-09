import numpy as np
from scipy.linalg import expm
import pandas as pd

def zero_H(n_qubits=2):
    dim_ = 2 ** n_qubits
    H_ = np.zeros([dim_, dim_])
    return H_

def param_H(h_):
    H_ = np.matrix([[1, h_], [h_, -1]])
    return H_


def param_Hmix(g_):
    H_ = (g_ / np.sqrt(2)) * np.matrix([[1, 1], [1, -1]])
    return H_


def U_from_H(H_):
    U_ = expm(-1j * np.pi / 2.0 * H_)
    return U_


def Projection(q, n_qubits=2):
    dim_ = 2 ** n_qubits
    d_ = np.zeros([dim_])
    for i in range(dim_):
        i_repr = np.binary_repr(i, width=n_qubits)
        print(i, i_repr, i_repr[q])
        if i_repr[q] == '1':
            d_[i] = 1
    P_ = np.diag(d_)
    return P_


def compose_H(H1_, H2_, Hmix_, n_qubits=2):
    dim_ = 2 ** n_qubits
    H_ = np.zeros([dim_, dim_])
    H_[:H1_.shape[0], :H1_.shape[1]] = H1_
    H_[H1_.shape[0]:, H1_.shape[1]:] = H2_
    H_[H1_.shape[0]:, :H1_.shape[1]] = Hmix_
    H_[:H1_.shape[0], H1_.shape[1]:] = np.transpose(Hmix_)

    return H_


def main():
    df = pd.read_csv('data/new_dataframe.csv', index_col=0)

    print(compose_H(param_H(0.5), param_H(-0.5), param_Hmix(0.2)))

main()