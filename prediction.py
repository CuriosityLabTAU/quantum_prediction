import pandas as pd
import numpy as np
from qutip import *
from tools import *
from numerical_quantum_coefficients import *

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
    Taking to psi and return their tensor product. (a_ij, a_kl) --> a_ijkl
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

# # def trace (a_ijkl, q1, q2) [q1=2, q2=3] --> rho_il,il
# # tracing psi_ijkl --> rho_il,il.
# # note: psi_ijkl - Qobj from QuTip
# #       In ptrace from QuTip the [i,l] you insert is the [i,l] you left with after the partial trace.
# rho_il = psi_ijkl.ptrace([i,l])
#
# # def perform_unitary (a_ijkl, U_ijkl,ijkl) --> a'_ijkl
# # U_ijkl, a_ijkl - Qobj fron QuTip
# a1_ijkl = U_ijkl * a_ijkl
#
# # def new_two_qubit (a_ijkl, U_ijkl,ijkl, q1, q2) --> Tr_q1,q2 U*a --> rho_il,il [perform_unitary, trace]
# (U_ijkl * a_ijkl).ptrace([i,l])
#
# # def overlap (rho_il,il, a_il) --> p = <psi_il | rho_il | psi_il>
# # rho, psi - Qobj fron QuTip
# p = psi_il * rho_il * psi_il


def calc_prob(qbt, fallacy, U, psi_ijkl_list, q_mn):
    '''
    Predict the probability
    :param qbt:
    :param fallacy:
    :param U:
    :param psi_ijkl:
    :return:
    '''

    prob_tilde = []
    post_state = None
    if len(qbt) == 1:
        if qbt[0] == 0:
            post_state = np.array([0, 1, 0, 1])
        elif qbt[0] == 1:
            post_state = np.array([0, 0, 1, 1])
    elif len(qbt) == 2:
        if fallacy == 1:
            post_state = np.array([0, 0, 0, 1])
        elif fallacy == 2:
            post_state = np.array([0, 1, 1, 1])
    if post_state is None:
        print('Error !!!!')
        return

    post_state = ndarray2Qobj(post_state, norm=True)
    plus_state = ndarray2Qobj(np.array([1, 1, 1, 1]), norm=True)

    # todo: add plus state to the other qubit


    for psi_ijkl in psi_ijkl_list:
        psi_ijkl_tilde = U * psi_ijkl
        rho_mn_tilde = psi_ijkl_tilde.ptrace([q_mn[0], q_mn[1]])
        temp_rho_mn_tilde = post_state.dag() * rho_mn_tilde * post_state
        p = plus_state.dag() * temp_rho_mn_tilde * plus_state
        prob_tilde.append(p[0][0][0].real)

    return prob_tilde


def get_unitary(x):     # --> U_ijkl,ijkl
    # x is a 256x1 np array
    M = np.reshape(x, [int(np.sqrt(x.shape[0])), int(np.sqrt(x.shape[0]))])
    [Q, _] = np.linalg.qr(M)
    Q = ndarray2Qobj(Q, typ='dm')
    return Q


# def fun_to_minimize (x=U_ijkl,ijkl, param=a_ijkl, q1, q2, a_il) --> min (p - 1)^2
# given a U (this is the x to check):
#   a_ijkl - matrix 16 x N(users)
#   a_il - matrix 4 x N(users)
# for each user, compute error (p-1)^2
# return sum of errors over users
# in (Unitary), out (sum of errors over users)

# def find_optimal_u (x0=I) --> optimal U
# run over many U's and take the minimal error


def fun_to_min(x, psi_ijkl, q_mn, psi_mn):
    U = get_unitary(x)

    psi_tag_ijkl = U * psi_ijkl
    rho_mn = psi_tag_ijkl.ptrace([q_mn[0], q_mn[1]])

    p = psi_mn.dag() * rho_mn * psi_mn  # overlap between them
    p = (1.0 - p[0][0][0].real) ** 2    # want to maximize overlap == minimize 1-overlap
    # print(p)
    return p


def fun_to_min_list(x, psi_ijkl, q_mn, psi_mn):
    U = get_unitary(x)

    p = []
    for i in range(len(psi_ijkl)):
        psi_tag_ijkl = U * psi_ijkl[i]
        rho_mn = psi_tag_ijkl.ptrace([q_mn[i][0], q_mn[i][1]])

        p_i = psi_mn[i].dag() * rho_mn * psi_mn[i]  # overlap between them
        p.append((1.0 - p_i[0][0][0].real) ** 2)    # want to maximize overlap == minimize 1-overlap

    mean_p = np.mean(p)
    # print(p)
    return mean_p

x_eye = np.eye(16, 16).reshape(256)
# x_eye = np.ones([16, 16]).reshape(256)
x_rand = np.random.rand(256,1)
U_rand = get_unitary(x_rand)


def make_users_qvariables(df, users2run):
    '''
    Return parameters to calculate probability
    :param df:
    :param users2run:
    :return: psi_ijkl_list, q_mn_list, psi_mn_list, user_without_nan
    '''
    psi_ijkl_list = []
    q_mn_list = []
    psi_mn_list = []
    user_without_nan = []
    for user in users2run:
        # get a_ij of q_pos_1
        a_ij = df.loc[(df.user == user) & (df.pos == 0), ['a00', 'a01', 'a10', 'a11']].values[0]
        if True in np.isnan(a_ij):
            continue

        # get a_kl of q_pos_2
        a_kl = df.loc[(df.user == user) & (df.pos == 1), ['a00', 'a01', 'a10', 'a11']].values[0]
        if True in np.isnan(a_kl):
            continue
        psi_ijkl = join(a_ij, a_kl)

        # get a_ij of q_pos_3
        q_mn = df.loc[(df.user == user) & (df.pos == 2), ['q1', 'q2']].values[0] - 1
        a_mn = df.loc[(df.user == user) & (df.pos == 2), ['a00', 'a01', 'a10', 'a11']].values[0]
        if True in np.isnan(a_mn):
            continue
        psi_mn = ndarray2Qobj(a_mn)

        psi_ijkl_list.append(psi_ijkl)
        q_mn_list.append(q_mn)
        psi_mn_list.append(psi_mn)
        user_without_nan.append(user)

    return psi_ijkl_list, q_mn_list, psi_mn_list, user_without_nan


def probs_quantum_prediction(current_fallacy, U, psi_ijkl_list_test, q_mn_list):
    '''
    Predict probabilities using U operator
    :param current_fallacy:
    :param U: the operator
    :param psi_ijkl_list_test: test list of the users.
    :param q_mn_list: which qubits we are using.
    :return: 3 probabilities: pA, pB, pAB
    '''
    pi_tilde  = calc_prob([0], current_fallacy, U, psi_ijkl_list_test, q_mn_list[0])
    pj_tilde  = calc_prob([1], current_fallacy, U, psi_ijkl_list_test, q_mn_list[0])
    pij_tilde = calc_prob([1, 2], current_fallacy, U, psi_ijkl_list_test, q_mn_list[0])

    return pi_tilde, pj_tilde, pij_tilde

def distance_calc(user_same_q_test, probs2compare = ['p1_U', 'p2_U', 'p12_U'], mean = False):
    '''
    Calculating distances between the predicted probabilities and the True (mean True)
    :param user_same_q_test: dataframe containing all the probabilities
    :param probs2compare: which probabilities to compare to. (List)
    :param mean: compare to mean of probability or to individual probability (True/ False)
    :return: dist_p1, dist_p2, dist_p12
    '''

    # mean over the difference between the predicted probability by UNITARY transformation and the true probability per participant
    if mean:
        dist_p1 = np.mean(np.abs(user_same_q_test['p1'].mean() - user_same_q_test['p1']))
        dist_p2 = np.mean(np.abs(user_same_q_test['p2'].mean() - user_same_q_test['p2']))
        dist_p12 = np.mean(np.abs(user_same_q_test['p12'].mean() - user_same_q_test['p12']))
    else:
        dist_p1 = np.mean(np.abs(user_same_q_test['p1'] - user_same_q_test[probs2compare[0]]))
        dist_p2 = np.mean(np.abs(user_same_q_test['p2'] - user_same_q_test[probs2compare[1]]))
        dist_p12 = np.mean(np.abs(user_same_q_test['p12'] - user_same_q_test[probs2compare[2]]))

    return dist_p1, dist_p2, dist_p12
