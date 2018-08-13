import pandas as pd
import numpy as np
from qutip import *
from tools import *
from numerical_quantum_coefficients import *
from prediction import *



def main():
    # df = pd.read_csv('data/new_dataframe.csv', index_col=0) # todo there is a problem with the DF i need to check what. something with the pos and qn.
    # # # normalizing the probabilities for quantum calculations.
    # # df.p1 = df.p1/4.
    # # df.p2 = df.p2/4.
    # # df.p12 = df.p12/4.
    #
    # df = numerical_quantum_coefficients(df)   # get the a_ij
    # df.to_csv('data/with_coef.csv')

    df = pd.read_csv('data/with_coef.csv', index_col=0)
    print(df.shape)

    user_same_q_list = {}
    for qn in df[(df.qn == 2.)].pos.unique():
        user_same_q_temp = df[(df.pos == 2.) & (df.qn == qn)] #
        # user_same_q_list.append(user_same_q_temp)
        user_same_q_list[qn] = user_same_q_temp

    # user_same_q = df[(df.qn == 3.) & (df.pos == 2.)] # just for checking
    for qn, user_same_q in user_same_q_list.items(  ):
        print('Prediction for position = 2. and question number  = %1.f'%(qn))

        current_fallacy = user_same_q.fal.tolist()[0]

        user_same_q = user_same_q.loc[~user_same_q.a00.isna(), :] # Using only the questions that we were able to calculate a00
        n_user = len(user_same_q)
        n_train = int(0.9 * n_user)
        user_rand_order = np.random.permutation(np.arange(n_user))
        user_rand_order = np.array(user_same_q.user.tolist())[user_rand_order]
        user_train = user_rand_order[:n_train]
        user_test = user_rand_order[n_train:]

        psi_ijkl_list, q_mn_list, psi_mn_list, user_without_nan_train = make_users_qvariables(df, user_train)

        # res = minimize(fun_to_min, x_eye, method='SLSQP', tol=1e-6, args=(psi_ijkl_list, q_mn_list, psi_mn_list))
        res = minimize(fun_to_min_list, x_eye, method='SLSQP', tol=1e-6, args=(psi_ijkl_list, q_mn_list, psi_mn_list))

        final_x = res.x
        final_U = get_unitary(final_x)
        check_unitary = final_U * final_U.dag()

        psi_ijkl_list_test, q_mn_list_test, psi_mn_list_test, user_without_nan_test  = make_users_qvariables(df, user_test)

        pi_tilde = calc_prob([0], current_fallacy, final_U, psi_ijkl_list_test, q_mn_list[0])
        pj_tilde = calc_prob([1], current_fallacy, final_U, psi_ijkl_list_test, q_mn_list[0])
        pij_tilde = calc_prob([1, 2], current_fallacy, final_U, psi_ijkl_list_test, q_mn_list[0])

        user_same_q_test = user_same_q[user_same_q.user.isin(user_without_nan_test)]

        probs = np.array([pi_tilde, pj_tilde, pij_tilde]).T
        probs_df = pd.DataFrame(data=probs, columns=['p1t', 'p2t', 'p12t'], index=user_same_q_test.index)
        user_same_q_test = pd.concat([user_same_q_test, probs_df], axis=1)

        dist_p1  = np.abs(user_same_q_test['p1'] - user_same_q_test['p1t'])
        dist_p2  = np.abs(user_same_q_test['p2'] - user_same_q_test['p2t'])
        dist_p12 = np.abs(user_same_q_test['p12'] - user_same_q_test['p12t'])
        user_same_q_test.to_csv('user_same_q_test' + str(int(qn)) + '.csv')
        print('mean over all users: %.2f' % (np.mean(dist_p1)))
        print('mean over all users: %.2f' % (np.mean(dist_p2)))
        print('mean over all users: %.2f' % (np.mean(dist_p12)))
        print('func_value = ', res.fun)
        # print(final_U)
        # print(check_unitary)







    # for row in range(df.shape[0]):
    #     qi, qj = df.loc[row, ['q1', 'q2']]
    #     psi_ij = df.loc[row, ['a00', 'a01', 'a10', 'a11']].values
    #     [qk, ql], _ = unasked_qubits(df.loc[row, ['q1', 'q2']])
    #     # rho_il = psi_ijkl.ptrace([qi, ql])
    #     pprint(1)

if __name__ == '__main__':
    main()
