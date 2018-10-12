from general_quantum_operators import *
from copy import deepcopy
from scipy.optimize import minimize


def fun_to_minimize(h_, real_p_, psi_0, all_h, all_q, all_P, n_qubits=2):
    # all_h = ['x', h_b, h_ab], [h_a, None, h_ab], [h_a, h_b, None]
    # all_q = [q1, q2] = [0,3] --> AD
    # all_P = '0' --> P_q1, '1' --> P_q2, 'C' --> P_q1 * P_q2, 'D' --> P_q1 + P_q2 - P_q1 * P_q2

    full_h = [h_[0] if type(v) is type('x') else v for v in all_h] # replace the None with the minimization parameter
    p_ = get_general_p(full_h, all_q, all_P, psi_0, n_qubits)
    err_ = rmse(p_, real_p_)
    return err_


def fun_to_minimize_grandH(x_, all_q, all_data):
    grand_U = U_from_H(grandH_from_x(x_))

    err_ = []
    for data in all_data.values():
        psi_0 = np.dot(grand_U, data[1]['psi'])

        h_a = data['h_q'][str(all_q[0])]
        p_a_calc = get_general_p(full_h=[h_a, None, None],
                                 all_q=all_q,
                                 all_P='0', psi_0=psi_0, n_qubits=4)
        p_a = data[2]['p_a']
        err_.append((p_a_calc - p_a) ** 2)

        h_b = data['h_q'][str(all_q[1])]
        p_b_calc = get_general_p(full_h=[None, h_b, None],
                                 all_q=all_q,
                                 all_P='1', psi_0=psi_0, n_qubits=4)
        p_b = data[2]['p_b']
        err_.append((p_b_calc - p_b) ** 2)

    return np.sqrt(np.mean(err_))


def general_minimize(f, args_, x_0):
    min_err = 100.0
    best_result = None
    for i in range(10):
        x_0_rand = np.random.random(x_0.shape) * 2.0 - 1.0
        res_temp = minimize(f, x_0_rand, args=args_, method='SLSQP', bounds=None, options={'disp': False})
        if res_temp.fun < min_err:
            min_err = res_temp.fun
            best_result = deepcopy(res_temp)

    return best_result
