from general_quantum_operators import *


def fun_to_minimize(h_, real_p_, psi_0, all_h, all_q, all_P, n_qubits=2):
    # all_h = [None, h_b, h_ab], [h_a, None, h_ab], [h_a, h_b, None]
    # all_q = [q1, q2] = [0,3] --> AD
    # all_P = '0' --> P_q1, '1' --> P_q2, 'C' --> P_q1 * P_q2, 'D' --> P_q1 + P_q2 - P_q1 * P_q2

    full_h = [h_ if v is None else v for v in all_h] # replace the None with the minimization parameter
    H_ = compose_H(full_h, all_q)
    psi_dyn = get_psi(H_, psi_0)
    P_ = MultiProjection(all_P, all_q, n_qubits)
    psi_final = np.dot(P_, psi_dyn)
    p_ = norm_psi(psi_final)
    err_ = rmse(p_, real_p_)
    return err_