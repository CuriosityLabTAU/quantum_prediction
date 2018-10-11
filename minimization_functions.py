from general_quantum_operators import *


def fun_to_minimize(h_, real_p_, psi_0, all_h, all_q, n_qubits=2):
    # all_h = [None, h_b, h_ab], [h_a, None, h_ab], [h_a, h_b, None]
    # all_q = [0,3] --> AD

    full_h = [h_ if v is None else v for v in all_h] # replace the None with the minimization parameter
    H_ = compose_H(full_h, all_q)
    p_ = prob_from_h(h_, psi_0, q, n_qubits=2)
    err_ = rmse(p_, real_p_)
    return err_