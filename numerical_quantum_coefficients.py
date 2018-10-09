from scipy.optimize import minimize
import numpy as np
from qutip import *
from tools import *

# function to minimize
def fun_to_minimize(x, p, fallacy=1):
    pA, pB, pA_B = calculate_p(x, fallacy)

    error = (pA - p['A']) ** 2 + \
            (pB - p['B']) ** 2 + \
            (pA_B - p['A_B']) ** 2

    return error


def fun_to_constraint(x):
    norm = abs(x[0]) ** 2 + abs(x[1]) ** 2 + abs(x[2]) ** 2 + abs(x[3]) ** 2 - 1
    return norm


def calculate_p(x, fallacy=1):
    pA = abs(x[2] + x[3]) ** 2 / (abs(x[2] + x[3]) ** 2 + abs(x[0] + x[1]) ** 2)
    pB = abs(x[1] + x[3]) ** 2 / (abs(x[1] + x[3]) ** 2 + abs(x[0] + x[2]) ** 2)
    if fallacy == 1:
        pA_B = abs(x[3]) ** 2 / (abs(x[3]) ** 2 + abs(x[0] + x[1] + x[2]) ** 2)
    elif fallacy == 2:
        pA_B = abs(x[1] + x[2] + x[3]) ** 2 / (abs(x[1] + x[2] + x[3]) ** 2 + abs(x[0]) ** 2)
    return pA, pB, pA_B


def numerical_quantum_coefficients(df):
    '''
    Calculating the coefficents based on the probabilities. (p1, p2, p12) --> a_ij
    :param df: dataframe with all the data.
    :return: df + a_ij
    '''
    n_runs = 100

    df = df.assign(**{'a10': np.nan, 'a01': np.nan, 'a11': np.nan, 'a00': np.nan})
    df = df.assign(**{'check_pA': np.nan, 'check_pB': np.nan, 'check_pA_B': np.nan})

    user_ids = df['userID'].unique()
    pos = df['pos'].unique()

    for ui, u_id in enumerate(user_ids):
        print('calculating states for user #:',  ui)
        # TORR: prev_x is always a 4-qubit state
        # --> initially its the all-superposition
        prev_x = ndarray2Qobj(np.ones([16]),norm = True)
        for p_id in range(6):

            d = df[(df['userID'] == u_id) & (df['pos'] == p_id)]
            p = {
                'A': d['p1'],
                'B': d['p2'],
                'A_B': d['p12']
            }
            fal = np.squeeze(np.array(d['fal']))

            # TORR: find the two qubits in question
            # --> prev_rho = Tr_24 prev_x
            i, l = int(d['q1']) , int(d['q2'])
            uq, qorder = unasked_qubits([i, l])
            prev_rho = prev_x.ptrace([i-1, l-1])
            solutions = []
            similarity = []
            distances = []
            for r in range(n_runs):
                x0 = np.random.random(4)

                cons = ({'type': 'eq', 'fun': fun_to_constraint})
                # run minimization
                res_temp = minimize(fun_to_minimize, x0, args=(p, fal),
                                    method='SLSQP', bounds=None, constraints=cons,
                                    options={'disp': False})

                #           print(res_temp.x, res_temp.fun)
                #           print('normalization =',fun_to_constraint(res_temp.x))
                if res_temp.fun < 1e-5:
                    solutions.append(res_temp.x)
                    x = ndarray2Qobj(res_temp.x)
                    # TORR: similarity = res_temp.x.dagger * prev_rho * res_temp.x
                    csimil = x.dag() * prev_rho * x
                    similarity.append(csimil.full().real)
                    # distances.append(np.sum((np.array(res_temp.x) - np.array(prev_x)) ** 2))

            # TORR: highest_similary = np.argmax(similarity)
            highest_similary = np.argmax(similarity)
            #         lowest_change = np.argmin(np.array(distances))
            # TORR: best_solution is a two-qubit state
            best_solution = ndarray2Qobj(solutions[highest_similary])
            #         print(solutions)
            #         print(distances)
            # print(p_id, best_solution)

            # TORR: prev_x = (best_solution * best_solution.dagger)_4x4 (*) I_other_4x4 * prev_x
            rho_best_solution = ket2dm(best_solution)
            # todo: pretty sure that I have order problem here
            rho_ijkl_not_ordered = tensor(rho_best_solution, tensor(qeye(2), qeye(2)))
            # rho_ijkl_not_ordered = tensor(tensor(qeye(2), qeye(2)), a)
            rho_ijkl_ordered = reorganize_operator(qorder, rho_ijkl_not_ordered.full().real)
            rho_ijkl_ordered = ndarray2Qobj(rho_ijkl_ordered)
            rho_ijkl_ordered.dims = (prev_x * prev_x.dag()).dims
            prev_x = rho_ijkl_ordered * prev_x
            prev_x = ndarray2Qobj(prev_x, norm=True)

            # print('position:',p_id, ', max_sim:',similarity[highest_similary], ', norm:', prev_x.norm())

        #         d[['a00','a01' ,'a10', 'a11']] = res_temp.x
        #         d[['check_pA', 'check_pB', 'check_pA_B']] = calculate_p(res_temp.x, fal)

        # break
    return df