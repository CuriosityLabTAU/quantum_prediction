from scipy.optimize import minimize
import numpy as np

# function to minimize
def fun_to_minimize(x, p, fallacy=1):
    pA, pB, pA_B = calculate_p(x, fallacy)

    error = (pA - p['A']) ** 2 + \
            (pB - p['B']) ** 2 + \
            (pA_B - p['A_B'])**2

    return error


def fun_to_constraint(x):
    norm = abs(x[0])**2 + abs(x[1])**2 + abs(x[2])**2 + abs(x[3])**2 - 1
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


    df = df.assign(**{'a10': np.nan, 'a01': np.nan, 'a11': np.nan, 'a00': np.nan})
    df = df.assign(**{'check_pA': np.nan, 'check_pB': np.nan, 'check_pA_B': np.nan})

    for d_iter in df.iterrows():
        d = d_iter[1]
        p = {
            'A': d['p1'],
            'B': d['p2'],
            'A_B': d['p12']
        }

        fal = d['fal']

        x0 = [0.5, 0.5, 0.5, 0.5]
        intials = [[0.5, 0.5, 0.5, 0.5],[.9,0.01,0.01,0.01],[0.01,.9,0.01,0.01],[0.01,0.01,.9,0.01],[0.01,0.01,0.01,.9],[.02,.05,-.6,.9]]
        solution = np.zeros([intials.__len__(), 4])
        for i, x0 in enumerate(intials):
            cons = ({'type': 'eq', 'fun': fun_to_constraint})
            # run minimization
            res_temp = minimize(fun_to_minimize, x0, args=(p, fal),
                                method='SLSQP', bounds=None, constraints=cons,
                                options={'disp': False})

            print(res_temp.x, res_temp.fun)
            print('normalization =',fun_to_constraint(res_temp.x))

            solution[i,:] = res_temp.x

            d[['a00','a01' ,'a10', 'a11']] = res_temp.x
            d[['check_pA', 'check_pB', 'check_pA_B']] = calculate_p(res_temp.x, fal)

        break #
    return df
    #

    #
    # # Conjunction
    # df.loc[df.fal == 1., 'a10'] = np.sqrt(2 * df[df.fal == 1.].p1) - np.sqrt(df[df.fal == 1.].p12)
    # df.loc[df.fal == 1., 'a01'] = np.sqrt(2 * df[df.fal == 1.].p2) - np.sqrt(df[df.fal == 1.].p12)
    # df.loc[df.fal == 1., 'a11'] = np.sqrt(df[df.fal == 1.].p12)
    #
    # # Disjunction
    # df.loc[df.fal == 2., 'a10'] = np.sqrt(3 * df[df.fal == 2.].p12) - np.sqrt(2 * df[df.fal == 2.].p2)
    # df.loc[df.fal == 2., 'a01'] = np.sqrt(2 * df[df.fal == 2.].p2) - np.sqrt(2 * df[df.fal == 2.].p1)
    # df.loc[df.fal == 2., 'a11'] = np.sqrt(2 * df[df.fal == 2.].p1) + np.sqrt(2 * df[df.fal == 2.].p2) - np.sqrt(
    #     3 * df[df.fal == 2.].p12)
    #
    # # a_00 from normalization
    # df.loc[:, 'a00'] = np.sqrt(1 - df.a10 ** 2 - df.a01 ** 2 - df.a11 ** 2)
    # return df