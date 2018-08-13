from scipy.optimize import minimize


# function to minimize
def fun_to_minimize(x, p, fallacy=1):
    pA = abs(x[2] + x[3]) ** 2 / (abs(x[2] + x[3]) ** 2 + abs(x[0] + x[1]) ** 2)
    pB = abs(x[1] + x[3]) ** 2 / (abs(x[1] + x[3]) ** 2 + abs(x[0] + x[2]) ** 2)
    if fallacy == 1:
        pA_B = abs(x[3]) ** 2 / (abs(x[3]) ** 2 + abs(x[0] + x[1] + x[2]) ** 2)
    elif fallacy == 2:
        pA_B = abs(x[1] + x[2] + x[3]) ** 2 / (abs(x[1] + x[2] + x[3]) ** 2 + abs(x[0]) ** 2)

    error = (pA - p['A']) ** 2 + \
            (pB - p['B']) ** 2 + \
            (pA_B - p['A_B'])**2

    return error


def fun_to_constraint(x):
    norm = abs(x[0])**2 + abs(x[1])**2 + abs(x[2])**2 + abs(x[3])**2 - 1
    return norm



def quantum_coefficients(df):
    '''
    Calculating the coefficents based on the probabilities. (p1, p2, p12) --> a_ij
    :param df: dataframe with all the data.
    :return: df + a_ij
    '''

    df_ = df[df['pos'] <= 2]

    for d in df_.iterrow():
        print(d)

    #
    #
    #
    # # run minimization
    # res_temp = minimize(fun_to_minimize, x0, args=(psi, p1, p2, p12, q1, q2, fal),
    #                     method='SLSQP', bounds=None, constraints=fun_to_constraint(),
    #                     options={'disp': False})
    #
    # print(res_temp.x, res_temp.fun)
    #
    # df = df.assign(**{'a10': np.nan, 'a01': np.nan, 'a11': np.nan, 'a00': np.nan})
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