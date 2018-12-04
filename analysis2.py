from general_quantum_operators import *
import pickle
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# best combination is UMNh = 1100

sns.set_context('notebook')


def prob_from_psi_1111():
    n_qubits = 4
    h_mix_type = 0
    psi_0 = uniform_psi(n_qubits)
    full_h = [None, None, None]
    all_q = [0,1]
    all_P = '0' #'D' # 0,1,C,D
    H_ = compose_H(full_h, all_q, n_qubits, h_mix_type)
    psi_dyn = get_psi(H_, psi_0)
    P_ = MultiProjection(all_P, all_q, n_qubits)
    psi_final = np.dot(P_, psi_dyn)
    p_ = norm_psi(psi_final)
    print(p_)


def visualize_U():
    control_str = '_U_%s_mixing_%s_neutral_%s_mix_type_%d' % (True, True, False, 0)
    all_data = pickle.load(open('data_all/all_data%s.pkl' % control_str, 'rb'))
    q_info = pickle.load(open('data_all/q_info%s.pkl' % control_str, 'rb'))

    # pd.DataFrame.from_dict(q_info).to_csv('q_info_pd.csv')

    fig, ax = plt.subplots(2,2)
    for i, q in enumerate(range(2,6)):
        sns.heatmap(q_info[q]['U'].real, ax= ax[i/2, i%2])
        # ax[i/2, i%2].imshow(q_info[q]['U'].real)
        ax[i/2, i%2].set_title('q{}'.format(q))

    fig.savefig('figs/u_visualization.png', dpi=300)


def main():
    prob_from_psi_1111()
    visualize_U()
    # plt.show()

if __name__ == '__main__':
    main()