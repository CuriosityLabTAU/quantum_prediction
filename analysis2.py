from general_quantum_operators import *
import pickle
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# best combination is UMNh = 1100

sns.set_context('notebook')


def prob_from_psi_general_initial(all_P):
    # psi 1111?
    n_qubits = 4
    h_mix_type = 0
    psi_0 = uniform_psi(n_qubits)

    full_h = [None, None, None]
    all_q = [0,1]
    H_ = compose_H(full_h, all_q, n_qubits, h_mix_type)
    psi_dyn = get_psi(H_, psi_0)
    P_ = MultiProjection(all_P, all_q, n_qubits)
    psi_final = np.dot(P_, psi_dyn)
    p_ = norm_psi(psi_final)
    # print(p_)
    return p_


def visualize_U(data_path):
    control_str = '_U_%s_mixing_%s_neutral_%s_mix_type_%d' % (True, True, False, 0)
    all_data = pickle.load(open(data_path + '/all_data%s.pkl' % control_str, 'rb'))
    q_info = pickle.load(open(data_path + '/q_info%s.pkl' % control_str, 'rb'))

    # pd.DataFrame.from_dict(q_info).to_csv('q_info_pd.csv')

    fig, ax = plt.subplots(2,2)
    for i, q in enumerate(range(2,6)):
        sns.heatmap(q_info[q]['U'].real, ax= ax[i/2, i%2], vmin=-.2, vmax=.2,  cmap="RdBu")
        # ax[i/2, i%2].imshow(q_info[q]['U'].real)
        ax[i/2, i%2].set_title('q{}'.format(q))

    fig.savefig('figs/u_visualization.png', dpi=300)


def main():
    # data_path = 'data_all'
    data_path = 'data'

    ### to see that the intial state gives 0.5 probability to all possible probabilities
    all_P = ['0','1','C','D']
    for P in all_P:
        print('%s probability for psi = |1111> is %.2f' % (str(P), prob_from_psi_general_initial(P)))

    ### to see if there is any connection between the different Us, dependent on the the third question.
    visualize_U(data_path)

    plt.show()

if __name__ == '__main__':
    main()