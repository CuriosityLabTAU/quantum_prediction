import pandas as pd

df = pd.read_csv('data/new_dataframe.csv', sep='\t')
print(df)

# def quantum_model(p1, p2, p12) --> a_ij

# def join 2+2=4 qubits (a_ij, a_kl) --> a_ijkl

# def trace (a_ijkl, q1, q2) [q1=2, q2=3] --> rho_il,il

# def perform_unitary (a_ijkl, U_ijkl,ijkl) --> a'_ijkl

# def new_two_qubit (a_ijkl, U_ijkl,ijkl, q1, q2) --> Tr_q1,q2 U*a --> rho_il,il [perform_unitary, trace]

# def overlap (rho_il,il, a_il) --> p = <psi_il | rho_il | psi_il>

# def get_unitary(x) --> U_ijkl,ijkl

# def fun_to_minimize (x=U_ijkl,ijkl, param=a_ijkl, q1, q2, a_il) --> min (p - 1)^2
# given a U (this is the x to check):
#   a_ijkl - matrix 16 x N(users)
#   a_il - matrix 4 x N(users)
# for each user, compute error (p-1)^2
# return sum of errors over users
# in (Unitary), out (sum of errors over users)

# def find_optimal_u (x0=I) --> optimal U
# run over many U's and take the minimal error
