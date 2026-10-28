import numpy as np
import scipy as s

epsilon = 0.1
d = 2 # dimension of system

# matrix for initial gaussian wavepacket
d1 = 1.0
d2 = 1.0
D1 = np.diag([d1,d2])
D2 = s.linalg.block_diag(D1, np.linalg.inv(D1))

q10 = 0.1
q20 = 0.0
p10 = 0.3
p20 = 0.0
lamb = 1 # lambda for Henon-Heiles system

N = 10000 # number of samples
dt = 0.01
final_time = 1.0
target_time = 1.0
n_steps = int(round(final_time / dt))

basis_indices = [
    ((1, 0), (1, 0)),
    ((0, 1), (0, 1)),
    ((2, 0), (2, 0)),
    ((1, 1), (1, 1)),
    ((0, 2), (0, 2)),
    ((1, 0), (0, 0)),
    ((0, 0), (1, 0)),
    ((0, 1), (0, 0)),
    ((0, 0), (0, 1)),
    ((1, 0), (0, 1)),
    ((0, 1), (1, 0)),
]

number_of_trials = 20

