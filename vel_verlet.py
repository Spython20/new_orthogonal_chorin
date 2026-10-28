import numpy as np
from config import *

def grad_V(q1,q2): # take in sampled position arrays
    grad_q1_values = []
    grad_q2_values = []
    for i in range(len(q1)):
        grad_q1_values.append(q1[i] + 2* lamb*(q1[i] * q2[i]))
        grad_q2_values.append(q2[i]+ lamb * (q1[i]**2 - q2[i]**2))

    return np.array(grad_q1_values), np.array(grad_q2_values)

# takes in array of positions and momentums of each sampled point
def flow_vel_verlet(q10,q20,p10, p20, dt, n_steps):
    q1 = []
    q2 = []
    p1 = []
    p2 = []

    q1.append(q10)
    q2.append(q20)
    p1.append(p10)
    p2.append(p20)

    # velocity verlet algorithm
    for n in range(n_steps):
        grad_q1, grad_q2 = grad_V(q1[n], q2[n])
        p1_half = p1[n] - 0.5 * dt * grad_q1
        p2_half = p2[n] - 0.5 * dt * grad_q2

        q1.append(q1[n] + dt * p1_half)
        q2.append(q2[n] + dt * p2_half)

        grad_q1, grad_q2 = grad_V(q1[n+1], q2[n+1]) # recompute grad after new position computed

        p1.append(p1_half- 0.5 * dt * grad_q1)
        p2.append(p2_half- 0.5 * dt * grad_q2)

    return np.array(q1), np.array(q2), np.array(p1), np.array(p2) 
'''
these arrays are of the form
[
  [sample1, sample2, sample3, ...]   <- time step 0
  [sample1, sample2, sample3, ...]   <- time step 1
  [sample1, sample2, sample3, ...]   <- time step 2
  ...
]
'''