import numpy as np
from vel_verlet import flow_vel_verlet
from config import *

rng = np.random.default_rng(seed=42) # MC sampling seed
z0 = np.array([q10,q20,p10,p20]) # gaussian wavepacket centre
covariance = (epsilon / 2.0) * np.linalg.inv(D2) # Covariance corresponding exactly to the Gaussian Wigner density

def MC_sample(sample_size,rng):
    samples = rng.multivariate_normal(mean=z0, cov=covariance, size=sample_size) # returns matrix of shape (N,4) since the mean is (4,) which is 1dim array with 4 entries
    q1_samples = samples[:,0]
    q2_samples = samples[:,1]
    p1_samples = samples[:,2]
    p2_samples = samples[:,3]

    return q1_samples, q2_samples, p1_samples, p2_samples

# computes observables without computing expectation value
def compute_observable(observable, q1, q2, p1, p2):
    observable_values = observable(q1, q2, p1, p2) # composes points with observable
    return observable_values

def compute_expectation(observable, q0, p0, dt, n_steps):
    times = []
    expectation_values = []
    observable_values_stored = []
    q_stored = []
    p_stored = []
    for t, q_t, p_t in flow_vel_verlet(q0=q0, p0=p0, dt=dt, n_steps=n_steps): # loops for each time step
        observable_values = observable(q_t, p_t) # composes points with observable
        times.append(t)
        expectation_values.append(np.mean(observable_values)) # Monte Carlo approximation of the phase-space integral, which is just a mean since we have even-weight quadrature

        observable_values_stored.append(observable_values.copy())
        q_stored.append(q_t.copy())
        p_stored.append(p_t.copy())

       # print("computed " + str(observable.__name__) + " at time " + str(t))    
    
    return (np.asarray(times), np.asarray(expectation_values), np.asarray(observable_values_stored), np.asarray(q_stored), np.asarray(p_stored))