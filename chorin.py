import numpy as np
from math import factorial
from scipy.special import comb, eval_genlaguerre
from config import *

# Original diagonal Laguerre-polynomial utilities (kept for compatibility).
def laguerre_coefficient(j, k):
    return ((-1)**j * comb(k, k-j)) / factorial(j)

def laguerre(k, x):
    value = 0
    i = k
    while i >= 0:
        value += laguerre_coefficient(i, k) * x**i
        i -= 1
    return value

def complex_z_1(q1, p1):
    return np.sqrt(d1) * (q1-q10) + 1j * (p1-p10) / np.sqrt(d1)

def complex_z_2(q2, p2):
    return np.sqrt(d2) * (q2-q20) + 1j * (p2-p20) / np.sqrt(d2)

def laguerre_mn(m, n, zeta):
    """L_{m,n}(zeta) from Lasser--Troppmann, Theorem 1 (Eq. 20)."""
    if m <= n:
        return 2**n * factorial(m) * np.conjugate(zeta)**(n-m) * eval_genlaguerre(m, n-m, 2*abs(zeta)**2)
    return 2**m * factorial(n) * (-zeta)**(m-n) * eval_genlaguerre(n, m-n, 2*abs(zeta)**2)

def H(index, q1, q2, p1, p2):
    k, l = index
    zeta1 = complex_z_1(q1, p1) / np.sqrt(2*epsilon)
    zeta2 = complex_z_2(q2, p2) / np.sqrt(2*epsilon)

    norm = np.sqrt(2.0**(sum(k)+sum(l)) * factorial(k[0])*factorial(k[1])*factorial(l[0])*factorial(l[1]))
    value = ((-1)**sum(l) / norm * laguerre_mn(k[0], l[0], zeta1) * laguerre_mn(k[1], l[1], zeta2))
    return np.real_if_close(value)

def chorin_step_one(observable_flow_composed, q1_samples, q2_samples, p1_samples, p2_samples):
    c = []
    for index in basis_indices:
        c.append(np.mean(observable_flow_composed * np.conjugate(H(index, q1_samples, q2_samples, p1_samples, p2_samples)), axis=1))
    return c

# MAKE SURE TO USE NEW INDEPENDENT SAMPLE (the original Chorin procedure).
def chorin_step_two(observable_flow_composed, c_alpha, q1_samples, q2_samples, p1_samples, p2_samples):
    correction = np.zeros_like(observable_flow_composed, dtype=complex)
    for i in range(len(c_alpha)):
        index = basis_indices[i]
        k, l = index
        if k == l == (0, 0):
            # The constant has nonzero expectation; never subtract it alone.
            continue
        H_values = H(index, q1_samples, q2_samples, p1_samples, p2_samples)
        correction += np.asarray(c_alpha[i])[:, None] * H_values[None, :]

    corrected_values = observable_flow_composed - correction
    estimate = np.mean(corrected_values, axis=1)
    return np.real(estimate) if np.isrealobj(observable_flow_composed) else estimate

