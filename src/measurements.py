import numpy as np
import cvxpy as cp
from scipy.stats import unitary_group


def generate_measurement(D2_fci, r, noisy=False, epsilon=1e-8):
    Un = unitary_group.rvs(r)  # returns Haar distributed unitary
    UxU = np.kron(Un, Un)
    Sn = np.diag(UxU @ D2_fci @ UxU.conj().T).reshape((r, r), order='C')
    if noisy:
        gaussian_noise = np.random.normal(loc=0.0, scale=epsilon, size=Sn.shape)
        Sn = Sn + gaussian_noise
    return Un, Sn.real  # .real discards imaginary part in case of tiny errors


def make_measurement_constraint(Dvar, Un, Sn, r, noisy=False, epsilon=1e-8):
    UxU = cp.kron(Un, Un)
    X = cp.real(cp.diag(UxU @ Dvar @ UxU.conj().T)).reshape((r, r), order='C')
    if noisy:
        return [Sn - 3 * epsilon <= X, X <= Sn + 3 * epsilon]
    else:
        return [Sn == X]
