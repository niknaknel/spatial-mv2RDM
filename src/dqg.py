"""
dqg.py

Functions for calculating D, Q and G matrices for N-representability using OpenFermion.
Also contains linear mappings between D, Q and G, implemented using both Numpy and CVXPY.

General notation guide:
    r = # of orbitals. When using the mapping function with qubit based matrices, this should be set
        equal to molecule.n_qubits. Otherwise, when using spatial orbital based RDMs this should be set
        to molecule.n_orbitals.

    N = # electrons. i.e. molecule.n_electrons

"""
from itertools import product

from openfermion import FermionOperator, jordan_wigner, get_sparse_operator, expectation
import numpy as np
import cvxpy as cp

"""
    Compute ¹D and ¹Q using OpenFermion.
"""
def replace1(s, i, j):
    repl = str.maketrans({'i': '%d' % i, 'j': '%d' % j})
    return s.translate(repl)


def __compute_expectation1(molecule, ground_state, second_quant_string, qubit):
    n_qubits = molecule.n_qubits
    M1 = np.zeros((n_qubits, n_qubits))

    for i, j in product(range(n_qubits), repeat=2):
        fs = replace1(second_quant_string, i, j)
        op = FermionOperator(fs, 1.0)
        if qubit:
            op = jordan_wigner(op)
        sparse = get_sparse_operator(op, n_qubits=n_qubits)
        M1[i, j] = expectation(sparse, ground_state)

    return M1


def computeD1(molecule, ground_state, qubit=True):
    return __compute_expectation1(molecule, ground_state, "i^ j", qubit)


def computeQ1(molecule, ground_state, qubit=True):
    return __compute_expectation1(molecule, ground_state, "i j^", qubit)


"""
    Define linear mappings between ¹D and ¹Q, as well as ¹D and ²D.

    References

    [1] J. G. Li, N. Michel, W. Zuo, and F. R. Xu. "Reexamining the variational two-particle reduced
        density matrix for nuclear systems", (2021). DOI: 10.1103/PhysRevC.103.064324

        Equation 13:
            ¹Qij = δij − ¹Dij   (13)

    [2] David A. Mazziotti et al. "Reduced-density-matrix Mechanics: with Application to Many-electron
        Atoms and Molecules", John Wiley & Sons, Inc. (2007). ISBN: 978-0-471-79056-3.

        Equations 16 & 17 on p25:
            ¹Dik = 1/(N-1)   Σj ²Dijkj     (16)
            ¹Qik = 1/(r-N-1) Σj ²Qijkj     (17)
"""
def Q1_numpy(D1):
    return np.eye(D1.shape[0]) - D1


def D1_ik_numpy(D2, r, N):
    D4 = D2.reshape(r, r, r, r)
    D2_traced = D4.trace(axis1=1, axis2=3)
    D1 = D2_traced / (N - 1)
    return D1


def Q1_ik_numpy(Q2, r, N):
    Q4 = Q2.reshape(r, r, r, r)
    Q2_traced = Q4.trace(axis1=1, axis2=3)
    Q1 = Q2_traced / (r - N - 1)
    return Q1


"""
    Compute ²D, ²Q and ²G matrices for N-representability using OpenFermion.

    References:
    -----------
    [1] David A. Mazziotti. "Significant conditions for the two-electron reduced density matrix from the constructive
        solution of N representability", (2012). 10.1103/PhysRevA.85.06250.

        Equations 13-15 read:
            ²Dijkl = <Ψ| ai^ aj^ al  ak  |Ψ>    (two-particle)  (13)
            ²Qijkl = <Ψ| ai  aj  al^ ak^ |Ψ>    (two-hole)      (14)
            ²Gijkl = <Ψ| ai^ aj  al^ ak  |Ψ>    (particle-hole) (15)
"""
def replace2(s, i, j, k, l):
    repl = str.maketrans({'i': '%d' % i, 'j': '%d' % j, 'k': '%d' % k, 'l': '%d' % l})
    return s.translate(repl)


def __compute_expectation2(molecule, ground_state, second_quant_string, qubit):
    n_qubits = molecule.n_qubits
    M2 = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))

    for i, j, k, l in product(range(n_qubits), repeat=4):
        fs = replace2(second_quant_string, i, j, k, l)
        op = FermionOperator(fs, 1.0)
        if qubit:
            op = jordan_wigner(op)
        sparse = get_sparse_operator(op, n_qubits=n_qubits)
        M2[i, j, k, l] = expectation(sparse, ground_state)

    M2 = np.reshape(M2, (n_qubits ** 2, n_qubits ** 2), order='C')
    return M2


def computeD2(molecule, ground_state, qubit=True):
    return __compute_expectation2(molecule, ground_state, "i^ j^ l k", qubit)


def computeQ2(molecule, ground_state, qubit=True):
    return __compute_expectation2(molecule, ground_state, "i j l^ k^", qubit)


def computeG2(molecule, ground_state, qubit=True):
    return __compute_expectation2(molecule, ground_state, "i^ j l^ k", qubit)


"""
    Define linear mappings between ²D, ²Q and ²G matrices for N-representability using Numpy.

    References:
    -----------
    [1] David A. Mazziotti. "Pure-N-representability conditions of two-fermion reduced density matrices",
        (2016). DOI: 10.1103/PhysRevA.94.032516.

        Equation 17 (with relabelled indices):
            ²Qijkl = 2 (δik ∧ δjl)  −  4¹Dik ∧ δjl  +  ²Dijkl    (17)

    [2] David A. Mazziotti et al. "Reduced-density-matrix Mechanics: with Application to Many-electron
        Atoms and Molecules", John Wiley & Sons, Inc. (2007). ISBN: 978-0-471-79056-3.

        Equations 14-15 on p25:
            ²Qijkl = 2 ²Iijkl  -  4¹Dik ∧ ¹Ijl  +  ²Dijkl    (14)
            ²Gijkl = ¹Ijl ¹Dik  -  ²Dilkj                   (15)

    [3] J. G. Li, N. Michel, W. Zuo, and F. R. Xu. "Reexamining the variational two-particle reduced
        density matrix for nuclear systems", (2021). DOI: 10.1103/PhysRevC.103.064324

        Equations 17-18 (using above notation for clarity. Notice the terms are equivalent):
            ²Qijkl = (δik δjl − δjk δil)  −  (δik ¹Djl − δil ¹Djk − δjk ¹Dil + δjl ¹Dik)  +  ²Dijkl
            ²Gijkl = δjl ¹Dik  −  ²Dilkj 


    Explanation
    -----------
    1) The term  "δjl¹Dik" or equivalently "¹Ijl¹Dik" means you need to trace out Dijkl where j=l,
    i.e. δjl¹Dik gives has the trace of each square submatrix of Dijkl as its elements. E.g.
    if Dijkl = [1  2  |  3  4]
               [5  6  |  7  8]
               ---------------
               [9  10 | 11 12]
               [13 14 | 15 16]
    then
        δjl¹Dik = [7  11]
                  [23 27]
                  
    Note that Dik also needs to be appropriately normalized, i.e. * 1/(N-1)

    2) The term "¹Dik ∧ δjl" is the Grassman wedge product with the identity and is equal to term 2 in [3]
        up to factor of 4.

    3) Note the swapped l & j indices last term in the equations, "²Dilkj". This is just transposing
        each square submatrix of ²Dijkl.

    NOTE: Versions [1]/[2] were compared with [3] and found to be numerically equivalent.

"""
def wedge_product_identity_numpy(A):
    r = A.shape[0]
    i, j, k, l = np.ogrid[:r, :r, :r, :r]
    term1 = np.multiply(A[i, k], (j == l))
    term2 = np.multiply(A[j, k], (i == l))
    term3 = np.multiply(A[i, l], (j == k))
    term4 = np.multiply(A[j, l], (i == k))
    C = (term1 - term2 - term3 + term4) / 4
    C2d = np.reshape(C, (r ** 2, r ** 2), order='C')
    return C2d


def Q2_numpy(D2, r, N):
    t1 = wedge_product_identity_numpy(np.eye(r))
    D1 = D1_ik_numpy(D2, r, N)
    wp = wedge_product_identity_numpy(D1)
    return 2 * t1 - 4 * wp + D2


def G2_numpy(D2, r, N):
    D1 = D1_ik_numpy(D2, r, N)
    t1 = np.kron(D1, np.eye(r))
    t2 = D2.reshape(r, r, r, r).transpose(0, 3, 2, 1).reshape(r ** 2, r ** 2)
    return t1 - t2  # D1 only returns for one spin orientation.


"""
    CVXPY implementation of linear mappings between ²D, ²Q and ²G matrices for N-representability.

    Not all Numpy functions are available in CVXPY. So the implementation is slightly more long-winded but equivalent.

    See references and explanation above.
"""
def wedge_product_identity_cvxpy(A):
    r = A.shape[0]
    i, j, k, l = np.ogrid[:r, :r, :r, :r]
    term1 = cp.multiply(A[i, k], (j == l))
    term2 = cp.multiply(A[j, k], (i == l))
    term3 = cp.multiply(A[i, l], (j == k))
    term4 = cp.multiply(A[j, l], (i == k))
    C = (term1 - term2 - term3 + term4) / 4
    C2d = cp.reshape(C, (r ** 2, r ** 2), order='C')
    return C2d


def D1_ik_cvxpy(D2, r, N):
    rows = []
    for i in range(r):
        columns = []
        for k in range(r):
            Dsub = D2[i * r: (i + 1) * r, k * r: (k + 1) * r]
            columns.append(cp.trace(Dsub) / (N - 1))
        rows.append(columns)
    return cp.bmat(rows)


def Q1_ik_cvxpy(Q2, r, N):
    rows = []
    for i in range(r):
        columns = []
        for k in range(r):
            Qsub = Q2[i * r: (i + 1) * r, k * r: (k + 1) * r]
            columns.append(cp.trace(Qsub) / (r - N - 1))
        rows.append(columns)
    return cp.bmat(rows)


def subdivide_cvxpy(D2, r):  # weirdly, faster than vectorized version!
    cuts = r ** 2 // r
    submatrices = []
    for i in range(cuts):
        for j in range(cuts):
            submatrices.append(D2[i * r: (i + 1) * r, j * r: (j + 1) * r])
    return submatrices


def D2_jl_T_cvxpy(D2, r):
    submatrices = subdivide_cvxpy(D2, r)
    transposed = [cp.transpose(m) for m in submatrices]
    D2ilkj = np.block([[transposed[i * r + j] for j in range(r)] for i in range(r)])
    return cp.bmat(D2ilkj)


def Q2_cvxpy(D2, r, N):
    t1 = wedge_product_identity_cvxpy(np.eye(r))
    D1 = D1_ik_cvxpy(D2, r, N)
    wp = wedge_product_identity_cvxpy(D1)
    return 2 * t1 - 4 * wp + D2


def G2_cvxpy(D2, r, N):
    D1 = D1_ik_cvxpy(D2, r, N)
    t1 = cp.kron(D1, np.eye(r)).T
    t2 = D2_jl_T_cvxpy(D2, r)  # swap j & l
    return t1 - t2
