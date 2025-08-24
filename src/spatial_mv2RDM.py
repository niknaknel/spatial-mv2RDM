import cvxpy as cp
from .molecule_helper import getK2
from .dqg import Q2_cvxpy, G2_cvxpy
from .measurements import make_measurement_constraint


def run_sdp(molecule, conditions='DQG', measurements=[], noisy=False, epsilon=1e-8):
    r = molecule.n_orbitals
    N = molecule.n_electrons

    # Define optmization variable
    D2 = cp.Variable((r**2, r**2))

    # Define objective function
    K2 = getK2(molecule)
    E = lambda D : cp.trace(K2 @ D) + molecule.nuclear_repulsion
    objective = cp.Minimize(E(D2))

    # Generate measurement constraints
    measurement_constraints = []
    for (Un, Sn) in measurements:
        measurement_constraints += make_measurement_constraint(D2, Un, Sn, r, noisy, epsilon)

    # Define DQG constraints
    D_constraint = [D2 >> 0]
    Q_constraint = [Q2_cvxpy(D2, r, N) >> 0]
    G_constraint = [G2_cvxpy(D2, r, N) >> 0]
    trace_constraint = [cp.trace(D2) == 0.5 * N * (N-1)]

    # Build full constraints
    constraints = measurement_constraints + \
        (D_constraint if 'D' in conditions else []) + \
        (Q_constraint if 'Q' in conditions else []) + \
        (G_constraint if 'G' in conditions else []) + \
        (trace_constraint if len(conditions) > 0 else [])

    # Solve SDP
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, eps=1e-8) # change to cp.MOSEK if you have a license (remember to set license path)

    # Check status
    if problem.status in cp.settings.INF_OR_UNB:
        print('Problem %s. Exiting run.' % problem.status)
        return None

    # Get results
    primal = E(D2).value
    dual = -constraints[-1].dual_value * 0.5 * N*(N-1) # dual obtained from trace

    return {'primal': primal, 'dual': dual, 'D2': D2.value}