"""
Boilerplate code for generating molecules with OpenFermion.
Also contains K2 and D2fci functions used in the v2RDM.

"""
from itertools import product

from openfermionpyscf import run_pyscf
from openfermion import MolecularData, get_fermion_operator, jordan_wigner, get_sparse_operator
import numpy as np

"""
    Helper functions for molecule setup.
"""
MOLECULES = {
    'HYDROGEN': {
        'geometry': lambda x: [('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, x))],
        'multiplicity': 1,
        'charge': 0,
        'description': lambda x: f"Hydrogen_H2_{x}"
    },
    'HYDROGEN_CHAIN': {
        'geometry': lambda x: [
                ('H', (0.0, 0.0, 0.0)),
                ('H', (0.0, 0.0, x)),
                ('H', (0.0, 0.0, 2*x)),
                ('H', (0.0, 0.0, 3*x))
            ],
        'multiplicity': 1,
        'charge': 0,
        'description': lambda x: f"Hydrogen_chain_H4_{x}"
    },
    'LITHIUM_HYDRIDE': {
        'geometry': lambda x: [('Li', (0., 0., 0.)), ('H', (0., 0., x))],
        'multiplicity': 1,
        'charge': 0,
        'description': lambda x: f"Lithium_hydride_LiH_{x}"
    },
    'HYDROGEN_FLUORIDE': {
        'geometry': lambda x: [('H', (0.0, 0.0, 0.0)), ('F', (0.0, 0.0, x))],
        'multiplicity': 1,
        'charge': 0,
        'description': lambda x: f"Hydrogen_fluoride_HF_{x}"
    },
    'WATER': {
        'geometry': lambda x: [ # triangular
            ('O', (0., 0., 0.)),
            ('H', (x, x, 0.)),
            ('H', (-x, x, 0.))
        ],
        'multiplicity': 1,
        'charge': 0,
        'description': lambda x: f"Water_H2O_{x}"
    },
    'AMMONIA': {
        'geometry': lambda x: [
            ('N', (0., 0., 0.)),
            ('H', (0., 2*x, -x)),  # H1
            ('H', (-x, -x, -x)),   # H2
            ('H', (x, -x, -x))     # H3
        ],
        'multiplicity': 1,
        'charge': 0,
        'description': lambda x: f"Ammonia_NH3_{x}"
    },
    'NITROGEN': {
        'geometry': lambda x: [('N', (0., 0., 0.)), ('N', (0., 0., x))],
        'multiplicity': 1,
        'charge': 0,
        'description': lambda x: f"Nitrogen_N2_{x}"
    }
}

def make_summary(molecule):
    d = {
        'Molecule': molecule.description,
        'Electrons': molecule.n_electrons,
        'Orbitals': molecule.n_orbitals,
        'Qubits': molecule.n_qubits,
        'FCI': molecule.fci_energy,
        'Basis': molecule.basis
    }
    strout = ""
    for k, v in d.items():
        strout += "{:<10} {:<10}\n".format(k + ':', v)
    return strout


def create_molecule(option_name, distance, basis='sto-3g'):
    opt = MOLECULES[option_name.upper()]
    molecule = MolecularData(
        geometry=opt['geometry'](distance),
        basis=basis,
        multiplicity=opt['multiplicity'],
        charge=opt['charge'],
        description=opt['description'](distance)
    )
    molecule = run_pyscf(molecule, run_scf=True, run_cisd=True, run_fci=True)
    return molecule


def get_fci_curve(option_name, linspace=np.linspace(0.25, 2.5, 50)):
    fcis = [create_molecule(option_name, x).fci_energy for x in linspace]
    return linspace, fcis


def get_hamiltonian(molecule, qubit=True):
    molecular_hamiltonian = molecule.get_molecular_hamiltonian()
    hamiltonian = get_fermion_operator(molecular_hamiltonian)
    if qubit:
        hamiltonian = jordan_wigner(hamiltonian)

    h_matrix = get_sparse_operator(hamiltonian, molecule.n_qubits).todense()
    return h_matrix


def get_groundstate(h_matrix, k=0):
    eigenvalues, eigenvectors = np.linalg.eigh(h_matrix)  # use sparse for larger systems
    ground_state_energy = eigenvalues[k]
    ground_state = eigenvectors[:, k]  # Ground state is first eigenvector
    return ground_state_energy, ground_state


"""
    Functions for reduced Hamiltonian ²K and ²D_fci.
"""
def getK2(molecule):
    r, N = molecule.n_orbitals, molecule.n_electrons
    K2 = np.zeros((r, r, r, r))
    h1 = molecule.one_body_integrals
    h2 = molecule.two_body_integrals.transpose(0, 1, 3, 2)  # chemist -> physicist notation

    for i, j, k, l in product(range(r), repeat=4):
        # Embed one-body terms into two-body form
        term1 = h1[i, k] * (1 if j == l else 0)
        term1 += h1[j, l] * (1 if i == k else 0)
        K2[i, j, k, l] = term1 / (N - 1) + h2[i, j, k, l]  # Add two-body integrals

    return K2.reshape((r ** 2, r ** 2), order='C')


def get_spatial_D2(molecule):
    r = molecule.n_orbitals
    D2 = molecule.fci_two_rdm  # returns spatial orbital version
    D2 = 0.5 * D2.transpose(0, 1, 3, 2)  # adjust for normalization and chemist's notation
    return D2.reshape((r ** 2, r ** 2), order='C')


if __name__ == '__main__':
    # Example
    mol = create_molecule("hydrogen", 0.7414)
    print(mol.description, ":")
    print("n_electrons:", mol.n_electrons)
    print("n_orbitals:", mol.n_orbitals)
    print("n_qubits:", mol.n_qubits)