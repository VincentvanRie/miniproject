# Includes copy from https://github.com/quantumlib/OpenFermion/tree/v1.6.1

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cirq
import openfermion
import numpy
from openfermion import FermionOperator, jordan_wigner
from openfermion.transforms import get_quadratic_hamiltonian


x_dimension = 2
y_dimension = 2
n_sites = x_dimension*y_dimension
n_spin_orbitals = 2 * n_sites
t = 1
U = 2
S = 3

def vfop_to_qop(theta, fop, coefficient=1.0):
    """
    Transform a FermionOperator to a QubitOperator using Jordan-Wigner transformation.
    
    Args:
    - theta (float): Parameter for the transformation.
    - fop (FermionOperator): The Fermion operator to transform.
    - coefficient (float, optional): Coefficient for the transformation. Default is 1.0.
    
    Returns:
    - QubitOperator: The transformed qubit operator.
    """
    fop = (1j)* coefficient * theta * fop
    return jordan_wigner(fop)

def rightNeighbour(site):
    """
    Get the right neighbor of a site in a 2D lattice.
    
    Args:
    - site (int): The index of the site.
    
    Returns:
    - int: The index of the right neighbor.
    """
    if (site + 1) % x_dimension == 0:
        return site + 1 - x_dimension
    return site + 1

def bottomNeighbour(site):
    """
    Get the bottom neighbor of a site in a 2D lattice.
    
    Args:
    - site (int): The index of the site.
    
    Returns:
    - int or None: The index of the bottom neighbor or None if it doesn't exist.
    """
    if site + x_dimension >= n_sites:
        return None
    return site + x_dimension

def hoppingTerm(i, j, coefficient):
    """
    Create a hopping term between two sites.
    
    Args:
    - i (int): The index of the first site.
    - j (int): The index of the second site.
    - coefficient (complex): The coefficient for the hopping term.
    
    Returns:
    - FermionOperator: The hopping term operator.
    """
    op_class = FermionOperator
    hopping_term = op_class(((i, 1), (j, 0)), coefficient)
    hopping_term += op_class(((j, 1), (i, 0)), coefficient.conjugate())
    return hopping_term

def coulombInteractionTerm(i, j):
    """
    Create a Coulomb interaction term between two sites.
    
    Args:
    - i (int): The index of the first site.
    - j (int): The index of the second site.
    
    Returns:
    - FermionOperator: The Coulomb interaction term operator.
    """
    number_operator_i = numberOperator(i)
    number_operator_j = numberOperator(j)
    return U * number_operator_i * number_operator_j

def numberOperator(i):
    """
    Create a number operator for a site.
    
    Args:
    - i (int): The index of the site.
    
    Returns:
    - FermionOperator: The number operator.
    """
    op_class = FermionOperator
    return op_class(((i, 1), (i, 0)))

def createHubbardHamiltonian(isInteracting=True):
    """
    Create the Hubbard Hamiltonian.
    
    Args:
    - isInteracting (bool, optional): Whether to include Coulomb interaction terms. Default is True.
    
    Returns:
    - dict: A dictionary with keys 'horizontal', 'vertical', and 'repulsion' containing the respective FermionOperators.
    """
    # Initialise operator.
    hubbard_model_horizontal = FermionOperator()
    hubbard_model_vertical = FermionOperator()
    hubbard_model_repulsion = FermionOperator()

    # Loop through sites and add terms.
    for site in range(n_sites):
        # Get indices of right and bottom neighbors
        right_neighbour = rightNeighbour(site)
        bottom_neighbour = bottomNeighbour(site)

        # Avoid double-counting edges when one of the dimensions is 2
        # and the system is periodic
        if x_dimension == 2 and site % 2 == 1:
            right_neighbour = None
        if y_dimension == 2 and site >= x_dimension:
            bottom_neighbour = None

        # Add hopping terms with neighbours to the right and bottom.
        if right_neighbour is not None:
            hubbard_model_horizontal += hoppingTerm(2*(site), 2*(right_neighbour), -t)
            hubbard_model_horizontal += hoppingTerm(2*(site) + 1, 2*(right_neighbour) + 1, -t)
        if bottom_neighbour is not None:
            hubbard_model_vertical += hoppingTerm(2*(site), 2*(bottom_neighbour), -t)
            hubbard_model_vertical += hoppingTerm(
                2*(site) + 1, 2*(bottom_neighbour) + 1, -t
            )

        # Add local pair Coulomb interaction terms.
        if isInteracting:
            hubbard_model_repulsion += coulombInteractionTerm(
                2*(site), 2*(site)+1
            )

    return {"horizontal": hubbard_model_horizontal,
        "vertical": hubbard_model_vertical,
        "repulsion": hubbard_model_repulsion}

def fermionOperatorToCircuit(*qubits, pauliStrings, theta, coefficient=1.0):
    """
    Convert a FermionOperator to a sequence of Cirq gates.
    
    Args:
    - qubits (tuple): The qubits to apply the gates to.
    - pauliStrings (FermionOperator): The Fermion operator representing the Pauli strings.
    - theta (float): The angle for the rotation gates.
    - coefficient (float, optional): Coefficient for the transformation. Default is 1.0.
    
    Yields:
    - cirq.Operation: The Cirq operations to apply the Fermion operator.
    """
    rot_dic = {'X': lambda q,s: cirq.ry(-1*s*numpy.pi/2).on(q),
            'Y': lambda q,s: cirq.rx(-1*s*numpy.pi/2).on(q),
            'Z': lambda q,s: cirq.I.on(q)}
      
    qubitList = [*qubits]
    operatorList = list(vfop_to_qop(theta=theta, fop=pauliStrings, coefficient=coefficient))
    
    for i, operator in enumerate(operatorList):
        paulis = list(operator.terms.keys())[0]
        factor = numpy.real(-1j*operator.terms[paulis])

        for qbt, pau in paulis:
            yield rot_dic[pau](qubitList[qbt], 1.0)

        if len(paulis) > 1:
            for j in range(len(paulis)-1):
                yield cirq.CNOT(qubitList[paulis[j][0]], qubitList[paulis[j+1][0]])  

        if len(paulis) > 0:
            yield cirq.rz(factor).on(qubitList[paulis[-1][0]])

        if len(paulis) > 1:
            for j in range(len(paulis)-1, 0, -1):
                yield cirq.CNOT(qubitList[paulis[j-1][0]], qubitList[paulis[j][0]])

        for qbt, pau in paulis:
            yield rot_dic[pau](qubitList[qbt], -1.0)

def initialCircuit(qubits):
    """
    Create the initial circuit with a Gaussian state.
    
    Args:
    - qubits (list of cirq.LineQubit): The qubits to prepare the Gaussian state on.
    
    Returns:
    - cirq.Circuit: The initial quantum circuit.
    """
    quadraticHamiltonian = get_quadratic_hamiltonian(sum(createHubbardHamiltonian(isInteracting=False).values()))
    gaussianState = openfermion.circuits.prepare_gaussian_state(qubits, quadraticHamiltonian)
    circuit = cirq.Circuit()
    circuit.append(gaussianState)
    return circuit

def variatonalHamiltonianCircuit(*qubits, params):
    """
    Create the variational Hamiltonian circuit.
    
    Args:
    - qubits (tuple): The qubits to apply the gates to.
    - params (list of float): Parameters for the variational circuit.
    
    Yields:
    - cirq.Operation: The Cirq operations for the variational Hamiltonian.
    """
    H_h = createHubbardHamiltonian()["horizontal"]
    H_v = createHubbardHamiltonian()["vertical"]
    H_U = createHubbardHamiltonian()["repulsion"]

    for i in range(0, 3*S, 3):
        yield fermionOperatorToCircuit(*qubits, pauliStrings=H_U, theta=params[i], coefficient=0.5)
        yield fermionOperatorToCircuit(*qubits, pauliStrings=H_v, theta=params[i+1])
        yield fermionOperatorToCircuit(*qubits, pauliStrings=H_h, theta=params[i+2])
        yield fermionOperatorToCircuit(*qubits, pauliStrings=H_U, theta=params[i], coefficient=0.5)

def cost_function(params):
    """
    Evaluate the cost function for the given parameters.
    
    Args:
    - params (list of float): Parameters for the variational circuit.
    
    Returns:
    - float: The real part of the expectation value of the Hamiltonian.
    """
    simulator = cirq.Simulator()
    result = simulator.simulate(prepareCircuit(params))
    sim_state = result.final_state_vector
    H = openfermion.get_sparse_operator(sum(createHubbardHamiltonian().values()))
    return numpy.real(openfermion.expectation(H, sim_state))

def cost_functionNoisy(params):
    """
    Evaluate the cost function with noise for the given parameters.
    
    Args:
    - params (list of float): Parameters for the variational circuit.
    
    Returns:
    - float: The real part of the expectation value of the Hamiltonian under noise.
    """
    simulator = cirq.Simulator()
    result = simulator.simulate(prepareCircuit(params, isNoisy=True))
    sim_state = result.final_state_vector
    H = openfermion.get_sparse_operator(sum(createHubbardHamiltonian().values()))
    return numpy.real(openfermion.expectation(H, sim_state))

def prepareCircuit(params, isNoisy=False):
    """
    Prepare the quantum circuit with the given parameters.
    
    Args:
    - params (list of float): Parameters for the variational circuit.
    - isNoisy (bool, optional): Whether to include noise in the circuit. Default is False.
    
    Returns:
    - cirq.Circuit: The prepared quantum circuit.
    """
    qubits = cirq.LineQubit.range(n_spin_orbitals)
    circuit = cirq.Circuit()
    circuit.append(initialCircuit(qubits))
    circuit.append(variatonalHamiltonianCircuit(*qubits, params=params))
    
    if isNoisy:
        # Add depolarizing noise
        noise = cirq.depolarize(p=0.01)
        noisy_circuit = cirq.Circuit(noise.on_each(circuit.all_qubits()))
        circuit.append(noisy_circuit)

    return circuit
