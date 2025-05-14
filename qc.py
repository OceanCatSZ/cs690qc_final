import numpy as np
from qutip import *

class Node:
    def __init__(self, name):
        self.name = name
        self.prev = None
        self.next = None  # adjacent nodes
        self.prevdist = 0
        self.nextdist = 0

    def set_next(self, node, distance):
        self.next = node
        self.nextdist = distance
        node.prev = self
        node.prevdist = distance
    
    def set_prev(self, node, distance):
        self.prev = node
        self.prevdist = distance
        node.next = self
        node.nextdist = distance


class Entanglement:
    def __init__(self, node1: Node, node2: Node, fidelity: float, T_depol: float):
        self.node1 = node1
        self.node2 = node2
        self.fidelity = fidelity

        # Ideal Bell state: |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
        self.phi_plus = (tensor(basis(2, 0), basis(2, 0)) + tensor(basis(2, 1), basis(2, 1))).unit()
        self.phi_plus_dm = ket2dm(self.phi_plus)

        # Construct Werner state: ρ = F |Φ⁺⟩⟨Φ⁺| + (1 - F) * (I/4)
        identity = tensor(qeye(2), qeye(2))
        self.rho = fidelity * self.phi_plus_dm + (1 - fidelity) * (identity / 4)
        self.T_depol = T_depol

    def calFid(self, target_state=None):
        if target_state is None:
            target_state = self.phi_plus_dm
        return fidelity(self.rho, target_state)
    
    def decohere(self, t):
        

def purify(state1, state2):
    return 

def build_uniform_chain(L_total, N_repeaters=4):
    # Step size per link
    link_distance = L_total / (N_repeaters + 1)
    
    # Create nodes
    node_names = ["A"] + [f"R{i}" for i in range(1, N_repeaters + 1)] + ["B"]
    nodes = {name: Node(name) for name in node_names}

    # Connect nodes linearly
    for i in range(len(node_names) - 1):
        node1 = nodes[node_names[i]]
        node2 = nodes[node_names[i + 1]]
        node1.set_next(node2, distance=link_distance)

    return nodes

def generate_next_entanglement_BK(node1, node2, L_att=22.5, tau_attempt=1.0):
    """
    Attempts to generate an entangled pair between two *connected* nodes via the BK scheme.
    Returns (Entanglement, time_taken) if successful.
    Raises an error if the nodes are not direct neighbors.
    """
    # Ensure bidirectional connection
    # if node2 not in node1.neighbors or node1 not in node2.neighbors:
    #     raise ValueError(f"{node1.name} and {node2.name} are not connected as neighbors.")

    distance = node1.nextdist/2
    eta = np.exp(-distance / L_att)
    p_success = eta**2/2

    # Sample geometric number of attempts until success
    attempts = np.random.geometric(p_success)
    time_taken = attempts * tau_attempt

    fidelity = (3 * eta + 1) / 4
    ent = Entanglement(node1, node2, p=fidelity)
    return ent, time_taken

    
def main():
    # Let's assume a linked list structure
    alice = Node("A")
    Bob = Node("B")
    L_total = 200 #km
    L_att = 22.5

    nodes = build_uniform_chain(L_total)
    print(nodes)
    return

if __name__ == '__main__':
    main()
    print("fin")