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
    def __init__(self, node1: Node, node2: Node, fidel: float, T_depol: float):
        self.node1 = node1
        self.node2 = node2
        self.fidelity = fidel

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
    
    def depolarize(self, t_ms: float, T_depol_ms: float = 5.0):
        """
        Depolarizes self.rho for time t_ms (in milliseconds),
        assuming characteristic decoherence time T_depol_ms.
        """
        epsilon = 1 - np.exp(-t_ms / T_depol_ms)
        identity = tensor(qeye(2), qeye(2))
        self.rho = (1 - epsilon) * self.rho + epsilon * (identity / 4)


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

def entanglement_swap(ent1: Entanglement, ent2: Entanglement) -> Entanglement:
    # Validate topology: ent1.node2 must be ent2.node1
    if ent1.node2 != ent2.node1:
        raise ValueError(
            f"Cannot swap: middle nodes do not match. "
            f"Got {ent1.node2.name} and {ent2.node1.name}."
        )

    # Combine full state: A–R1–R2–B
    rho_full = tensor(ent1.rho, ent2.rho)

    # Project onto Φ⁺ Bell state between R1 and R2 (qubits 1 and 2)
    bell = (tensor(basis(2,0), basis(2,0)) + tensor(basis(2,1), basis(2,1))).unit()
    P = ket2dm(bell)
    projector = tensor(qeye(2), P, qeye(2))  # Acts on qubits A-R1-R2-B

    # Apply projection and normalize
    rho_proj = projector * rho_full * projector
    rho_proj = rho_proj / rho_proj.tr()

    # Trace out internal qubits R1 and R2 (indices 1 and 2)
    rho_AB = rho_proj.ptrace([0, 3])

    # Build resulting entanglement object
    ent_swapped = Entanglement(ent1.node1, ent2.node2, fidelity=fidelity(rho_AB, ent1.phi_plus_dm))
    ent_swapped.rho = rho_AB

    return ent_swapped


def generate_next_entanglement_BK(node1:Node, L_att=22.5, tau_attempt=1.0):
    """
    Attempts to generate an entangled pair between two *connected* nodes via the BK scheme.
    Returns (Entanglement, time_taken) if successful.
    Raises an error if the nodes are not direct neighbors.
    """
    # Ensure bidirectional connection
    # if node2 not in node1.neighbors or node1 not in node2.neighbors:
    #     raise ValueError(f"{node1.name} and {node2.name} are not connected as neighbors.")
    if node1.next is None:
        return None, None
    distance = node1.nextdist/2
    eta = np.exp(-distance / L_att)
    p_success = eta**2/2

    # Sample geometric number of attempts until success
    attempts = np.random.geometric(p_success)
    time_taken = attempts * tau_attempt

    fidelity = (3 * eta + 1) / 4
    ent = Entanglement(node1, node1.next, p=fidelity)
    return ent, time_taken

    
def main():
    # Let's assume a linked list structure
    alice = Node("A")
    Bob = Node("B")
    L_total = 200 #km
    L_att = 22.5

    nodes = build_uniform_chain(L_total)
    entlist = []
    gentime = []
    for key in nodes:
        ent, t = generate_next_entanglement_BK(nodes[key])
        if ent is None:
            continue
        entlist.append(ent)
        gentime.append(t)
    maxt = max(gentime)
    for i in range(len(entlist)):
        t_depol = maxt - gentime[i]
        entlist[i].depolarize(t_depol)
    
    return

if __name__ == '__main__':
    main()
    print("fin")