import numpy as np
import math
from qutip import *

c = 2 * 10**2
operation_time = 10 ** -3

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
    def __init__(self, node1: Node, node2: Node, fidel: float, T_depol: float = 100):
        self.node1 = node1
        self.node2 = node2
        self.fidelity = fidel

        # Ideal Bell state: |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
        self.phi_plus = (tensor(basis(2, 0), basis(2, 0)) + tensor(basis(2, 1), basis(2, 1))).unit()
        self.phi_plus_dm = ket2dm(self.phi_plus)
    
        # Construct Werner state: ρ = F |Φ⁺⟩⟨Φ⁺| + (1 - F) * (I/4)
        identity = tensor(qeye(2), qeye(2))
        self.rho = self.fidelity * self.phi_plus_dm + (1 - self.fidelity) * (identity / 4)
        self.T_depol = T_depol

    def calFid(self, target_state=None):
        if target_state is None:
            target_state = self.phi_plus_dm
        self.fidelity = fidelity(self.rho, target_state)
        return self.fidelity
    
    def depolarize(self, t_ms: float):
        """
        Depolarizes self.rho for time t_ms (in milliseconds),
        assuming characteristic decoherence time T_depol_ms.
        """
        epsilon = 1 - np.exp(-t_ms / self.T_depol)
        identity = tensor(qeye(2), qeye(2))
        self.rho = (1 - epsilon) * self.rho + epsilon * (identity / 4)


def purify(state1, state2):
    return 

def build_uniform_chain(L_total, N_repeaters=0):
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
    ent_swapped = Entanglement(ent1.node1, ent2.node2, fidel=fidelity(rho_AB, ent1.phi_plus_dm))
    ent_swapped.rho = rho_AB

    return ent_swapped


def generate_next_entanglement_BK(node1:Node, L_att=22.5):
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
    tau_attempt = 2 * distance / c
    eta = np.exp(-distance / L_att)
    p_success = eta**2/2

    # Sample geometric number of attempts until success
    attempts = np.random.geometric(p_success)
    time_taken = attempts * tau_attempt

    # fidelity = (3 * eta + 1) / 4
    fidelity = 1
    ent = Entanglement(node1, node1.next, fidel=fidelity)
    return ent, time_taken

def get_dist_in_ent(node_dict, ent: Entanglement):
    pointer = ent.node1
    target = ent.node2
    distance = 0
    while pointer.next != target:
        distance += node_dict[pointer.name].nextdist
        pointer = pointer.next
    return distance
    
def main():
    # Let's assume a linked list structure
    alice = Node("A")
    Bob = Node("B")
    L_total = 200 # km
    L_att = 22.5
    t_total = 0

    ### step 1: initial ent generation and depolarize:
    nodes = build_uniform_chain(L_total)
    root = nodes["A"]
    pointer = root
    entlist = []
    gentime = []
    while pointer.next != None:
        ent, t = generate_next_entanglement_BK(pointer)
        if ent is None:
            continue
        entlist.append(ent)
        gentime.append(t)
        pointer = pointer.next
    maxt = max(gentime)
    t_total += maxt
    for i in range(len(entlist)):
        t_depol = maxt - gentime[i]
        entlist[i].depolarize(t_depol)
    
    ### step 2: 
    while len(entlist) != 1:
        next_entlist = []
        sub_total_time = 0
        for i in range(0, len(entlist), 2):
            new_ent = None
            if i + 1 >= len(entlist):
                entlist[i].depolarize(operation_time)
                new_ent = entlist[i]
            else:
                entlist[i].depolarize(operation_time)
                entlist[i + 1].depolarize(operation_time)
                new_ent = entanglement_swap(entlist[i], entlist[i + 1])
            t_depol = get_dist_in_ent(nodes, new_ent) / c
            sub_total_time = max(sub_total_time, t_depol)
            new_ent.depolarize(t_depol)
            next_entlist.append(new_ent)
        t_total += sub_total_time + operation_time
        entlist = next_entlist
    print(entlist[0].calFid(entlist[0].phi_plus_dm))
    print(t_total)
    return

if __name__ == '__main__':
    main()
    print("fin")