import numpy as np
import matplotlib.pyplot as plt
import math
from qutip import *
import ctypes

c = 2 * 10**2
operation_time = 10 ** -3 # quantum gates operation time

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
    def __init__(self, node1: Node, node2: Node, fidel: float, T_depol: float = 10):
        self.node1 = node1
        self.node2 = node2
        self.fidelity = fidel

        # Ideal Bell state: |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
        self.phi_plus = (tensor(basis(2, 0), basis(2, 0)) + tensor(basis(2, 1), basis(2, 1))).unit()
        self.phi_plus_dm = ket2dm(self.phi_plus)
    
        # Construct Werner state: ρ = F |Φ⁺⟩⟨Φ⁺| + (1 - F) * (I/4)
        self.identity = tensor(qeye(2), qeye(2))
        self.rho = self.fidelity * self.phi_plus_dm + (1 - self.fidelity) * ((self.identity - self.phi_plus_dm) / 3)
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
        self.calFid(self.phi_plus_dm)
    
    def update_fidelity(self, new_fid):
        self.fidelity = new_fid
        self.rho = self.fidelity * self.phi_plus_dm + (1 - self.fidelity) * ((self.identity - self.phi_plus_dm) / 3)

def edging(F, final_F):
    if F < 0.5:
        return 0
    iter = 0
    while True:
        if F > final_F:
            return iter
        psucc = F**2 + 2 * F * (1 - F) / 3 + 5 * ((1 - F) / 3) ** 2
        F = (F**2 + ((1 - F) / 3)**2) / psucc
        iter += 1

def purify(ent: Entanglement):
    F = ent.fidelity
    psucc = F**2 + 2 * F * (1 - F) / 3 + 5 * ((1 - F) / 3) ** 2
    F = (F**2 + ((1 - F) / 3)**2) / psucc
    ent.update_fidelity(F)

def find_init_fid(final_F, nlevels):
    # levels = math.floor(np.log2(n))
    init_F = final_F
    for _ in range(nlevels):
        init_F = (np.sqrt(12 * init_F - 3) + 1) / 4
    return init_F


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
    rho_proj = projector * rho_full * projector #Projector is hermitian here
    rho_proj = rho_proj / rho_proj.tr() #Post measurement state of outcome 1

    # Trace out internal qubits R1 and R2 (indices 1 and 2)
    rho_AB = rho_proj.ptrace([0, 3])

    # Build resulting entanglement object
    ent_swapped = Entanglement(ent1.node1, ent2.node2, fidel=fidelity(rho_AB, ent1.phi_plus_dm), T_depol = ent1.T_depol)
    ent_swapped.rho = rho_AB

    return ent_swapped


def generate_next_entanglement_BK(node1:Node, L_att=22.5, t_depol=1):
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
    tau_attempt = 2 * distance / c # Send a photon to the middle heralding station, then the heralding station send back
    eta = np.exp(-distance / L_att)
    p_success = eta**2/2

    # Sample geometric number of attempts until success
    attempts = ctypes.c_uint64(np.random.geometric(p_success)).value
    time_taken = attempts * tau_attempt

    ent = Entanglement(node1, node1.next, fidel=1, T_depol=t_depol)
    return ent, time_taken

def get_dist_in_ent(node_dict, ent: Entanglement):
    pointer = ent.node1
    target = ent.node2
    distance = 0
    while pointer.next != target:
        distance += node_dict[pointer.name].nextdist
        pointer = pointer.next
    return distance

def auto_purify_entlist(nodes, entlist, target_fidelity, safety_iters = 1):
    times = []
    werner_states_cost = 0
    for ent in entlist:
        iters = edging(entlist[0].fidelity, target_fidelity) + safety_iters
        werner_states_cost += 2 ** iters
        tau = get_dist_in_ent(nodes, ent) / c
        for _ in range(iters):
            ent.depolarize(operation_time)
            purify(ent)
            ent.depolarize(tau)
        times.append((tau + operation_time) * iters)
    t_total = max(times)
    return t_total, werner_states_cost
    
def sim(L_total, num_node, T_DEPOL):
    # Let's assume a linked list structure
    # L_total = 200 # km
    L_att = 22.5
    t_total = 0
    target_final_fid = 0.9
    werner_states_costs = []

    ### step 1: initial ent generation and depolarize:
    # num_node = 6
    nodes = build_uniform_chain(L_total, num_node)
    root = nodes["A"]
    
    pointer = root
    entlist = []
    gentime = []
    while pointer.next != None:
        ent, t = generate_next_entanglement_BK(pointer, L_att, T_DEPOL)
        ent.depolarize(ent.node1.nextdist / c)
        if ent is None:
            continue
        entlist.append(ent)
        gentime.append(t)
        pointer = pointer.next
    maxt = max(gentime)

    t_total += maxt
    # Simulate other repeaters waiting for the slowest one
    for i in range(len(entlist)):
        temp = maxt - gentime[i]
        entlist[i].depolarize(temp)

    # Simulating purifying the initial entanglement layer
    print(f"Initial fidelity is {entlist[0].fidelity}")
    target_init_fid = find_init_fid(target_final_fid, math.floor(np.log2(len(entlist))))
    print(f"Our desired initial fidelity is: {target_init_fid}")

    time_taken, cost = auto_purify_entlist(nodes, entlist, target_init_fid, safety_iters=0)
    werner_states_costs.append(cost)
    # print(f'Werner states cost in current level: {cost}')
    t_total += time_taken
    print(f"Fidelity after purification is {entlist[0].fidelity}")
    
    ### step 2: Start with entanglement swap
    while len(entlist) != 1:
        # Entanglement swap process
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

        # Perform a purification
        target_init_fid = find_init_fid(target_final_fid, math.floor(np.log2(len(entlist))))
        # print(target_init_fid)
        # print(entlist[0].fidelity)
        time_taken, cost = auto_purify_entlist(nodes, entlist, target_final_fid, safety_iters=0)
        # print(entlist[0].fidelity)
        # print()
        werner_states_costs.append(cost)
        t_total += time_taken    

    print(f"Final fidelity is {entlist[0].calFid(entlist[0].phi_plus_dm)}")
    print(f"total time taken for this process is {t_total}")
    print(f"Werner states cost for each level: {werner_states_costs}")
    return werner_states_costs, t_total, entlist[0].calFid(entlist[0].phi_plus_dm)

def main():
    # question 1-3
    # sample_number = 10
    # num_node = 6
    # t_depol_list = np.arange(1, 101, 1)
    # fid_list = []
    # t_list = []
    # cost_list = []
    # for i in t_depol_list:
    #     fid_list_samples = []
    #     t_list_samples = []
    #     cost_list_samples = []
    #     for _ in range(sample_number):
    #         num_ent = num_node + 1
    #         cost, t, fid = sim(200, num_node, i)
    #         cost_avg_based_on_level = []
    #         for c in cost:
    #             cost_avg_based_on_level.append(c / num_ent)
    #             num_ent = math.ceil(num_ent / 2)
    #         fid_list_samples.append(fid)
    #         t_list_samples.append(t)
    #         cost_list_samples.append(cost_avg_based_on_level)
    #     fid_list.append(np.mean(fid_list_samples))
    #     t_list.append(np.mean(t_list_samples))
    #     cost_total = 1
    #     costs = [sum(col) / len(col) for col in zip(*cost_list_samples)]
    #     for c in costs:
    #         cost_total *= c
    #     cost_list.append(cost_total)
    # plt.plot(t_depol_list, t_list)
    # plt.title("T_depol vs generation time")
    # plt.xlabel("T_depol parameter")
    # plt.ylabel("generation time/t")
    # plt.show()
    
    # threshold_fid = 0.9
    # for i in range(len(t_depol_list)):
    #     if fid_list[i] >= threshold_fid:
    #         print(i)
    #         break
    # plt.plot(t_depol_list, fid_list)
    # plt.title("T_depol vs final fidelity")
    # plt.xlabel("T_depol parameter")
    # plt.ylabel("final fidelity")
    # plt.show()
    
    # plt.plot(t_depol_list, cost_list)
    # plt.title("T_depol vs final cost")
    # plt.xlabel("T_depol parameter")
    # plt.ylabel("log # Werner State Sacrificed")
    # plt.yscale('log')
    # plt.show()
    
    # question 4
    
    sample_number = 10
    num_node_list = np.arange(4, 40, 1)
    distance_list = [200, 500, 1000]
    t_depol = 10
    for d in distance_list:
        fid_list = []
        t_list = []
        cost_list = []
        for n in num_node_list:
            fid_list_samples = []
            t_list_samples = []
            cost_list_samples = []
            for _ in range(sample_number):
                num_ent = n + 1
                cost, t, fid = sim(d, n, t_depol)
                cost_avg_based_on_level = []
                for c in cost:
                    cost_avg_based_on_level.append(c / num_ent)
                    num_ent = math.ceil(num_ent / 2)
                fid_list_samples.append(fid)
                t_list_samples.append(t)
                cost_list_samples.append(cost_avg_based_on_level)
            fid_list.append(np.mean(fid_list_samples))
            t_list.append(np.mean(t_list_samples))
            cost_total = 1
            costs = [sum(col) / len(col) for col in zip(*cost_list_samples)]
            for c in costs:
                cost_total *= c
            cost_list.append(cost_total)
        plt.plot(num_node_list, t_list)
        plt.title(f"num_nodes vs generation time for L = {d}")
        plt.xlabel("# of nodes")
        plt.ylabel("generation time/t")
        plt.show()
    
        threshold_fid = 0.9
        for i in range(len(num_node_list)):
            if fid_list[i] >= threshold_fid:
                print(i)
                break
        plt.plot(num_node_list, fid_list)
        plt.title(f"num_nodes vs final fidelity for L = {d}")
        plt.xlabel("# of nodes")
        plt.ylabel("final fidelity")
        plt.show()
        
        plt.plot(num_node_list, cost_list)
        plt.title(f"num_nodes vs final cost for L = {d}")
        plt.xlabel("# of nodes")
        plt.ylabel("log # Werner State Sacrificed")
        plt.yscale('log')
        plt.show()

if __name__ == '__main__':
    main()