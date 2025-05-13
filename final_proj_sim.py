import numpy as np
from qutip import basis, ket2dm, tensor


bell_ket = (tensor(basis(2,0), basis(2,0)) + tensor(basis(2,1), basis(2,1))).unit()
rho = ket2dm(bell_ket)  # convert to density matrix

print(bell_ket)
print(basis(4, 3))


class Graph:
    def __init__(self):
        self.adjacency_list = {}
        self.current_index = 0
    
    def get_nodes(self):
        return self.adjacency_list.keys()

    def add_node(self, n = None):
        if n is None:
            n = self.current_index
        self.adjacency_list[n] = []
        self.current_index += 1

    # Check if i's adjaency list contains n, also its index 
    def contains_node(self, adj_list_i, n):
        for i, (j, _) in enumerate(adj_list_i):
            if j == n:
                return True, i
        return False, -1
    
    # Adds a link with the defined weight. Update the weight if the link exists
    # The link is added both ways
    def add_link(self, n1, n2, weight):
        if n1 not in self.adjacency_list or n2 not in self.adjacency_list:
            return # Not a valid edge to add
        n1_contains_n2, index1 = self.contains_node(self.adjacency_list[n1], n2)
        n2_contains_n1, index2 = self.contains_node(self.adjacency_list[n2], n1)
        if n1_contains_n2:
            self.adjacency_list[n1][index1][1] = weight # Update weight
            self.adjacency_list[n2][index2][1] = weight # Update weight in the other direction
        else:
            self.adjacency_list[n1].append((n2, weight))
            self.adjacency_list[n2].append((n1, weight))

class QubitGroup:
    def __init__(self, indices, rho):
        self.qubit_indices = indices.copy()
        self.rho = rho

class Sim:
    def __init__(self, graph:Graph):
        groups = {}
        for node in graph.get_nodes():
            ket0 = basis(2, 0)
            rho = ket2dm(ket0)
            groups[node] = QubitGroup(node, rho)

        

g = Graph()
g.add_node()
g.add_node()
g.add_link(0, 1, 100)
print(g.adjacency_list)

        
# class Link:
#     def __init__(self, node1, node2, length_km, L_att=22):
#         self.node1 = node1
#         self.node2 = node2
#         self.length = length_km
#         self.p_success = np.exp(-length_km / L_att)  # one-way loss on both photons

