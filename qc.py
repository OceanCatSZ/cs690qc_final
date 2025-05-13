import numpy as np
from qutip import basis, ket2dm, tensor, fidelity

class Entanglement:
    def __init__(self, p):
        # Define Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
        phi_plus = (tensor(basis(2, 0), basis(2, 0)) + tensor(basis(2, 1), basis(2, 1))).unit()
        self.phi_plus = phi_plus
        self.phi_plus_dm = ket2dm(phi_plus)
        
        # Mixed state: ρ = p * |Φ⁺⟩⟨Φ⁺| + (1 - p) * I/4
        identity = tensor([ket2dm(basis(2, 0)) + ket2dm(basis(2, 1))] * 2)  # I = sum_k |k⟩⟨k|
        self.rho = p * self.phi_plus_dm + (1 - p) * (identity / 4)
        self.p = p
    
    def calFid(self, target_state=None):
        # Default: fidelity with respect to |Φ⁺⟩
        if target_state is None:
            target_state = self.phi_plus_dm
        return fidelity(self.rho, target_state)

def purify(state1, state2):
    return 

def main():
    return

if __name__ == '__main__':
    print("fin")