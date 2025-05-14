import numpy as np

def edging(F):
    if F < 0.55:
        raise NameError("You dumb ass, use a higher fidelity")
    edge = np.sqrt(3)/2
    iter = 0
    while True:
        iter += 1
        psucc = F**2 + 2 * F * (1 - F) / 3 + 5 * ((1 - F) / 3) ** 2
        F = (F**2 + ((1 - F) / 3)**2) / psucc
        if F > edge:
            return iter

print(edging(0.7))