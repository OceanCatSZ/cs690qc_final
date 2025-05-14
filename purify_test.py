import numpy as np

F = 0.5
flist = []
for i in range(16):
    psucc = F**2 + 2 * F * (1 - F) / 3 + 5 * ((1 - F) / 3) ** 2
    F = (F**2 + ((1 - F) / 3)**2) / psucc
    flist.append(F)
print(flist)