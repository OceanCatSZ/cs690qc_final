# Quantum Repeater Chain Simulation

This project simulates a **quantum repeater network** for long-distance quantum communication. It models entanglement generation, purification, and swapping across a chain of quantum nodes (repeaters), tracking the generation time, fidelity, and cost (in terms of sacrificed Werner states) at different stages.

## Features

- Supports configurable total distances (e.g., 200km, 500km, 1000km)
- Simulates entanglement generation using the **Briegel–Dür–Cirac–Zoller (BDCZ)** protocol
- Implements **Werner-state-based purification**
- Performs **entanglement swapping** with fidelity degradation due to depolarization
- Tracks:
  - Final fidelity of the entangled state
  - Time required to establish long-distance entanglement
  - Resource cost in terms of consumed Werner states

## Dependencies

- Python 3.8+
- [QuTiP](http://qutip.org/)
- NumPy
- Matplotlib

Install requirements via:

```bash
pip install qutip numpy matplotlib
