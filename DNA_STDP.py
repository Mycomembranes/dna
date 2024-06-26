import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

# Define basis states for nucleotides
def encode_dna_sequence(dna_sequence):
    encoding = {
        'A': np.array([1, 0], dtype=complex),
        'T': np.array([0, 1], dtype=complex),
        'G': np.array([1, 1], dtype=complex) / np.sqrt(2),
        'C': np.array([1, -1], dtype=complex) / np.sqrt(2)
    }
    encoded_seq = np.concatenate([encoding[nuc] for nuc in dna_sequence])
    return encoded_seq

# DNA sequence for simulation
dna_sequence = "GGGGCCGCCG"
encoded_state = encode_dna_sequence(dna_sequence)
encoded_state = encoded_state / np.linalg.norm(encoded_state)  # Normalize the state

# Define a simplified Hamiltonian for the small state space
n = len(encoded_state)
H_internal = np.eye(n, dtype=complex)
H_interaction = np.random.rand(n, n) / 10  # Small random interaction
H_interaction = (H_interaction + H_interaction.T.conj()) / 2  # Make it Hermitian
H = H_internal + H_interaction

# Define neuronal transmission rules (simplified STDP model)
def stdp_update(H, delta_t):
    A_plus = 0.005
    A_minus = 0.005
    tau_plus = 20.0
    tau_minus = 20.0
    
    if delta_t > 0:
        return H + A_plus * np.exp(-delta_t / tau_plus)
    else:
        return H - A_minus * np.exp(delta_t / tau_minus)

# Time evolution function
def time_evolve(psi, H, t, hbar=1):
    return expm(-1j * H * t / hbar).dot(psi)

# Initial state
psi_initial = encoded_state

# Evolve quantum states over time
time_steps = 100  # Shorter time steps for quick simulation
t_values = np.linspace(0, 1, time_steps)
states_over_time = np.zeros((time_steps, n), dtype=complex)

for i, t in enumerate(t_values):
    states_over_time[i] = time_evolve(psi_initial, H, t)
    if i > 0:
        delta_t = t_values[i] - t_values[i-1]
        H = stdp_update(H, delta_t)

# Plotting the evolution of the quantum state over time
plt.figure(figsize=(10, 6))
for i in range(n):
    plt.plot(t_values, np.abs(states_over_time[:, i])**2, label=f"State {i}")

plt.title("Evolution of Quantum States Over Time with Neuronal Transmission Rules")
plt.xlabel("Time")
plt.ylabel("Probability")
plt.legend()
plt.show()
