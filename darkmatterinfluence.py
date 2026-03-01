import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

# Define a simple quantum state for dark matter
def initialize_dark_matter_states(num_particles):
    # Initialize states as complex numbers for each particle; two states per particle
    states = np.random.rand(num_particles, 2) + 1j * np.random.rand(num_particles, 2)
    states = states / np.linalg.norm(states, axis=1, keepdims=True)  # Normalize states
    return states

# Define the Hamiltonian for the system
def hamiltonian(num_particles):
    # Create a larger Hamiltonian to match the flattened state dimensions (num_particles * 2)
    H = -1 * np.eye(num_particles * 2)  # Gravitational well potential, simplified
    interaction = np.random.rand(num_particles * 2, num_particles * 2)
    H += (interaction + interaction.T) / 2  # Symmetric interaction matrix
    return H

# Time evolution of the system
def time_evolve(states, H, dt, hbar=1):
    # Flatten the states to match the Hamiltonian dimension for matrix multiplication
    flat_states = states.flatten()
    U = expm(-1j * H * dt / hbar)
    evolved_states = U @ flat_states
    return evolved_states.reshape(num_particles, 2)

# Parameters
num_particles = 10  # Number of particles influenced by dark matter
time_steps = 20
dt = 0.1  # Time step for evolution

# Initialization
states = initialize_dark_matter_states(num_particles)
H = hamiltonian(num_particles)

# Simulation over time
states_over_time = np.zeros((time_steps, num_particles, 2), dtype=complex)
for t in range(time_steps):
    states = time_evolve(states, H, dt)
    states = states / np.linalg.norm(states, axis=1, keepdims=True)  # Renormalize states
    states_over_time[t] = states

# Visualization
plt.figure(figsize=(12, 6))
for i in range(num_particles):
    plt.plot(np.abs(states_over_time[:, i, 0])**2, label=f'Particle {i+1}')
plt.title("Evolution of Dark Matter Influenced Quantum States Over Time")
plt.xlabel("Time Step")
plt.ylabel("State Probability Amplitude")
plt.legend()
plt.show()