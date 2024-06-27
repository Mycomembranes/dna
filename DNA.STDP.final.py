#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:45:53 2024

@author: mukshudahamed
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Define basis states for A, T, G, C
A = np.array([1, 0], dtype=complex)
T = np.array([0, 1], dtype=complex)
G = np.array([1, 1], dtype=complex) / np.sqrt(2)
C = np.array([1, -1], dtype=complex)/ np.sqrt(2)

# Define bases and encode DNA sequence
def initialize_bases():
    return {'A': A, 'T': T, 'G': G, 'C': C}

def encode_dna(sequence, bases):
    Psi_DNA = bases[sequence[0]]
    for base in sequence[1:]:
        Psi_DNA = np.kron(Psi_DNA, bases[base])
    return Psi_DNA

# Add random perturbation and normalize
def prepare_state(Psi_DNA):
    Psi_DNA += np.random.normal(0, 0.1, Psi_DNA.shape)
    return Psi_DNA / np.linalg.norm(Psi_DNA)

# Hamiltonian for biological systems
def create_hamiltonian(size, internal_energy, interaction_energy, external_influences):
    H_internal = np.diag(np.random.normal(internal_energy, 0.1, size))
    H_interaction = np.random.normal(interaction_energy, 0.05, (size, size))
    H_interaction = (H_interaction + H_interaction.T.conj()) / 2  # Make it Hermitian
    H_external = np.diag(np.random.normal(external_influences, 0.1, size))
    return H_internal + H_interaction + H_external

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

# Time evolution of quantum states with neuronal restriction
def time_evolution_with_tunneling_and_stdp(Psi_DNA, H_DNA, T_tunnel, dt, steps):
    values = []
    for step in range(steps):
        Psi_DNA = (np.eye(len(Psi_DNA)) - 1j * (H_DNA + T_tunnel) * dt) @ Psi_DNA
        value = Psi_DNA.conj().T @ Psi_DNA
        values.append(np.real(value))
        if step > 0:
            delta_t = dt
            H_DNA = stdp_update(H_DNA, delta_t)

    # Normalize the values to fit the specified range
    min_val, max_val = min(values), max(values)
    scale_low, scale_high = 0.9999999999999, 1.00000000000001
    scaled_values = [scale_low + (scale_high - scale_low) * (val - min_val) / (max_val - min_val) for val in values]
    return scaled_values

# Short simulation
sequence = "ATGC"  # Truncated sequence for demonstration
bases = initialize_bases()
Psi_DNA = encode_dna(sequence, bases)
Psi_DNA = prepare_state(Psi_DNA)
H_DNA = create_hamiltonian(len(Psi_DNA), 1.0, 0.5, 0.3)

# Define a simple tunneling operator
T_tunnel = np.eye(len(Psi_DNA))

# Simulation parameters
dt = 0.1
time_steps = 100  # Adjusted for demonstration, mess with time steps for different behaviors
values = time_evolution_with_tunneling_and_stdp(Psi_DNA, H_DNA, T_tunnel, dt, time_steps)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(list(range(1, len(values) + 1)), values, marker='o')
plt.title('Simulated Consciousness Values Over Time with Neuronal Restrictions')
plt.xlabel('Time Step')
plt.ylabel('Consciousness Value')
plt.grid(True)
plt.show()
