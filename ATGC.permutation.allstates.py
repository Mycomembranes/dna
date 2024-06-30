#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 14:29:07 2024

@author: mukshudahamed
"""

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from itertools import product

# Function to encode DNA sequence
def encode_dna_sequence(dna_sequence):
    encoding = {
        'A': np.array([1, 0], dtype=complex),
        'T': np.array([0, 1], dtype=complex),
        'G': np.array([1, 1], dtype=complex) / np.sqrt(2),
        'C': np.array([1, -1], dtype=complex) / np.sqrt(2)
    }
    encoded_seq = np.concatenate([encoding[nuc] for nuc in dna_sequence])
    return encoded_seq

# Generate all permutations of ATGC sequences of length 4
sequences = [''.join(seq) for seq in product('ATGC', repeat=4)]

# Time evolution function
def time_evolve(psi, H, t, hbar=1):
    return expm(-1j * H * t / hbar).dot(psi)

# Simulate for each sequence
for dna_sequence in sequences:
    encoded_state = encode_dna_sequence(dna_sequence)
    encoded_state = encoded_state / np.linalg.norm(encoded_state)  # Normalize the state
    
    # Define a simplified Hamiltonian for the small state space
    n = len(encoded_state)
    H_internal = np.eye(n, dtype=complex)
    H_interaction = np.random.rand(n, n) / 10  # Small random interaction
    H_interaction = (H_interaction + H_interaction.T.conj()) / 2  # Make it Hermitian
    H = H_internal + H_interaction

    # Initial state
    psi_initial = encoded_state

    # Evolve quantum states over time
    time_steps = 1000
    t_values = np.linspace(0, 100, time_steps)
    states_over_time = np.zeros((time_steps, n), dtype=complex)

    for i, t in enumerate(t_values):
        states_over_time[i] = time_evolve(psi_initial, H, t)

    # Plot Quantum State Encoding
    plt.figure(figsize=(10, 6))
    plt.bar(['A', 'T', 'C', 'G'], [np.abs(encoded_state[i])**2 for i in range(4)], color='blue')
    plt.title(f"Quantum State Encoding for DNA Sequence '{dna_sequence}'")
    plt.xlabel('Nucleotide')
    plt.ylabel('Probability')
    plt.savefig(f"quantum_state_encoding_{dna_sequence}.png")
    plt.close()

    # Plot Hamiltonian Construction
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(H), cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.title(f"Hamiltonian Construction for DNA Sequence '{dna_sequence}'")
    plt.xlabel('State Index')
    plt.ylabel('State Index')
    plt.savefig(f"hamiltonian_construction_{dna_sequence}.png")
    plt.close()

    # Plot Time Evolution of Quantum States
    plt.figure(figsize=(10, 6))
    for i in range(n):
        plt.plot(t_values, np.abs(states_over_time[:, i])**2, label=f"State {i}")
    plt.title(f"Time Evolution of Quantum States for DNA Sequence '{dna_sequence}'")
    plt.xlabel("Time")
    plt.ylabel("Probability")
    plt.legend()
    plt.savefig(f"time_evolution_quantum_states_{dna_sequence}.png")
    plt.close()

# Example plot for STDP and Synaptic Plasticity
t_delta = np.linspace(-10, 10, 200)
A_plus = 1
A_minus = 0.5
tau_plus = 1
tau_minus = 1
delta_H = np.where(t_delta > 0, A_plus * np.exp(-t_delta / tau_plus), -A_minus * np.exp(t_delta / tau_minus))

# Save STDP and Synaptic Plasticity Figure
plt.figure(figsize=(10, 6))
plt.plot(t_delta, delta_H, label="STDP", color='red')
plt.title("STDP and Synaptic Plasticity")
plt.xlabel("Time Difference (Δt)")
plt.ylabel("Synaptic Weight Change (ΔH)")
plt.legend()
plt.savefig("stdp_synaptic_plasticity.png")
plt.close()

# Save Simulation Results Figure
plt.figure(figsize=(10, 6))
for i in range(n):
    plt.plot(t_values, np.abs(states_over_time[:, i])**2, label=f"State {i}")
plt.title("Simulation Results")
plt.xlabel("Time")
plt.ylabel("Probability")
plt.legend()
plt.savefig("simulation_results.png")
plt.close()