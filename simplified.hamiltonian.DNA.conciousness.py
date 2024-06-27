#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:07:31 2024

@author: mukshudahamed
"""

import numpy as np
import matplotlib.pyplot as plt

# Define basis states for A, T, G, C
A = np.array([1, 0])
T = np.array([0, 1])
G = np.array([1, 1]) / np.sqrt(2)
C = np.array([1, -1])/np.sqrt(2)

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
def create_hamiltonian(size):
    return np.random.normal(0, 1, (size, size))

# Time evolution of quantum states
def time_evolution(Psi_DNA, H_DNA, dt, steps):
    values = []
    for _ in range(steps):
        Psi_DNA = (np.eye(len(Psi_DNA)) - 1j * H_DNA * dt) @ Psi_DNA
        value = np.real(Psi_DNA.conj().T @ Psi_DNA)
        values.append(value)
    return values

# Main execution function
def main():
    sequence = "GGGGCCGCCGG"  # Short sequence for demonstration
    bases = initialize_bases()
    Psi_DNA = encode_dna(sequence, bases)
    Psi_DNA = prepare_state(Psi_DNA)
    H_DNA = create_hamiltonian(len(Psi_DNA))

    # Simulation parameters
    dt = 0.1
    time_steps = 100  # Shorter simulation for demonstration
    values = time_evolution(Psi_DNA, H_DNA, dt, time_steps)

    # Print the calculated values
    for i, value in enumerate(values):
        print(f"Time step {i+1}: Consciousness Value = {value}")

    # Prepare data for plotting
    time_data = np.linspace(0, time_steps * dt, time_steps)
    value_data = np.array(values)

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(time_data, value_data, marker='o')
    plt.title('Simulated Consciousness Values Over Time')
    plt.xlabel('Time')
    plt.ylabel('Consciousness Value')
    plt.grid(True)
    plt.show()

# Run the main function
if __name__ == "__main__":
    main()
