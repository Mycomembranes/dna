
import numpy as np
import matplotlib.pyplot as plt

# Define basis states for A, T, G, C
A = np.array([1, 0])
T = np.array([0, 1])
G = np.array([1, 1]) / np.sqrt(2)
C = np.array([0, 0])

def initialize_bases():
    return {'A': A, 'T': T, 'G': G, 'C': C}

def encode_dna(sequence, bases):
    Psi_DNA = bases[sequence[0]]
    for base in sequence[1:]:
        Psi_DNA = np.kron(Psi_DNA, bases[base])
    return Psi_DNA

def prepare_state(Psi_DNA):
    Psi_DNA += np.random.normal(0, 0.1, Psi_DNA.shape)
    return Psi_DNA / np.linalg.norm(Psi_DNA)

def create_hamiltonian(size, internal_energy, interaction_energy, external_influences):
    H_internal = np.diag(np.random.normal(internal_energy, 0.1, size))
    H_interaction = np.random.normal(interaction_energy, 0.05, (size, size))
    H_external = np.diag(np.random.normal(external_influences, 0.1, size))
    return H_internal + H_interaction + H_external

def apply_stdp(H, dt, A_plus, A_minus, tau_plus, tau_minus):
    if dt > 0:
        delta_H = A_plus * np.exp(-dt / tau_plus)
    else:
        delta_H = -A_minus * np.exp(dt / tau_minus)
    return H + delta_H

def time_evolution_with_tunneling(Psi_DNA, H_DNA, T_tunnel, dt, steps, A_plus, A_minus, tau_plus, tau_minus):
    values = []
    for t in range(steps):
        H_DNA = apply_stdp(H_DNA, dt, A_plus, A_minus, tau_plus, tau_minus)
        Psi_DNA = (np.eye(len(Psi_DNA)) - 1j * (H_DNA + T_tunnel) * dt) @ Psi_DNA
        Psi_DNA /= np.linalg.norm(Psi_DNA)  # Normalize at each step
        value = Psi_DNA.conj().T @ Psi_DNA
        values.append(np.real(value))
    return values

def add_tunneling_effects(H_DNA, barriers):
    for barrier in barriers:
        H_DNA[barrier, barrier] += 5  # Add high potential barrier
    return H_DNA

def main():
    sequence = "GGCGATACAG"
    bases = initialize_bases()
    Psi_DNA = encode_dna(sequence, bases)
    Psi_DNA = prepare_state(Psi_DNA)

    H_DNA = create_hamiltonian(len(Psi_DNA), 1.0, 0.5, 0.3)
    H_DNA = add_tunneling_effects(H_DNA, [4, 8])
    T_tunnel = np.eye(len(Psi_DNA))

    # Parameters for STDP
    A_plus = 0.005
    A_minus = 0.005
    tau_plus = 20
    tau_minus = 20

    dt = 0.1
    time_steps = 100
    values = time_evolution_with_tunneling(Psi_DNA, H_DNA, T_tunnel, dt, time_steps, A_plus, A_minus, tau_plus, tau_minus)

    # Diagnostic print to check values
    print("Sample values:", values[:10])  # Print first 10 values to inspect them

    # Plot the results directly
    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, time_steps * dt, time_steps), values, lw=2)
    ax.set_xlim(0, time_steps * dt)
    ax.set_ylim(min(values) - 0.05, max(values) + 0.05)  # Adjust y-axis dynamically based on values
    ax.set_xlabel('Time')
    ax.set_ylabel('Consciousness Value')
    ax.set_title('Real-time Consciousness Simulation with Tunneling and STDP')

    plt.show()

if __name__ == "__main__":
    main()
