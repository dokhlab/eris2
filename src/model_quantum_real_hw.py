# Update : The scaling factor 20 is in multiplied with quatum output, earlier it was at the quantum input.
# The scaling factor is used to adjust the quantum output to match the expected range of the classical model.
# The final clasical leayer is updated to have more layers. 
# To load the previous trained model use v0 version. 
# Using all the features in the model.
# Reduced the no of EnhanscedResidualBlock to 1 for each feature.
import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import torch.nn.functional as F

from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(channel="ibm_quantum", token="<insert your IBM account API here>", overwrite=True)

num_qubits = 12
n_shots = int(2e2)
# backend = GenericBackendV2(num_qubits=num_qubits, seed=42)
# qk_noise_model = NoiseModel.from_backend(backend)
# dev_qk_noisy = qml.device("qiskit.aer", wires=num_qubits, shots=n_shots, noise_model=qk_noise_model)


class Quantum_FFN(nn.Module):
    def __init__(self, input_dim, num_qubits):
        super(Quantum_FFN, self).__init__()
        self.input_dim = input_dim
        self.num_qubits = num_qubits
        self.q_input_dim = 2 ** num_qubits

        # Map classical input to quantum state dimension
        self.classical_to_quantum = nn.Linear(input_dim, self.q_input_dim)
        # Noisy simulator
        # n_shots = int(2e2)
        # backend = GenericBackendV2(num_qubits=num_qubits, seed=42)
        # qk_noise_model = NoiseModel.from_backend(backend)
        # self.q_device = qml.device("qiskit.aer", wires=num_qubits, shots=n_shots, noise_model=qk_noise_model)
        # Real hardware
        service = QiskitRuntimeService(channel="ibm_quantum")
        backend = service.least_busy(min_num_qubits=127, operational=True, simulator=False )
        self.q_device = qml.device('qiskit.remote', wires=num_qubits, backend=backend)
        # Define PennyLane device
        #self.q_device = qml.device('default.qubit', wires=num_qubits)

        @qml.qnode(self.q_device, interface='torch')
        def circuit(inputs):
            qml.AmplitudeEmbedding(inputs, wires=range(num_qubits), normalize=True)
            #qml.Barrier(wires=range(num_qubits))
            for i in range(num_qubits):
                qml.RY(np.pi / 4, wires=i)
                qml.CNOT(wires=[i, (i + 1) % num_qubits])
            return qml.probs(wires=range(num_qubits))

        self.q_circuit = circuit

    # def forward(self, x):
    #     # Map to quantum input dimension
    #     x = self.classical_to_quantum(x)
    #     batch_size = x.shape[0]

    #     # Normalize for amplitude embedding
    #     x = x / torch.norm(x, dim=1, keepdim=True)

    #     # Apply quantum circuit batch-wise
    #     probs = torch.stack([self.q_circuit(x[i]) for i in range(batch_size)]).float()

    #     return probs

    def forward(self, x):
        # Map to quantum input dimension
        x = self.classical_to_quantum(x)
        device = x.device  # Get the device of input tensor

        batch_size = x.shape[0]
        x = x / torch.norm(x, dim=1, keepdim=True)

        # Run quantum circuit (PennyLane only supports CPU), convert to same device
        probs = torch.stack([self.q_circuit(x[i].cpu()) for i in range(batch_size)]).float()

        return probs.to(device)  # Move result to original device


class EnhancedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(EnhancedResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        return F.relu(out)    
class DDGPredictor(nn.Module):
    def __init__(self, embedding_dim=1280, ca_matrix_size=15, num_qubits=8,
                 use_batchnorm=True, dropout_rate=0.2):
        super(DDGPredictor, self).__init__()
        self.embedding_dim = embedding_dim
        self.ca_matrix_size = ca_matrix_size
        self.num_qubits = num_qubits
        self.q_output_dim = 2 ** num_qubits
        # Feature processors
        self.rsa_processor = nn.Sequential(
            EnhancedResidualBlock(1, 64),

            nn.AdaptiveAvgPool1d(1)
        )
        
        self.angles_processor = nn.Sequential(
            EnhancedResidualBlock(2, 64),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.hbond_processor = nn.Sequential(
            EnhancedResidualBlock(3, 64),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.ss_processor = nn.Sequential(
            EnhancedResidualBlock(8, 64),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.charge_processor = nn.Sequential(
            EnhancedResidualBlock(1, 64),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.hydrophobicity_processor = nn.Sequential(
            EnhancedResidualBlock(1, 64),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.atom_processor = nn.Sequential(
            EnhancedResidualBlock(4, 64),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Feature normalization
        self.feature_norm = nn.LayerNorm(64)        
        input_dim = 2 * embedding_dim + 2*(ca_matrix_size * ca_matrix_size) + 7*64
        self.qffn = Quantum_FFN(input_dim=input_dim, num_qubits=num_qubits)

        # Classical head after quantum circuit
        layers = []
        layers.append(nn.Linear(self.q_output_dim, 1024))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(1024))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(1024, 256))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(256))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(256, 64))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(64))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(64, 1))

        

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, wt_embedding, mut_embedding, ca_distance_matrix, atom_distance_matrix,
                rsa_values, backbone_angles, hbond_features, ss_features,
                charge_features, hydrophobicity_features, atom_features):
        ca_flat = ca_distance_matrix.view(ca_distance_matrix.size(0), -1)
        atom_flat = atom_distance_matrix.view(atom_distance_matrix.size(0), -1)

        # Process other features
        rsa = rsa_values.unsqueeze(1)
        rsa_features = self.feature_norm(self.rsa_processor(rsa).squeeze(-1))
        
        angles = backbone_angles.transpose(1, 2)
        angles_features = self.feature_norm(self.angles_processor(angles).squeeze(-1))
        
        hbonds = hbond_features.transpose(1, 2)
        hbond_features = self.feature_norm(self.hbond_processor(hbonds).squeeze(-1))
        
        ss = ss_features.transpose(1, 2)
        ss_features = self.feature_norm(self.ss_processor(ss).squeeze(-1))
        
        charge = charge_features.transpose(1, 2)
        charge_features = self.feature_norm(self.charge_processor(charge).squeeze(-1))
        
        hydrophobicity = hydrophobicity_features.transpose(1, 2)
        hydrophobicity_features = self.feature_norm(self.hydrophobicity_processor(hydrophobicity).squeeze(-1))
        
        atoms = atom_features.transpose(1, 2)
        atom_features = self.feature_norm(self.atom_processor(atoms).squeeze(-1))

        x = torch.cat([wt_embedding, mut_embedding, ca_flat, atom_flat, rsa_features, angles_features, hbond_features, ss_features,charge_features, hydrophobicity_features, atom_features], dim=1)
        
        q_features = self.qffn(x)
        print(f"q_features : {q_features}")
        
        print(f"q_features * 20 : {20*q_features}")
        out = self.fc_layers(20*q_features)
        print(f"output : {out}")
        return out.squeeze(1)



