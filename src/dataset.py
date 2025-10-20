import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from Bio import PDB
import os
import esm
import pickle
from Bio.PDB import DSSP, is_aa
from Bio.PDB.Polypeptide import PPBuilder
import os
import gc
import json
import hashlib
from Bio.PDB.vectors import calc_angle
from Bio.PDB.NeighborSearch import NeighborSearch
import math
from Bio.PDB.vectors import Vector
import warnings
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
warnings.filterwarnings('ignore')

gc.collect()
# Set GPU devices
os.environ["CUDA_VISIBLE_DEVICES"] = "14"

def calculate_phi_psi(chain):
    """Calculate phi/psi angles using PPBuilder"""
    angles = []
    ppb = PPBuilder()
    
    # Get the polypeptides
    for pp in ppb.build_peptides(chain):
        phi_psi_angles = pp.get_phi_psi_list()
        for phi, psi in phi_psi_angles:
            if phi is None:
                phi = 0.0  # Default value for N-terminus
            if psi is None:
                psi = 0.0  # Default value for C-terminus
                
            # Convert to degrees and normalize to [-1, 1]
            phi = float(phi) / 180.0 if not math.isnan(float(phi)) else 0.0
            psi = float(psi) / 180.0 if not math.isnan(float(psi)) else 0.0
            
            angles.append([phi, psi])
    
    return np.array(angles)

def count_atoms_in_pdb(pdb_file):
    """Count the number of atoms in a PDB file."""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
    atom_count = sum(len(residue) for model in structure for chain in model for residue in chain)
    return atom_count

class ProteinDataset(Dataset):
    def __init__(self, csv_file, pdb_folder, embedding_folder, cache_dir='cached_data'):
        self.data = pd.read_csv(csv_file) if not isinstance(csv_file, pd.DataFrame) else csv_file
        self.pdb_folder = pdb_folder
        self.embedding_folder = embedding_folder
        
        print("Initializing ESM model...")
        self.esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = alphabet.get_batch_converter()
        self.esm_model.eval()

        self.aa3_to_aa1 = {aa3: aa1 for aa3, aa1 in zip(PDB.Polypeptide.aa3, PDB.Polypeptide.aa1)}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.esm_model = self.esm_model.to(self.device)
        print(f"ESM model loaded on {self.device}")
        self.pdb_parser = PDB.PDBParser(QUIET=True)

        self.max_seq_length = self._calculate_max_seq_length()
        
        # Add cache directory setup
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.hbond_distance_cutoff = 3.5  # Å
        self.hbond_angle_cutoff = 120  # degrees

        # Secondary structure mapping
        self.ss_mapping = {
            'H': [1, 0, 0, 0, 0, 0, 0, 0],  # α-helix
            'G': [0, 1, 0, 0, 0, 0, 0, 0],  # 3-10 helix
            'I': [0, 0, 1, 0, 0, 0, 0, 0],  # π-helix
            'E': [0, 0, 0, 1, 0, 0, 0, 0],  # β-strand
            'B': [0, 0, 0, 0, 1, 0, 0, 0],  # β-bridge
            'T': [0, 0, 0, 0, 0, 1, 0, 0],  # Turn
            'S': [0, 0, 0, 0, 0, 0, 1, 0],  # Bend
            '-': [0, 0, 0, 0, 0, 0, 0, 1]   # Coil
        }

        # Add physicochemical properties
        self.aa_charge = {
            'R': 1, 'K': 1, 'D': -1, 'E': -1, 'H': 0.5,
            'A': 0, 'N': 0, 'C': 0, 'G': 0, 'I': 0,
            'L': 0, 'M': 0, 'F': 0, 'P': 0, 'S': 0,
            'T': 0, 'W': 0, 'Y': 0, 'V': 0, 'Q': 0
        }
        
        self.aa_hydrophobicity = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
        
        self.atom_types = {
            'C': [1, 0, 0, 0],   # Carbon
            'N': [0, 1, 0, 0],   # Nitrogen
            'O': [0, 0, 1, 0],   # Oxygen
            'S': [0, 0, 0, 1],   # Sulfur
            'OTHER': [0, 0, 0, 0] # Other atoms
        }

        # Find the maximum atom count
        self.max_atom_count = self.find_max_atom_count(pdb_folder)

    def find_max_atom_count(self, pdb_folder):
        max_count = 0
        for pdb_file in os.listdir(pdb_folder):
            if pdb_file.endswith('.pdb'):
                full_path = os.path.join(pdb_folder, pdb_file)
                atom_count = count_atoms_in_pdb(full_path)
                max_count = max(max_count, atom_count)
        return max_count

    def _calculate_max_seq_length(self):
        max_len = 0
        for _, row in self.data.iterrows():
            structure = self._load_structure(row['uniprot'])
            if structure:
                sequence, _, _ = self.get_sequence_and_mapping(structure, row['chain'])
                max_len = max(max_len, len(sequence))
        return max_len

    def _load_structure(self, uniprot_id):
        pdb_file = os.path.join(self.pdb_folder, f"{uniprot_id}.pdb")
        if os.path.exists(pdb_file):
            return self.pdb_parser.get_structure(uniprot_id, pdb_file)
        return None

    def _get_cache_path(self, uniprot_id, chain_id, mutation):
        # Create a unique cache key
        cache_key = f"{uniprot_id}_{chain_id}_{mutation}"
        hash_key = hashlib.md5(cache_key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hash_key}.pkl")

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        uniprot_id = row['uniprot']
        chain_id = row['chain']
        mutation = row['mut']
        ddg = row['ddg']

        # Try to load from cache first
        cache_path = self._get_cache_path(uniprot_id, chain_id, mutation)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                return cached_data
            except Exception as e:
                print(f"Error loading cache: {e}")

        try:
            structure = self._load_structure(uniprot_id)
            if not structure:
                raise ValueError(f"No structure found for {uniprot_id}")

            sequence, residue_map, reverse_map = self.get_sequence_and_mapping(structure, chain_id)
            if not sequence:
                raise ValueError(f"No valid sequence found for {uniprot_id}_{chain_id}")

            # Calculate RSA using DSSP
            dssp = DSSP(structure[0], os.path.join(self.pdb_folder, f"{uniprot_id}.pdb"), dssp='mkdssp')
            rsa_values = []
            for model in structure:
                chain = model[chain_id]
                for residue in chain:
                    if PDB.is_aa(residue):
                        key = (chain_id, residue.id)
                        if key in dssp.keys():
                            rsa = dssp[key][3]  # Relative ASA
                            rsa_values.append(float(rsa))
                        else:
                            rsa_values.append(0.0)

            # Calculate backbone angles
            backbone_angles = self.get_backbone_angles(structure, chain_id)
            if len(backbone_angles) == 0:
                backbone_angles = np.zeros((len(sequence), 2))  # Default values if calculation fails
            
            # Pad RSA and backbone angles to max_seq_length
            padded_rsa = np.pad(
                rsa_values,
                (0, self.max_seq_length - len(rsa_values)),
                'constant',
                constant_values=0
            )
            
            padded_angles = np.pad(
                backbone_angles,
                ((0, self.max_seq_length - len(backbone_angles)), (0, 0)),
                'constant',
                constant_values=0
            )

            wt_aa, pdb_position, mut_aa = mutation[0], int(mutation[1:-1]), mutation[-1]
            
            if pdb_position not in reverse_map:
                print(f"PDB position {pdb_position} not found in reverse residue map.")
                return None

            seq_position = reverse_map[pdb_position]
            
            if sequence[seq_position] != wt_aa:
                return None

            # Get embeddings
            wt_embedding = self.get_embedding(sequence)[seq_position]
            mutated_sequence = sequence[:seq_position] + mut_aa + sequence[seq_position+1:]
            mut_embedding = self.get_embedding(mutated_sequence)[seq_position]

            # Move embeddings to CPU immediately after calculation
            wt_embedding = wt_embedding.cpu()
            mut_embedding = mut_embedding.cpu()

            # Get distance matrices
        #     ca_distance_matrix = self.get_ca_distance_matrix(structure, chain_id)  # CA distance matrix
        #     atom_distance_matrix = self.get_all_atom_distance_matrix(structure, chain_id)  # Atom distance matrix

        #     # Apply 10 Ångström cutoff to distance matrices
        #    # Pad the distance matrices if needed
        #     pad_size_ca = max(0, self.max_seq_length - ca_distance_matrix.shape[0])
        #     pad_size_atom = max(0, self.max_atom_count - atom_distance_matrix.shape[0])

        #     padded_ca_distance_matrix = np.pad(
        #         ca_distance_matrix,
        #         ((0, pad_size_ca), (0, pad_size_ca)),
        #         'constant',
        #         constant_values=0
        #     )

        #     padded_atom_distance_matrix = np.pad(
        #         atom_distance_matrix,
        #         ((0, pad_size_atom), (0, pad_size_atom)),
        #         'constant',
        #         constant_values=0
        #     )

        # Define the fixed size for the sub-matrices
            # fixed_size = 15  # Example fixed size, adjust as needed

            # # Function to extract sub-matrix around a position with a given cutoff
            # def extract_sub_matrix(matrix, position, cutoff, fixed_size):
            #     start = max(0, position - cutoff)
            #     end = min(matrix.shape[0], position + cutoff + 1)
            #     sub_matrix = matrix[start:end, start:end]
                
            #     # Pad or truncate the sub-matrix to the fixed size
            #     if sub_matrix.shape[0] < fixed_size:
            #         pad_size = fixed_size - sub_matrix.shape[0]
            #         sub_matrix = np.pad(sub_matrix, ((0, pad_size), (0, pad_size)), 'constant', constant_values=0)
            #     elif sub_matrix.shape[0] > fixed_size:
            #         sub_matrix = sub_matrix[:fixed_size, :fixed_size]
                
            #     return sub_matrix

            # # Get distance matrices
            # ca_distance_matrix = self.get_ca_distance_matrix(structure, chain_id)  # CA distance matrix
            # atom_distance_matrix = self.get_all_atom_distance_matrix(structure, chain_id)  # Atom distance matrix

            # # Extract sub-matrices around the sequence position with 15 Ångström cutoff
            # padded_ca_distance_matrix = extract_sub_matrix(ca_distance_matrix, seq_position, 15, fixed_size)
            # padded_atom_distance_matrix = extract_sub_matrix(atom_distance_matrix, seq_position, 15, fixed_size)

            # Define the fixed size for the sub-matrices
            fixed_size = 15  # Example fixed size, adjust as needed

            # Function to extract sub-matrix around a position with a given cutoff
            def extract_sub_matrix(matrix, position, cutoff, fixed_size):
                start = max(0, position - cutoff)
                end = min(matrix.shape[0], position + cutoff + 1)
                sub_matrix = matrix[start:end, start:end]
                
                # Pad or truncate the sub-matrix to the fixed size
                if sub_matrix.shape[0] < fixed_size:
                    pad_size = fixed_size - sub_matrix.shape[0]
                    sub_matrix = np.pad(sub_matrix, ((0, pad_size), (0, pad_size)), 'constant', constant_values=0)
                elif sub_matrix.shape[0] > fixed_size:
                    sub_matrix = sub_matrix[:fixed_size, :fixed_size]
                
                return sub_matrix

            # Function to extract atom indices within a cutoff from the CA distance matrix
            def get_atom_indices_within_cutoff(ca_distance_matrix, position, cutoff):
                indices = np.where(ca_distance_matrix[position] <= cutoff)[0]
                return indices

            # Get distance matrices
            ca_distance_matrix = self.get_ca_distance_matrix(structure, chain_id)  # CA distance matrix
            atom_distance_matrix = self.get_all_atom_distance_matrix(structure, chain_id)  # Atom distance matrix

            # Extract sub-matrix around the sequence position with 15 Ångström cutoff for CA distance matrix
            padded_ca_distance_matrix = extract_sub_matrix(ca_distance_matrix, seq_position, 15, 15)

            # Get atom indices within 15 Ångström cutoff from the CA distance matrix
            atom_indices = get_atom_indices_within_cutoff(ca_distance_matrix, seq_position, 15)

            # Extract sub-matrix for atom distance matrix using the identified atom indices
            atom_sub_matrix = atom_distance_matrix[np.ix_(atom_indices, atom_indices)]
            # padded_atom_distance_matrix = atom_sub_matrix
            # Pad or truncate the atom sub-matrix to the fixed size
            if atom_sub_matrix.shape[0] < fixed_size:
                pad_size = fixed_size - atom_sub_matrix.shape[0]
                padded_atom_distance_matrix = np.pad(atom_sub_matrix, ((0, pad_size), (0, pad_size)), 'constant', constant_values=0)
            elif atom_sub_matrix.shape[0] > fixed_size:
                padded_atom_distance_matrix = atom_sub_matrix[:fixed_size, :fixed_size]
            else:
                padded_atom_distance_matrix = atom_sub_matrix

            # Calculate hydrogen bond features
            hbond_features = self.calculate_hbonds(structure, chain_id)
            if len(hbond_features) == 0:
                hbond_features = np.zeros((len(sequence), 3))  # Default values if calculation fails
            
            # Pad hydrogen bond features
            padded_hbonds = np.pad(
                hbond_features,
                ((0, self.max_seq_length - len(hbond_features)), (0, 0)),
                'constant',
                constant_values=0
            )

            # Calculate secondary structure features
            ss_features = self.calculate_secondary_structure(structure, chain_id)
            if len(ss_features) == 0:
                ss_features = np.zeros((len(sequence), 8))  # Default values if calculation fails
            
            # Pad secondary structure features
            padded_ss = np.pad(
                ss_features,
                ((0, self.max_seq_length - len(ss_features)), (0, 0)),
                'constant',
                constant_values=0
            )

            # Calculate physicochemical features
            charge_features, hydrophobicity_features = self.get_physicochemical_features(sequence)
            atom_features = self.get_atom_type_features(structure, chain_id)

            # Pad physicochemical features
            padded_charge = np.pad(
                charge_features,
                ((0, self.max_seq_length - len(charge_features)), (0, 0)),
                'constant',
                constant_values=0
            )
            
            padded_hydrophobicity = np.pad(
                hydrophobicity_features,
                ((0, self.max_seq_length - len(hydrophobicity_features)), (0, 0)),
                'constant',
                constant_values=0
            )
            
            padded_atoms = np.pad(
                atom_features,
                ((0, self.max_seq_length - len(atom_features)), (0, 0)),
                'constant',
                constant_values=0
            )

            result = {
                'wt_embedding': wt_embedding,
                'mut_embedding': mut_embedding,
                'ca_distance_matrix': torch.tensor(padded_ca_distance_matrix, dtype=torch.float32),
                'atom_distance_matrix': torch.tensor(padded_atom_distance_matrix, dtype=torch.float32),
                'rsa_values': torch.tensor(padded_rsa, dtype=torch.float32),
                'backbone_angles': torch.tensor(padded_angles, dtype=torch.float32),
                'hbond_features': torch.tensor(padded_hbonds, dtype=torch.float32),
                'ss_features': torch.tensor(padded_ss, dtype=torch.float32),
                'charge_features': torch.tensor(padded_charge, dtype=torch.float32),
                'hydrophobicity_features': torch.tensor(padded_hydrophobicity, dtype=torch.float32),
                'atom_features': torch.tensor(padded_atoms, dtype=torch.float32),
                'ddg': torch.tensor(ddg, dtype=torch.float32),
            }

            # Save to cache
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(result, f)
            except Exception as e:
                print(f"Error saving to cache: {e}")

            return result

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def __len__(self):
        return len(self.data)

    def get_sequence_and_mapping(self, structure, chain_id):
        seq = ""
        residue_map = {}
        reverse_map = {}
        seq_index = 0

        for model in structure:
            if chain_id in model:
                chain = model[chain_id]
                for residue in chain:
                    if PDB.is_aa(residue):
                        residue_id = residue.id[1]
                        try:
                            one_letter = self.aa3_to_aa1.get(residue.resname, 'X')
                            seq += one_letter
                            residue_map[seq_index] = residue_id
                            reverse_map[residue_id] = seq_index
                            seq_index += 1
                        except Exception as e:
                            print(f"Error processing residue {residue.resname} at position {residue_id}: {str(e)}")
                return seq, residue_map, reverse_map
        return "", {}, {}

    def get_ca_distance_matrix(self, structure, chain_id):
        """Calculate the distance matrix for CA atoms in the structure."""
        try:
            for model in structure:
                if chain_id in model:
                    chain = model[chain_id]
                    
                    # Get coordinates of all CA atoms
                    ca_atoms = [res['CA'] for res in chain if PDB.is_aa(res) and 'CA' in res]
                    if not ca_atoms:
                        print(f"Warning: No CA atoms found in chain {chain_id}")
                        return np.array([])

                    # Get coordinates of CA atoms
                    coords = np.array([atom.coord for atom in ca_atoms])
                    
                    # Calculate the distance matrix for CA atoms
                    ca_dist_matrix = np.linalg.norm(coords[:, None] - coords, axis=2)
                    return ca_dist_matrix

            print(f"Warning: Chain {chain_id} not found in structure")
            return np.array([])

        except Exception as e:
            print(f"Error creating CA distance matrix: {str(e)}")
            return np.array([])

    def get_all_atom_distance_matrix(self, structure, chain_id):
        """Calculate the distance matrix for all atoms in the structure."""
        try:
            for model in structure:
                if chain_id in model:
                    chain = model[chain_id]
                    
                    # Get coordinates of all atoms
                    all_atoms = [atom for residue in chain for atom in residue]
                    if not all_atoms:
                        print(f"Warning: No atoms found in chain {chain_id}")
                        return np.array([])

                    # Get coordinates of all atoms
                    coords = np.array([atom.coord for atom in all_atoms])
                    
                    # Calculate the distance matrix for all atoms
                    all_atom_dist_matrix = np.linalg.norm(coords[:, None] - coords, axis=2)
                    return all_atom_dist_matrix

            print(f"Warning: Chain {chain_id} not found in structure")
            return np.array([])

        except Exception as e:
            print(f"Error creating all atom distance matrix: {str(e)}")
            return np.array([])

    def get_embedding(self, sequence):
        try:
            # Create a cache key for embeddings
            cache_key = hashlib.md5(sequence.encode()).hexdigest()
            embedding_cache_path = os.path.join(self.cache_dir, f"embedding_{cache_key}.pt")

            # Try to load from cache with weights_only=True
            if os.path.exists(embedding_cache_path):
                embeddings = torch.load(embedding_cache_path, weights_only=True)
                return embeddings

            # Calculate if not in cache
            with torch.no_grad():
                _, _, batch_tokens = self.batch_converter([(None, sequence)])
                batch_tokens = batch_tokens.to(self.device)
                results = self.esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
                embeddings = results["representations"][33].squeeze(0)
                
                # Save to cache
                torch.save(embeddings.cpu(), embedding_cache_path)
                
                # Clear memory
                del results, batch_tokens
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return embeddings

        except Exception as e:
            print(f"Error in get_embedding: {str(e)}")
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def get_backbone_angles(self, structure, chain_id):
        """Calculate backbone torsion angles (phi, psi) for each residue"""
        angles = []
        
        for model in structure:
            if chain_id in model:
                chain = model[chain_id]
                angles = calculate_phi_psi(chain)
                if len(angles) > 0:
                    return angles
        
        return np.array([])


    def calculate_hbonds(self, structure, chain_id):
        """Calculate hydrogen bond features for each residue using stricter criteria"""
        try:
            chain = structure[0][chain_id]
            residues = list(chain)
            n_residues = len(residues)
            hbond_features = []

            # print(f"\nCalculating H-bonds for chain {chain_id}")
            # print(f"Number of residues: {n_residues}")

            def calculate_distance(coord1, coord2):
                return np.sqrt(np.sum((coord1 - coord2) ** 2))

            def calculate_angle(coord1, coord2, coord3):
                """Calculate angle between three points in degrees"""
                v1 = coord1 - coord2
                v2 = coord3 - coord2
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                return np.degrees(np.arccos(cos_angle))

            # Iterate through each residue
            for i, residue in enumerate(residues):
                if not PDB.is_aa(residue):
                    continue

                backbone_hbonds = 0
                sidechain_hbonds = 0

                # Get coordinates of backbone atoms
                try:
                    n_coord = np.array(residue['N'].get_coord())
                    o_coord = np.array(residue['O'].get_coord())
                    h_coord = None
                    if 'H' in residue:
                        h_coord = np.array(residue['H'].get_coord())
                    
                    # print(f"\nResidue {residue.get_resname()}{residue.get_id()[1]}")
                    # print(f"Has H atom: {'Yes' if h_coord is not None else 'No'}")
                    
                except KeyError as e:
                    # print(f"Missing backbone atom in residue {residue.get_resname()}{residue.get_id()[1]}: {e}")
                    continue

                # Check for backbone hydrogen bonds
                for other_res in residues:
                    if not PDB.is_aa(other_res) or other_res == residue:
                        continue

                    try:
                        # Check N-H...O hydrogen bonds
                        if h_coord is not None and 'O' in other_res:
                            other_o_coord = np.array(other_res['O'].get_coord())
                            h_o_dist = calculate_distance(h_coord, other_o_coord)
                            n_h_o_angle = calculate_angle(n_coord, h_coord, other_o_coord)

                            if h_o_dist <= 3.5 and n_h_o_angle >= 120:  # Relaxed criteria for testing
                                backbone_hbonds += 1
                                # print(f"Found backbone H-bond with residue {other_res.get_resname()}{other_res.get_id()[1]}")
                                # print(f"Distance: {h_o_dist:.2f}Å, Angle: {n_h_o_angle:.2f}°")

                    except KeyError as e:
                        continue

                # Check for sidechain hydrogen bonds
                sidechain_donors = {
                    'ARG': ['NE', 'NH1', 'NH2'],
                    'ASN': ['ND2'],
                    'GLN': ['NE2'],
                    'HIS': ['ND1', 'NE2'],
                    'LYS': ['NZ'],
                    'SER': ['OG'],
                    'THR': ['OG1'],
                    'TRP': ['NE1'],
                    'TYR': ['OH']
                }

                sidechain_acceptors = {
                    'ASP': ['OD1', 'OD2'],
                    'GLU': ['OE1', 'OE2'],
                    'ASN': ['OD1'],
                    'GLN': ['OE1'],
                    'HIS': ['ND1', 'NE2'],
                    'SER': ['OG'],
                    'THR': ['OG1'],
                    'TYR': ['OH']
                }

                res_name = residue.get_resname()
                
                # Check sidechain donors
                if res_name in sidechain_donors:
                    for donor_atom in sidechain_donors[res_name]:
                        if donor_atom not in residue:
                            continue
                        donor_coord = np.array(residue[donor_atom].get_coord())

                        for other_res in residues:
                            if not PDB.is_aa(other_res) or other_res == residue:
                                continue

                            other_name = other_res.get_resname()
                            if other_name in sidechain_acceptors:
                                for acceptor_atom in sidechain_acceptors[other_name]:
                                    if acceptor_atom not in other_res:
                                        continue
                                    acceptor_coord = np.array(other_res[acceptor_atom].get_coord())
                                    dist = calculate_distance(donor_coord, acceptor_coord)

                                    if dist <= 3.5:  # Distance criterion for sidechain H-bonds
                                        sidechain_hbonds += 1
                                        # print(f"Found sidechain H-bond: {res_name}{donor_atom} -> {other_name}{acceptor_atom}")
                                        # print(f"Distance: {dist:.2f}Å")

                # Normalize and store features
                total_hbonds = backbone_hbonds + sidechain_hbonds
                hbond_features.append([
                    min(backbone_hbonds / 4, 1.0),    # Normalize backbone H-bonds
                    min(sidechain_hbonds / 6, 1.0),   # Normalize sidechain H-bonds
                    min(total_hbonds / 10, 1.0)       # Normalize total H-bonds
                ])
                
                # print(f"Total H-bonds for residue: Backbone={backbone_hbonds}, Sidechain={sidechain_hbonds}")

            return np.array(hbond_features)

        except Exception as e:
            print(f"Error calculating H-bonds: {str(e)}")
            import traceback
            traceback.print_exc()
            return np.array([])

    def calculate_secondary_structure(self, structure, chain_id):
        """Calculate secondary structure features for each residue"""
        try:
            # Get the PDB file path
            uniprot_id = structure.id
            pdb_file = os.path.join(self.pdb_folder, f"{uniprot_id}.pdb")
            
            # Get DSSP calculator
            dssp = PDB.DSSP(structure[0], pdb_file, dssp='mkdssp')
            
            # Initialize features list
            ss_features = []
            
            # Get chain residues
            chain = structure[0][chain_id]
            residues = list(chain)
            
            # Debug information
            # print(f"Processing structure: {uniprot_id}")
            # print(f"Number of residues: {len(residues)}")
            # print(f"DSSP keys available: {len(dssp.keys())}")
            
            for residue in residues:
                if not PDB.is_aa(residue):
                    continue
                    
                # Get residue key for DSSP
                key = (chain_id, residue.get_id())
                
                # Get secondary structure from DSSP
                if key in dssp:
                    ss_type = dssp[key][2]
                    # print(f"Residue {residue.get_id()[1]}: {ss_type}")  # Debug print
                    # Convert DSSP code to one-hot encoding using mapping
                    ss_vector = self.ss_mapping.get(ss_type, self.ss_mapping['-'])
                else:
                    # If residue not in DSSP, assume coil
                    # print(f"Residue {residue.get_id()[1]} not found in DSSP")  # Debug print
                    ss_vector = self.ss_mapping['-']
                
                ss_features.append(ss_vector)
            
            ss_array = np.array(ss_features)
            # print(f"Shape of ss_features: {ss_array.shape}")  # Debug print
            # print(f"Number of helices: {np.sum(ss_array[:, 0])}")  # Count of α-helices
            # print(f"Number of strands: {np.sum(ss_array[:, 3])}")  # Count of β-strands
            
            return ss_array
            
        except Exception as e:
            print(f"Error calculating secondary structure: {str(e)}")
            import traceback
            traceback.print_exc()  # Print full error traceback
            return np.array([])

    def get_physicochemical_features(self, sequence):
        """Calculate charge and hydrophobicity features"""
        charge_features = []
        hydrophobicity_features = []
        
        for aa in sequence:
            charge_features.append([self.aa_charge.get(aa, 0)])
            hydrophobicity_features.append([self.aa_hydrophobicity.get(aa, 0)])
            
        return np.array(charge_features), np.array(hydrophobicity_features)

    def get_atom_type_features(self, structure, chain_id):
        """Calculate atom type features for each residue"""
        atom_features = []
        
        for model in structure:
            if chain_id in model:
                chain = model[chain_id]
                for residue in chain:
                    if PDB.is_aa(residue):
                        residue_atoms = []
                        for atom in residue:
                            atom_type = self.atom_types.get(atom.element, self.atom_types['OTHER'])
                            residue_atoms.append(atom_type)
                        # Average the atom features for the residue
                        if residue_atoms:
                            avg_atoms = np.mean(residue_atoms, axis=0)
                        else:
                            avg_atoms = np.zeros(4)
                        atom_features.append(avg_atoms)
        
        return np.array(atom_features)
