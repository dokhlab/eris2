import argparse
import torch
import pandas as pd
import os
import shutil
from model import DDGPredictor
from dataset import ProteinDataset
from torch.utils.data import DataLoader

def collate_fn(batch):
    """Custom collate function for single-item prediction."""
    try:
        # Filter out None values
        batch = [item for item in batch if item is not None and all(v is not None for v in item.values())]
        if len(batch) == 0:
            print("Warning: No valid items in batch.")
            return None

        # Check for required features
        required_features = [
            'wt_embedding', 'mut_embedding', 'atom_distance_matrix',
            'ca_distance_matrix', 'rsa_values', 'backbone_angles',
            'hbond_features', 'ss_features', 'charge_features',
            'hydrophobicity_features', 'atom_features', 'ddg'
        ]
        
        for item in batch:
            missing_features = [feat for feat in required_features if feat not in item]
            if missing_features:
                print(f"Warning: Missing features in batch item: {missing_features}")
                return None

        collated = torch.utils.data.dataloader.default_collate(batch)
        
        # Move tensors to CPU to save GPU memory
        if isinstance(collated, dict):
            for k, v in collated.items():
                if isinstance(v, torch.Tensor):
                    collated[k] = v.cpu()
        
        return collated
    except Exception as e:
        print(f"Collate error: {e}")
        return None

def predict_ddg(pdb_file, chain_id, mutation, model_path, device="cuda"):
    """Predict ΔΔG for a single mutation."""
    # Load model
    model = DDGPredictor(embedding_dim=1280)
    # Add map_location to handle CPU loading of CUDA models
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.to(device)
    model.eval()
    
    # Create temporary directory structure
    temp_dir = "temp_pdb_dir"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Copy PDB file to temp directory
    pdb_filename = os.path.basename(pdb_file)
    temp_pdb_path = os.path.join(temp_dir, pdb_filename)
    shutil.copy2(pdb_file, temp_pdb_path)
    
    try:
        # Get PDB ID from filename (remove .pdb extension)
        pdb_id = os.path.splitext(pdb_filename)[0]
        
        # Create a minimal DataFrame for single prediction
        data_dict = {
            'uniprot': [pdb_id],
            'chain': [chain_id],
            'mut': [mutation],
            'ddg': [0.0]  # Dummy value, won't be used
        }
        single_data = pd.DataFrame(data_dict)
        
        # Create dataset for single prediction
        dataset = ProteinDataset(
            csv_file=single_data,
            pdb_folder=temp_dir,
            embedding_folder=None,
            cache_dir="cache"
        )
        
        # Create data loader with custom collate function
        data_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True
        )
        
        # Get batch from data loader
        batch = next(iter(data_loader))
        if batch is None:
            raise ValueError("Failed to load data batch")
        
        # Move all tensors to the correct device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Make prediction
        with torch.no_grad():
            prediction = model(
                batch['wt_embedding'],
                batch['mut_embedding'],
                batch['ca_distance_matrix'],
                batch['atom_distance_matrix'],
                batch['rsa_values'],
                batch['backbone_angles'],
                batch['hbond_features'],
                batch['ss_features'],
                batch['charge_features'],
                batch['hydrophobicity_features'],
                batch['atom_features']
            )
        
        return prediction.item()
    
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)

def main():
    parser = argparse.ArgumentParser(description="Predict protein stability change (ΔΔG) for a mutation")
    parser.add_argument("--pdb", required=True, help="Path to PDB file")
    parser.add_argument("--chain", required=True, help="Chain ID")
    parser.add_argument("--mutation", required=True, help="Mutation in format WT+Position+Mut (e.g., A10G)")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Check if CUDA is available when cuda is requested
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"
    
    ddg = predict_ddg(
        pdb_file=args.pdb,
        chain_id=args.chain,
        mutation=args.mutation,
        model_path=args.model,
        device=args.device
    )
    
    print(f"Predicted ΔΔG: {ddg:.2f} kcal/mol")

if __name__ == "__main__":
    main() 
