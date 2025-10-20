import argparse
import torch
import pandas as pd
import os
from model_quantum import DDGPredictor
from dataset import ProteinDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def collate_fn(batch):
    """Custom collate function for batch prediction."""
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
            'hydrophobicity_features', 'atom_features'
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

def predict_multiple_mutations(csv_file, pdb_folder, model_path, output_file, batch_size=32, device="cuda"):
    """Predict ΔΔG for multiple mutations from a CSV file."""
    # Load model
    model = DDGPredictor(embedding_dim=1280)
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.to(device)
    model.eval()
    
    # Read CSV file and add dummy ddg column if it doesn't exist
    df = pd.read_csv(csv_file)
    if 'ddg' not in df.columns:
        df['ddg'] = 0.0  # Add dummy ddg column with zeros
    
    # Create dataset
    dataset = ProteinDataset(
        csv_file=df,  # Pass DataFrame directly
        pdb_folder=pdb_folder,
        embedding_folder=None,
        cache_dir="cache"
    )
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Prepare output DataFrame
    results = []
    
    # Make predictions
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Making predictions")):
            if batch is None:
                continue
                
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Get predictions
            predictions = model(
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
            
            # Get corresponding rows from original DataFrame
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            # Store results
            for i, pred in enumerate(predictions):
                if i < len(batch_df):
                    results.append({
                        'uniprot': batch_df.iloc[i]['uniprot'],
                        'chain': batch_df.iloc[i]['chain'],
                        'mutation': batch_df.iloc[i]['mut'],
                        'predicted_ddg': pred.item()
                    })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description="Predict protein stability changes (ΔΔG) for multiple mutations")
    parser.add_argument("--csv", required=True, help="Path to CSV file with mutations")
    parser.add_argument("--pdb_folder", required=True, help="Path to folder containing PDB files")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for prediction")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Check if CUDA is available when cuda is requested
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"
    
    # Make predictions
    results = predict_multiple_mutations(
        csv_file=args.csv,
        pdb_folder=args.pdb_folder,
        model_path=args.model,
        output_file=args.output,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Print summary statistics
    print("\nPrediction Summary:")
    print(f"Total mutations processed: {len(results)}")
    print(f"Average predicted ΔΔG: {results['predicted_ddg'].mean():.2f} kcal/mol")
    print(f"Min predicted ΔΔG: {results['predicted_ddg'].min():.2f} kcal/mol")
    print(f"Max predicted ΔΔG: {results['predicted_ddg'].max():.2f} kcal/mol")

if __name__ == "__main__":
    main() 