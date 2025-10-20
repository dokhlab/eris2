import argparse
import torch
from src.model import DDGPredictor
from src.dataset import ProteinDataset

def predict_ddg(pdb_file, chain_id, mutation, model_path, device="cuda"):
    """Predict ΔΔG for a single mutation."""
    # Load model
    model = DDGPredictor(embedding_dim=1280)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    # Create dataset for single prediction
    dataset = ProteinDataset(
        csv_file=None,  # We'll create a single-item dataset
        pdb_folder=pdb_file,
        embedding_folder=None,
        cache_dir="cache"
    )
    
    # Make prediction
    with torch.no_grad():
        prediction = model(dataset[0])
    
    return prediction.item()

def main():
    parser = argparse.ArgumentParser(description="Predict protein stability change (ΔΔG) for a mutation")
    parser.add_argument("--pdb", required=True, help="Path to PDB file")
    parser.add_argument("--chain", required=True, help="Chain ID")
    parser.add_argument("--mutation", required=True, help="Mutation in format WT+Position+Mut (e.g., A10G)")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
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