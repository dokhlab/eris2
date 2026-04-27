import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from dataset_checking import ProteinDataset
from model import DDGPredictor
from mut_seq import train_model, evaluate_model
import pandas as pd 
import os
import gc

gc.collect()
# Set GPU devices
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6,7,8,9,10"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def validate_first_batch(loader):
    """Validate the first batch to ensure all features are present"""
    first_batch = next(iter(loader))
    if first_batch is None:
        raise ValueError("First batch is None")
    
    required_features = [
        'wt_embedding', 
        'mut_embedding', 
        'atom_distance_matrix',  # Updated
        'ca_distance_matrix',     # Updated
        'rsa_values',
        'backbone_angles',
        'hbond_features',
        'ss_features',
        'charge_features',          
        'hydrophobicity_features',  
        'atom_features',           
        'ddg'
    ]
    
    for feature in required_features:
        if feature not in first_batch:
            raise ValueError(f"Missing required feature: {feature}")
        
        # Shape validation for specific features
        if feature == 'backbone_angles':
            shape = first_batch[feature].shape
            if len(shape) != 3 or shape[2] != 2:  # [batch_size, seq_len, 2]
                raise ValueError(f"Incorrect backbone angles shape: {shape}")
        elif feature == 'hbond_features':
            shape = first_batch[feature].shape
            if len(shape) != 3 or shape[2] != 3:  # [batch_size, seq_len, 3]
                raise ValueError(f"Incorrect hydrogen bond features shape: {shape}")
        elif feature == 'ss_features':
            shape = first_batch[feature].shape
            if len(shape) != 3 or shape[2] != 8:  # [batch_size, seq_len, 8]
                raise ValueError(f"Incorrect secondary structure features shape: {shape}")
        elif feature == 'charge_features':
            shape = first_batch[feature].shape
            if len(shape) != 3 or shape[2] != 1:  # [batch_size, seq_len, 1]
                raise ValueError(f"Incorrect charge features shape: {shape}")
        elif feature == 'hydrophobicity_features':
            shape = first_batch[feature].shape
            if len(shape) != 3 or shape[2] != 1:  # [batch_size, seq_len, 1]
                raise ValueError(f"Incorrect hydrophobicity features shape: {shape}")
        elif feature == 'atom_features':
            shape = first_batch[feature].shape
            if len(shape) != 3 or shape[2] != 4:  # [batch_size, seq_len, 4]
                raise ValueError(f"Incorrect atom features shape: {shape}")
    
    print("All required features present in the dataset")
    return True
from sklearn.model_selection import KFold  # Add this import

def main():
    # Hyperparameters
    batch_sizes = [256]     # Updated
    learning_rates = [1e-4]
    num_epochs = 15
    cache_dir = '/storage/home/sje5459/scratch/Jan25_PMSCNN_thermompnn/cached_data'
    n_splits = 5  # Number of folds for cross-validation

    # Load data
    csv_file = 'mega_full.csv'
    pdb_folder = '/storage/home/sje5459/scratch/Jan25_PMSCNN_thermompnn/PCNN_dir_rev'
    embedding_folder = 'embeddings'

    # Initialize dataset with cache directory
    dataset = ProteinDataset(
        csv_file=csv_file,
        pdb_folder=pdb_folder,
        embedding_folder=embedding_folder,
        cache_dir=cache_dir
    )

    # Filter out invalid items and get cache statistics
    valid_indices = []
    cached_count = 0
    print("Checking dataset validity and cache status...")

    for i in range(len(dataset)):
        # Check cache status
        row = dataset.data.iloc[i]
        cache_path = dataset._get_cache_path(row['uniprot'], row['chain'], row['mut'])
        if os.path.exists(cache_path):
            cached_count += 1

        # Check validity
        item = dataset[i]
        if item is not None and all(v is not None for v in item.values()):
            # Additional checks for feature shapes
            if (all(feat in item for feat in ['backbone_angles', 'hbond_features', 'ss_features',
                                             'charge_features', 'hydrophobicity_features', 'atom_features',
                                             'atom_distance_matrix', 'ca_distance_matrix']) and
                item['backbone_angles'].shape[-1] == 2 and
                item['hbond_features'].shape[-1] == 3 and
                item['ss_features'].shape[-1] == 8 and
                item['charge_features'].shape[-1] == 1 and
                item['hydrophobicity_features'].shape[-1] == 1 and
                item['atom_features'].shape[-1] == 4):
                valid_indices.append(i)

        # Clear memory periodically
        if i % 100 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print(f"Processed {i}/{len(dataset)} items...")

    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(dataset)}")
    print(f"Valid samples: {len(valid_indices)}")
    print(f"Cached samples: {cached_count}")
    # Modified collate function
    def collate_fn(batch):
        try:
            # Filter out None values
            batch = [item for item in batch if item is not None and all(v is not None for v in item.values())]
            if len(batch) == 0:
                print("Warning: No valid items in batch.")
                return None  # Return None if no valid items

            # Check for required features and their shapes
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
                
                # Validate feature shapes
                # for feature in required_features:
                #     if feature in item:
                #         # print(f"Feature '{feature}' shape: {item[feature].shape}")

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
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    # Initialize KFold
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Define the grid search parameters
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            print(f"Training with batch size: {batch_size}, learning rate: {learning_rate}")

            # Store results for each fold
            fold_results = {
                'test_loss': [],
                'test_rmse': [],
                'mse': [],
                'r2': [],
                'pcc': [],
                'scc': []
            }

            # Perform 5-fold cross-validation
            for fold, (train_indices, val_indices) in enumerate(kfold.split(valid_indices)):
                print(f"\nFold {fold + 1}/{n_splits}")

                # Split into training and validation sets
                train_dataset = Subset(dataset, [valid_indices[i] for i in train_indices])
                val_dataset = Subset(dataset, [valid_indices[i] for i in val_indices])

                # Create data loaders
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=collate_fn,
                    num_workers=16,
                    pin_memory=True
                )

                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                    num_workers=16,
                    pin_memory=True
                )

                # Validate first batch
                print("Validating first batch...")
                validate_first_batch(train_loader)

                # Initialize model and training
                print("Initializing model...")
                model = DDGPredictor(
                    embedding_dim=1280  # ESM-2 embedding dimension
                ).to(device)

                if torch.cuda.device_count() > 1:
                    print(f"Using {torch.cuda.device_count()} GPUs!")
                    model = nn.DataParallel(model)

                print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")

                # Initialize optimizer and criterion
                criterion = nn.HuberLoss()
                optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

                # Create log directory and train
                log_dir = f'logs/batch_size_{batch_size}_lr_{learning_rate}_fold_{fold + 1}'
                os.makedirs(log_dir, exist_ok=True)

                print("Starting training...")
                model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, log_dir)

                print("Evaluating model...")
                test_loss, test_rmse, mse, r2, pcc, scc = evaluate_model(model, val_loader, device, log_dir)

                # Store fold results
                fold_results['test_loss'].append(test_loss)
                fold_results['test_rmse'].append(test_rmse)
                fold_results['mse'].append(mse)
                fold_results['r2'].append(r2)
                fold_results['pcc'].append(pcc)
                fold_results['scc'].append(scc)

                # Save model for this fold
                print("Saving model...")
                model_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save(model_state_dict, f'ddg_predictor_model_batch_size_{batch_size}_lr_{learning_rate}_fold_{fold + 1}.pth')

                # Clear cache and collect garbage
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            # Compute average results across folds
            avg_test_loss = sum(fold_results['test_loss']) / n_splits
            avg_test_rmse = sum(fold_results['test_rmse']) / n_splits
            avg_mse = sum(fold_results['mse']) / n_splits
            avg_r2 = sum(fold_results['r2']) / n_splits
            avg_pcc = sum(fold_results['pcc']) / n_splits
            avg_scc = sum(fold_results['scc']) / n_splits

            print(f"\nAverage Results for Batch Size {batch_size}, Learning Rate {learning_rate}:")
            print(f"Test Loss: {avg_test_loss}")
            print(f"Test RMSE: {avg_test_rmse}")
            print(f"MSE: {avg_mse}")
            print(f"R²: {avg_r2}")
            print(f"PCC: {avg_pcc}")
            print(f"SCC: {avg_scc}")

if __name__ == "__main__":
    main()
