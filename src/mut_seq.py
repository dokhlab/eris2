import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
import os
import json
import time
import matplotlib.pyplot as plt
import pandas as pd

# class EarlyStopping:
#     def __init__(self, patience=7, min_delta=1e-4, restore_best_weights=True):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.restore_best_weights = restore_best_weights
#         self.best_loss = None
#         self.best_weights = None
#         self.counter = 0
#         self.early_stop = False

#     def __call__(self, val_loss, model):
#         if self.best_loss is None:
#             self.best_loss = val_loss
#             if self.restore_best_weights:
#                 self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
#         elif val_loss > self.best_loss - self.min_delta:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_loss = val_loss
#             if self.restore_best_weights:
#                 self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
#             self.counter = 0

#     def restore_weights(self, model):
#         if self.restore_best_weights and self.best_weights is not None:
#             model.load_state_dict(self.best_weights)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, log_dir):
    torch.autograd.set_detect_anomaly(True)

    # Setup logging
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training_log.json')
    plot_file = os.path.join(log_dir, 'pcc_plot.png')

    log_data = {
        'epochs': [],
        'train_loss': [],
        'train_rmse': [],
        'train_pcc': [],
        'val_loss': [],
        'val_rmse': [],
        'val_pcc': [],
        'time_elapsed': [],
        'learning_rate': []
    }

    start_time = time.time()
    best_val_pcc = -float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_rmse = 0.0
        train_predictions = []
        train_targets = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            # Load all features from batch
            wt_embedding = batch['wt_embedding'].to(device)
            mut_embedding = batch['mut_embedding'].to(device)
            rsa_values = batch['rsa_values'].to(device)
            backbone_angles = batch['backbone_angles'].to(device)
            hbond_features = batch['hbond_features'].to(device)
            ss_features = batch['ss_features'].to(device)
            charge_features = batch['charge_features'].to(device)
            hydrophobicity_features = batch['hydrophobicity_features'].to(device)
            atom_features = batch['atom_features'].to(device)
            ddg = batch['ddg'].to(device)

            # Load the new distance matrices
            atom_distance_matrix = batch['atom_distance_matrix'].to(device)
            ca_distance_matrix = batch['ca_distance_matrix'].to(device)

            optimizer.zero_grad()
            outputs = model(wt_embedding, mut_embedding, atom_distance_matrix, ca_distance_matrix,
                             rsa_values, backbone_angles, hbond_features, ss_features, charge_features,
                             hydrophobicity_features, atom_features)

            # Check for NaN values
            if torch.isnan(outputs).any() or torch.isnan(ddg).any():
                print("Warning: NaN values detected in outputs or targets.")
                continue

            loss = criterion(outputs, ddg)
            loss.backward()

            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            train_loss += loss.item()
            train_rmse += mean_squared_error(ddg.cpu().numpy(), outputs.detach().cpu().numpy(), squared=False)
            train_predictions.extend(outputs.detach().cpu().numpy())
            train_targets.extend(ddg.cpu().numpy())

        train_loss /= len(train_loader)
        train_rmse /= len(train_loader)
        train_pcc, _ = pearsonr(train_predictions, train_targets)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_rmse = 0.0
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                # Load all features
                wt_embedding = batch['wt_embedding'].to(device)
                mut_embedding = batch['mut_embedding'].to(device)
                rsa_values = batch['rsa_values'].to(device)
                backbone_angles = batch['backbone_angles'].to(device)
                hbond_features = batch['hbond_features'].to(device)
                ss_features = batch['ss_features'].to(device)
                charge_features = batch['charge_features'].to(device)
                hydrophobicity_features = batch['hydrophobicity_features'].to(device)
                atom_features = batch['atom_features'].to(device)
                ddg = batch['ddg'].to(device)

                # Load the new distance matrices
                atom_distance_matrix = batch['atom_distance_matrix'].to(device)
                ca_distance_matrix = batch['ca_distance_matrix'].to(device)

                outputs = model(wt_embedding, mut_embedding, atom_distance_matrix, ca_distance_matrix,
                                 rsa_values, backbone_angles, hbond_features, ss_features, charge_features,
                                 hydrophobicity_features, atom_features)

                loss = criterion(outputs, ddg)
                val_loss += loss.item()
                val_rmse += mean_squared_error(ddg.cpu().numpy(), outputs.cpu().numpy(), squared=False)
                val_predictions.extend(outputs.cpu().numpy())
                val_targets.extend(ddg.cpu().numpy())

        val_loss /= len(val_loader)
        val_rmse /= len(val_loader)
        val_pcc, _ = pearsonr(val_predictions, val_targets)
        
        # Save best model
        if val_pcc > best_val_pcc:
            best_val_pcc = val_pcc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Log data
        current_lr = optimizer.param_groups[0]['lr']
        time_elapsed = time.time() - start_time
        
        log_data['epochs'].append(epoch + 1)
        log_data['train_loss'].append(float(train_loss))
        log_data['train_rmse'].append(float(train_rmse))
        log_data['train_pcc'].append(float(train_pcc))
        log_data['val_loss'].append(float(val_loss))
        log_data['val_rmse'].append(float(val_rmse))
        log_data['val_pcc'].append(float(val_pcc))
        log_data['time_elapsed'].append(float(time_elapsed))
        log_data['learning_rate'].append(float(current_lr))

        # Save log after each epoch
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=4)

        # Create plots with predictions
        create_training_plots(
            log_data, 
            log_dir,
            train_predictions=train_predictions,
            train_targets=train_targets,
            val_predictions=val_predictions,
            val_targets=val_targets
        )

        # Print epoch results
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train RMSE: {train_rmse:.4f}, Train PCC: {train_pcc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}, Val PCC: {val_pcc:.4f}, '
              f'LR: {current_lr:.6f}, Time: {time_elapsed:.2f}s')

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def evaluate_model(model, test_loader, device, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'test_results.json')

    model.eval()
    test_loss = 0.0
    test_rmse = 0.0
    criterion = nn.MSELoss()
    all_predictions = []
    all_targets = []

    start_time = time.time()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Load all features
            wt_embedding = batch['wt_embedding'].to(device)
            mut_embedding = batch['mut_embedding'].to(device)
            rsa_values = batch['rsa_values'].to(device)
            backbone_angles = batch['backbone_angles'].to(device)
            hbond_features = batch['hbond_features'].to(device)
            ss_features = batch['ss_features'].to(device)
            charge_features = batch['charge_features'].to(device)
            hydrophobicity_features = batch['hydrophobicity_features'].to(device)
            atom_features = batch['atom_features'].to(device)
            ddg = batch['ddg'].to(device)

            # Load the new distance matrices
            atom_distance_matrix = batch['atom_distance_matrix'].to(device)
            ca_distance_matrix = batch['ca_distance_matrix'].to(device)

            outputs = model(wt_embedding, mut_embedding, atom_distance_matrix, ca_distance_matrix,
                             rsa_values, backbone_angles, hbond_features, ss_features, charge_features,
                             hydrophobicity_features, atom_features)

            loss = criterion(outputs, ddg)
            test_loss += loss.item()
            test_rmse += mean_squared_error(ddg.cpu().numpy(), outputs.cpu().numpy(), squared=False)
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(ddg.cpu().numpy())

    test_loss /= len(test_loader)
    test_rmse /= len(test_loader)
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Calculate metrics
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_predictions)
    pcc, _ = pearsonr(all_predictions, all_targets)
    scc, _ = spearmanr(all_predictions, all_targets)

    time_elapsed = time.time() - start_time

    # Store results
    results = {
        'test_loss': float(test_loss),
        'test_rmse': float(test_rmse),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2),
        'pcc': float(pcc),
        'scc': float(scc),
        'time_elapsed': float(time_elapsed)
    }

    # Print results
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test RMSE: {test_rmse:.4f}')
    print(f'Mean Squared Error: {mse:.4f}')
    print(f'R2 Score: {r2:.4f}')
    print(f'Pearson Correlation Coefficient: {pcc:.4f}')
    print(f'Spearman Correlation Coefficient: {scc:.4f}')
    print(f'Time elapsed: {time_elapsed:.2f}s')

    # Save results
    with open(log_file, 'w') as f:
        json.dump(results, f, indent=4)

    # Save predictions
    test_df = pd.DataFrame({'True': all_targets, 'Predicted': all_predictions})
    test_df.to_csv(os.path.join(log_dir, 'test_predictions.csv'), index=False)

    # Plot results
    plot_pcc(test_df, 'Test', pcc, os.path.join(log_dir, 'test_pcc_plot.png'))

    return test_loss, test_rmse, mse, r2, pcc, scc

def create_training_plots(log_data, log_dir, train_predictions=None, train_targets=None, val_predictions=None, val_targets=None):
    # Create a figure with 5 subplots (2 rows, 3 columns)
    plt.figure(figsize=(20, 12))
    
    # Plot 1: PCC over epochs
    plt.subplot(2, 3, 1)
    plt.plot(log_data['epochs'], log_data['train_pcc'], label='Train PCC')
    plt.plot(log_data['epochs'], log_data['val_pcc'], label='Validation PCC')
    plt.xlabel('Epoch')
    plt.ylabel('PCC')
    plt.title('Training and Validation PCC')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Loss over epochs
    plt.subplot(2, 3, 2)
    plt.plot(log_data['epochs'], log_data['train_loss'], label='Train Loss')
    plt.plot(log_data['epochs'], log_data['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Learning Rate
    plt.subplot(2, 3, 3)
    plt.plot(log_data['epochs'], log_data['learning_rate'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate over Time')
    plt.grid(True)
    
    # Plot 4: Training Predictions vs True Values
    if train_predictions is not None and train_targets is not None:
        plt.subplot(2, 3, 4)
        train_pcc, _ = pearsonr(train_predictions, train_targets)
        plt.scatter(train_targets, train_predictions, alpha=0.5)
        plt.plot([min(train_targets), max(train_targets)], 
                [min(train_targets), max(train_targets)], 'r--', lw=2)
        plt.xlabel('True DDG')
        plt.ylabel('Predicted DDG')
        plt.title(f'Training Set: PCC = {train_pcc:.4f}')
        plt.grid(True)
    
    # Plot 5: Validation Predictions vs True Values
    if val_predictions is not None and val_targets is not None:
        plt.subplot(2, 3, 5)
        val_pcc, _ = pearsonr(val_predictions, val_targets)
        plt.scatter(val_targets, val_predictions, alpha=0.5)
        plt.plot([min(val_targets), max(val_targets)], 
                [min(val_targets), max(val_targets)], 'r--', lw=2)
        plt.xlabel('True DDG')
        plt.ylabel('Predicted DDG')
        plt.title(f'Validation Set: PCC = {val_pcc:.4f}')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_pcc(df, title, pcc, save_path):
    plt.figure(figsize=(10, 10))
    plt.scatter(df['True'], df['Predicted'], alpha=0.5)
    plt.plot([df['True'].min(), df['True'].max()], [df['True'].min(), df['True'].max()], 'r--', lw=2)
    plt.xlabel('True DDG')
    plt.ylabel('Predicted DDG')
    plt.title(f'{title} Set: PCC = {pcc:.4f}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

def load_training_log(log_file):
    """Load training log data from JSON file"""
    with open(log_file, 'r') as f:
        return json.load(f)

def plot_training_metrics(log_data, save_dir):
    """Plot training and validation metrics over epochs"""
    epochs = log_data['epochs']
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot PCC
    ax1.plot(epochs, log_data['train_pcc'], label='Training', marker='o')
    ax1.plot(epochs, log_data['val_pcc'], label='Validation', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Pearson Correlation Coefficient')
    ax1.set_title('PCC over Training')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Loss
    ax2.plot(epochs, log_data['train_loss'], label='Training', marker='o')
    ax2.plot(epochs, log_data['val_loss'], label='Validation', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss over Training')
    ax2.legend()
    ax2.grid(True)
    
    # Plot RMSE
    ax3.plot(epochs, log_data['train_rmse'], label='Training', marker='o')
    ax3.plot(epochs, log_data['val_rmse'], label='Validation', marker='o')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('RMSE')
    ax3.set_title('RMSE over Training')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_final_correlations(predictions_file, save_dir):
    """Plot correlation scatter plots for final predictions"""
    df = pd.read_csv(predictions_file)
    
    # Calculate correlation coefficients
    pearson_r = stats.pearsonr(df['True'], df['Predicted'])[0]
    spearman_r = stats.spearmanr(df['True'], df['Predicted'])[0]
    
    # Create scatter plot
    plt.figure(figsize=(10, 10))
    
    # Plot points
    sns.scatterplot(data=df, x='True', y='Predicted', alpha=0.5)
    
    # Plot perfect correlation line
    min_val = min(df['True'].min(), df['Predicted'].min())
    max_val = max(df['True'].max(), df['Predicted'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Correlation')
    
    # Add correlation coefficients to plot
    plt.text(0.05, 0.95, f'Pearson R: {pearson_r:.3f}\nSpearman R: {spearman_r:.3f}',
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel('True ΔΔG')
    plt.ylabel('Predicted ΔΔG')
    plt.title('True vs Predicted ΔΔG Values')
    plt.grid(True)
    
    plt.savefig(f'{save_dir}/final_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_learning_rate(log_data, save_dir):
    """Plot learning rate over epochs"""
    plt.figure(figsize=(10, 6))
    plt.plot(log_data['epochs'], log_data['learning_rate'], marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    plt.yscale('log')
    
    plt.savefig(f'{save_dir}/learning_rate.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Set paths
    log_dir = 'logs'
    training_log = f'{log_dir}/training_log.json'
    predictions_file = f'{log_dir}/test_predictions.csv'
    
    # Load training log
    log_data = load_training_log(training_log)
    
    # Create plots
    print("Creating training metrics plot...")
    plot_training_metrics(log_data, log_dir)
    
    print("Creating final correlation plot...")
    plot_final_correlations(predictions_file, log_dir)
    
    print("Creating learning rate plot...")
    plot_learning_rate(log_data, log_dir)
    
    print(f"All plots have been saved to {log_dir}")