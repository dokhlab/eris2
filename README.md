# Protein Stability Predictor

A deep learning model for predicting protein stability changes (ΔΔG) upon mutations.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/dokhlab/eris2.git
cd eris2
```

2. Create and activate conda environment for classical prediction:
```bash
conda env create -f environment.yml
conda activate ERIS2
```

Make sure to run the classical prediction first to generate feature vectors which will be store in the cache folder. 
For the Quantum part create a seperate virtual environment
3. Create and activate conda environment for quantum prediction

```bash
conda create --name environment_name python=3.11
conda activate environment_name
pip install -r requirements.txt
```

## Usage

### Single Mutation Prediction

To predict ΔΔG for a single mutation:

```bash
python src/predict.py \
    --pdb examples/1D5R.pdb \
    --chain A \
    --mutation R1G \
    --model model.pth \
    --device cuda
```

### Multiple Mutations Prediction

To predict ΔΔG for multiple mutations from a CSV file:

```bash
python src/predict_multiple.py \
    --csv examples/ddg.csv \
    --pdb_folder examples/pdb_files \
    --model model.pth \
    --output predictions.csv \
    --batch_size 32 \
    --device cuda
```
To predict ΔΔG for a single mutation with Quantum simulation:

```bash
python src/predict_quantum.py \
    --pdb examples/1D5R.pdb \
    --chain A \
    --mutation R1G \
    --model model_quantum.pth \
    --device cuda
```    

To predict ΔΔG for multiple mutations from a CSV file with Quantum simulation:

```bash
python src/predict_multiple_quantum.py \
    --csv examples/ddg.csv \
    --pdb_folder examples/pdb_files \
    --model model_quantum.pth \
    --output predictions.csv \
    --batch_size 32 \
    --device cuda
```

To predict ΔΔG for a single mutation with real Quantum hardware:
(Please update your IBM API token inside model_quantum_real_hw.py file at line 17)

```bash
python src/predict_quantum_real_hw.py \
    --pdb examples/1D5R.pdb \
    --chain A \
    --mutation R1G \
    --model model_quantum.pth \
    --device cuda
```    

To predict ΔΔG for multiple mutations from a CSV file with real Quantum hardware:
(Please update your IBM API token inside model_quantum_real_hw.py file at line 17)
```bash
python src/predict_multiple_quantum_real_hw.py \
    --csv examples/ddg.csv \
    --pdb_folder examples/pdb_files \
    --model model_quantum.pth \
    --output predictions.csv \
    --batch_size 32 \
    --device cuda
```

### Input Format

1. For single mutations:
   - PDB file: Standard PDB format
   - Mutation format: WT+Position+Mut (e.g., A10G)

2. For multiple mutations (CSV file):
   - Required columns:
     - uniprot: Protein identifier
     - chain: Chain identifier
     - mut: Mutation in format 'A123G'
     - ddg: Experimental ΔΔG value (optional)

### Example

```python
# Example CSV format
uniprot,chain,mut,ddg
1ABC,A,A10G,1.5
1ABC,A,L20M,-0.8
```

## Model Architecture

The model uses a combination of:
- ESM-2 embeddings for sequence information
- Distance matrices for structural information
- Secondary structure features
- Hydrogen bond features
- Physicochemical properties


All dependencies are automatically installed when creating the conda environment.

## License

MIT License 
