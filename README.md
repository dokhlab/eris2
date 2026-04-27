# ERIS2: A deep-learning web server for predicting mutation-induced protein stability changes

A deep learning model for predicting protein stability changes (ΔΔG) upon mutations.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/dokhlab/eris2.git
cd eris2
```

2. Create and activate conda environment:
```bash
conda env create -f environment.yml
conda activate ERIS2
```

3. Download Pre-trained Model File

The pre-trained model file is available on Zenodo:

**Zenodo Repository:** https://zenodo.org/records/17400047

The repository contains the model file:
- `model.pth` - Deep learning model for standard predictions

To download the model:
1. Visit the Zenodo repository link above
2. Download the `model.pth` file
3. Place it in the root directory of the project

Alternatively, download directly using wget:
```bash
wget "https://zenodo.org/records/17400047/files/model.pth?download=1" -O model.pth
```

After downloading, your directory structure should include:
```
eris2/
├── model.pth
├── environment.yml
├── requirements.txt
├── src/
│   ├── predict.py
│   ├── predict_multiple.py
│   ├── model.py
│   ├── dataset.py
│   ├── mut_seq.py
│   └── main_multi_grid_5fold.py
├── examples/
│   ├── 1D5R.pdb
│   ├── ddg.csv
│   └── pdb_files/
└── ...
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

## Input Format

1. For single mutations:
   - PDB file: Standard PDB format
   - Mutation format: WT+Position+Mut (e.g., A10G)

2. For multiple mutations (CSV file):
   - Required columns:
     - `uniprot`: Protein identifier
     - `chain`: Chain identifier
     - `mut`: Mutation in format `A123G`
     - `ddg`: Experimental ΔΔG value (optional)

### Example CSV format

```
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

## License

MIT License
