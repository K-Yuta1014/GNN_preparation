# GNN-based Molecular Property Prediction + GUI Inference

This repository provides:

- Jupyter notebooks for implementing various GNN models
- A single-molecule property prediction GUI using trained MPNN models

**Suggested workflow**

- **`notebook/` NREL series:** Trains models (MPNN, GCN, SchNet) to predict computed properties for photovoltaic materials. For MPNN, embeddings are also used to build Applicability Domain (AD) models for the GUI.
- **`notebook/` ZINC series:** Trains models for drug-like properties on a medicinal-chemistry-oriented dataset; AD models are also built for the GUI.
- **`src/predict_gui.py`:** GUI for property prediction using models trained from the notebooks above. Enter any molecule you like and obtain predictions.
- **`notebook/` Deep4Chem series:** PL wavelength prediction—optional / extra.

> **Hardware note**  
> Training notebooks strongly recommend a GPU; the GUI can run on CPU only.  
> The GUI has been tested only in a local Miniforge/Conda environment (it does not run in the Docker / Dev Container).

## Repository structure

- `.devcontainer/`
  - `Dockerfile`, `devcontainer.json`, `environment.yml` (for training notebooks)
- `notebook/*.ipynb`
  - Notebooks for training, evaluation, and AD
- `models/`
  - Trained model checkpoints (local)
- `src/predict_gui.py`
  - GUI application using trained MPNN models
- `src/gnn_infer.py`
  - Inference utilities shared by the GUI and notebooks
- `environment_gui.yml` — Conda environment file for the GUI


## Quick Start A: Notebooks (Dev Container)

### Requirements

- VS Code
- Dev Containers extension (Remote - Containers)

### Steps

1. Open this repository in VS Code.
2. Run **"Reopen in Container"**.
3. Download the required dataset CSVs (e.g. NREL, ZINC) from the links under **Data and Model Directories → Datasets**, create a `data` directory at the repository root, and place the files there.
4. Open a notebook (`notebook/*.ipynb`) and run the cells.

Notes:

- The Dev Container environment is defined in `.devcontainer/environment.yml`.
- Notebooks assume the **repository root** as the working directory.

## Quick Start B: GUI (Local Miniforge/Conda)

If pretrained models are available, you can use this GUI to predict various molecular properties.

### Requirements

- Miniforge (or Anaconda/Miniconda)
- Windows / macOS / Linux (tested primarily on local Conda)

### Environment setup

Create a Conda environment for the GUI from the YAML file (see the “Export Conda YAML” section).

Example:

```bash
conda env create -f environment_gui.yml
conda activate <ENV_NAME>
```

### Run GUI

After activating the Conda environment you created above, run the following from the **repository root**:

```bash
python -m src.predict_gui
```

Notes:

- The GUI does **not** run inside the Docker / Dev Container; use a Conda environment.
- Always run from the **repository root** (current working directory matters). The command above is the supported entry point (`python -m src.predict_gui`).

### Step-by-step usage

1. **Enter a SMILES string**  
   Paste the SMILES of the molecule you want to predict into the input field.

2. **Run prediction**  
   Click the **“Predict (ZINC + NREL)”** button.

3. **View results**  
   The structure and predictions are shown:

   - **Left (ZINC model):** Predictions from a GNN trained on ZINC—three drug-like properties (logP, QED, SAS).
   - **Right (NREL model):** Predictions from a GNN trained on NREL—eight properties related to solar cells.

### Applicability Domain (AD) evaluation

For each GNN model, an **Applicability Domain (AD)** is defined from the distribution of **readout-layer embeddings** on the training data.

- A threshold (`thr`) is derived from the training distribution.
- For an input molecule, an **AD score** is computed using the **Mahalanobis distance** in the GNN latent space.
- The AD score is defined as

$$
\text{AD score (\%)} = \frac{\text{distance}}{\text{threshold}} \times 100
$$

**Interpretation**

- **AD score &lt; 100%** → Inside the applicability domain (close to the training distribution).
- **AD score ≈ 100%** → Near the boundary of the applicability domain.
- **AD score &gt; 100%** → Outside the applicability domain (extrapolation).

Smaller AD scores mean the molecule is closer to the training distribution; larger scores indicate stronger extrapolation.

### Examples

**Example 1: Acetaminophen**  
![alt text](docs/image/IMAGE1.png)

Both ZINC and NREL models yield small AD scores; the molecule sits comfortably inside both domains (blue bars). Predictions are relatively trustworthy.

**Example 2: Photovoltaic material**  
![alt text](docs/image/IMAGE2.png)

Both models produce predictions, but the ZINC model shows a large AD score—this molecule is outside the ZINC AD. ZINC-based predictions may be unreliable here; NREL is more appropriate.

**Example 3: Drug-like molecule**  
![alt text](docs/image/IMAGE3.png)

The ZINC prediction is inside the AD, while the NREL prediction is near the AD boundary. Trust ZINC more; interpret NREL with caution.

### Notes

- The progress bar is capped at **500%** to make extrapolation visible.
- **Color coding**
  - Blue: safely inside AD  
  - Yellow: near AD boundary  
  - Red: outside AD (extrapolation)

## Data and Model Directories

Large datasets are **not** tracked in this repository because of GitHub file size limits.

Prepare the `data/` directory locally. Dataset sources:

### Datasets

- **ZINC**  
  - https://www.kaggle.com/datasets/basu369victor/zinc250k
- **NREL**  
  - https://data.nrel.gov/submissions/236  

  For MPNN in this project we used `smiles_train.csv.gz`, `smiles_test.csv.gz`, and `smiles_valid.csv.gz` from that page.  

  For **SchNet** notebooks that require 3D coordinates, also download `mol_train.csv.gz`, `mol_test.csv.gz`, and `mol_valid.csv.gz`.
- **Deep4Chem**  
  - https://figshare.com/articles/dataset/DB_for_chromophore/12045567/2?file=23637518

Place downloaded data under:

- `data/zinc/`
- `data/NREL/`
- `data/Deep4Chem/`

With data in place, use the notebooks to preprocess, train models, and produce AD artifacts.

## GNN model overview

On the **NREL** dataset we implemented and compared three GNNs. The reference paper uses MPNN. We did **not** tune hyperparameters, so this is not a strict benchmark; nonetheless, a simple **GCN** achieved the best metrics. Results for the target **`gap`** (raw scale):

### Results

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| MPNN | 0.071597 | 0.094652 | 0.965922 |
| GCN | 0.045813 | 0.062873 | 0.985441 |
| SchNet | 0.057380 | 0.080000 | 0.976429 |

### MPNN (Message Passing Neural Network)

#### Emphasizes **bond types**—a highly expressive model

![alt text](docs/image/MPNN_gemini.jpg)

- **Inputs:** Discrete features (atomic number, degree, etc.) embedded and concatenated.
- **Messages:** Bond-type-specific weight matrices \(W_{\text{bond\_type}}\) (single, double, …).
- **Update:** GRU—combines past state with new messages like a recurrent cell.
- **Aggregation:** Sum of neighbor messages.
- **Readout:** Nodes lifted from 128D to 1024D before pooling—rich molecular fingerprints.

**Procedure**

1. From SMILES, build node features `x` and edge features `edge_attr`:
   - Nodes: `x` ∈ ℝ^{N × 5} (atomic number, degree, aromatic, formal charge, hybridization)
   - Edges: `edge_attr` ∈ ℝ^{E × 3} (bond type, etc.)
   - `N` = number of atoms

2. Embed and concatenate node features to **104D**:
   - Atomic number → 64D  
   - Degree → 16D  
   - Aromatic → 8D  
   - Formal charge → 8D  
   - Hybridization → 8D  

$$
h \in \mathbb{R}^{N \times 104}
$$

3. Linear layer → hidden state **128D**:

$$
h^{(0)} = \text{Linear}(h) \in \mathbb{R}^{N \times 128}
$$

4. **Message passing** (M layers): each layer:
   - (a) **Message** (per edge):  
     $$m_{ij} = W_{\text{bond\_type}} \cdot h_j^{(l)}$$  
     \(m_{ij} \in \mathbb{R}^{E \times 128}\)
   - (b) **Aggregate** (per node):  
     $$m_i = \sum_{j \in \mathcal{N}(i)} m_{ij}$$  
     \(m_i \in \mathbb{R}^{N \times 128}\)
   - (c) **Update** (GRU):  
     $$h_i^{(l+1)} = \mathrm{GRUCell}(m_i,\; h_i^{(l)})$$

5. Final node representation: \(\hat{h} = h^{(M)} \in \mathbb{R}^{N \times 128}\)

6. **Global sum pooling** → graph vector \(g \in \mathbb{R}^{B \times 128}\)

7. **MLP head (regression):**  
   \(1024 \rightarrow 512 \rightarrow 256 \rightarrow \text{num\_targets}\)  
   Output: \(y \in \mathbb{R}^{B \times T}\)

### GCN (Graph Convolutional Network)

#### Emphasizes **connectivity**—standard and lightweight

![alt text](docs/image/GCN_gemini.jpg)

- **Inputs:** Same 5 discrete embeddings as MPNN; **edge_attr (bond type) is not used** in the convolution.
- **Messages:** Single shared weight \(W\) for all edges.
- **Normalization:** Scale by \(\frac{1}{\sqrt{d_i d_j}}\) to limit oversmoothing.
- **Update:** Linear + ReLU—no GRU-style memory.
- **Readout:** `global_add_pool` at 128D; MLP expands dimensions.

**Procedure**

0. Same embeddings as MPNN up to concatenation; **GCN ignores `edge_attr`.**

1. Map 104D → 128D:  
   \(h^{(0)} = \text{Linear}(h) \in \mathbb{R}^{N \times 128}\)

2. GCNConv on \((h, \text{edge\_index})\): neighborhood of \(j\), linear \(W h_j\), normalize by \(1/\sqrt{d_i d_j}\), aggregate \(\sum_j\):

$$
h_i^{(l+1)} = \sum_{j \in \mathcal{N}(i)} \frac{1}{\sqrt{d_i d_j}} W h_j^{(l)}
$$

3. ReLU, dropout (typical).
4. Repeat for `num_layers` (e.g. 3).
5. \(\hat{h} = h^{(M)} \in \mathbb{R}^{N \times 128}\)
6. Global sum pooling → \(g \in \mathbb{R}^{B \times 128}\)
7. MLP: \(128 \rightarrow 512 \rightarrow 256 \rightarrow \text{num\_targets}\), output \(y \in \mathbb{R}^{B \times T}\)

### SchNet

#### Emphasizes **3D distances**—quantum-chemistry oriented

![alt text](docs/image/schnet_gemini.jpg)

- **Inputs:** Atomic numbers \(z\) and 3D positions `pos` only—no discrete chemoinformatics features like MPNN/GCN.
- **Messages (CFConv):** Gaussian-expanded distances fed through MLPs → continuous, distance-dependent filter weights.
- **Update:** Residual connections—add aggregated messages to \(h\).
- **Activation:** Shifted Softplus (not ReLU) for smooth physical behavior.
- **Readout:** Default `global_mean_pool`—size-normalized graph features.

**Procedure**

0. From mol blocks: `z`, `pos`; no hand-crafted discrete node features.

1. Embed \(z\) → \(h^{(0)} = \text{Embedding}(z) \in \mathbb{R}^{N \times 128}\)

2. Build a neighborhood graph from `pos`:
   - `edge_index`, `edge_weight` (distances), `edge_attr` (Gaussian expansion)

   ```python
   edge_index, edge_weight = self.interaction_graph(pos, batch)
   edge_attr = self.distance_expansion(edge_weight)
   ```

3. SchNet interaction blocks take \((h^{(l)}, \text{edge\_index}, \text{edge\_weight}, \text{edge\_attr})\).

   Conceptually:

$$
m_{ij} = f(r_{ij}) \odot h_j^{(l)}, \quad
m_i = \sum_{j \in \mathcal{N}(i)} m_{ij}
$$

4. Residual update: \(h^{(l+1)} = h^{(l)} + \text{MLP}(m_i^{(l)})\)

5. Repeat for `num_interactions` layers.

6. \(\hat{h} = h^{(M)} \in \mathbb{R}^{N \times 128}\)

7. Pooling (here `global_mean_pool`):  
   \(g = \text{global\_mean\_pool}(\hat{h}) \in \mathbb{R}^{B \times 128}\)

8. MLP: \(128 \rightarrow 512 \rightarrow 256 \rightarrow \text{num\_targets}\), output \(y \in \mathbb{R}^{B \times T}\)

## References

- **Message-passing neural networks for high-throughput polymer screening**  
  https://arxiv.org/abs/1807.10363
