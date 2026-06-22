# Physics-Informed-Discovery-of-Topological-Materials-Using-Descriptor-Graph-Learning-
## Overview

This repository contains a complete machine-learning workflow for the prediction and inverse discovery of topological materials. The notebook integrates materials informatics, feature engineering, machine learning, and graph-based approaches to identify materials with a high likelihood of exhibiting non-trivial topological phases.

The workflow includes:

* Dataset preprocessing and cleaning
* Composition-based descriptor generation
* Physics-informed feature engineering
* Topological classification using machine-learning models
* Model evaluation and interpretation
* Inverse candidate generation
* Ranking and screening of potential topological materials
* Visualization and analysis of results

---

## Repository Contents

```text
Topological_Files_New.ipynb
Files.zip
requirements.txt
environment.yml
LICENSE
README.md
```

---

## Important Setup Instructions

This notebook requires the accompanying **Files** directory.

After downloading the repository:

1. Extract `Files.zip`.
2. Place the extracted `Files` folder in the same directory as the notebook.

The directory structure should look like:

```text
Project/
│
├── Topological_Files_New.ipynb
├── Files/
│   ├── bands.out/
│   ├── scf.out/
│   ├── output/
│   └── ...
├── requirements.txt
├── environment.yml
└── README.md
```

The notebook assumes that the `Files` directory is present in the project root. If the folder is moved or renamed, some cells may fail due to missing paths.

---

## Installation

### Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate topo_ml
```

### Pip

```bash
pip install -r requirements.txt
```

---

## Running the Notebook

Launch Jupyter Notebook or JupyterLab:

```bash
jupyter lab
```

or

```bash
jupyter notebook
```

Open:

```text
Topological_Files_New.ipynb
```

and execute the cells sequentially from top to bottom.

---

## Reproducibility

The notebook is designed to be executed sequentially from a clean environment. All required dependencies are listed in `requirements.txt` and `environment.yml`.

For full reproducibility, ensure that:

* The `Files` directory is extracted correctly.
* The directory structure is preserved.
* The specified package versions are installed.

---

## Citation

If you use this workflow in academic research, please cite the associated publication and acknowledge this repository.

---

## Author

Tasneem U. Rehman

PhD Research: Data-Driven Design and Prediction of Spintronic and Topological Materials Using Artificial Intelligence.

---

## License

This project is distributed under the MIT License.

