# Predicting how Immune Cell from Lupus Patients Respond to Drug Stimulation with Generative Models  

## Overview  
This project trains a **generative model** in PyTorch to predict how lupus patient immune cells (PBMCs) respond to drug(interferon-β) stimulation.  
We use a generative model called **Conditional Flow Matching**. This model was developed and licensed under MIT License (© 2023 Alexander Tong). The original papers are cited below.
- A. Tong, N. Malkin, G. Huguet, Y. Zhang, J. Rector-Brooks, K. Fatras, G. Wolf, Y. Bengio. Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport, 2023.
- A. Tong, N. Malkin, K. Fatras, L. Atanackovic, Y. Zhang, G. Huguet, G. Wolf, Y. Bengio. Simulation-Free Schrödinger Bridges via Score and Flow Matching, 2023. 

## Getting Started
## Prerequisites
- Git must be installed to clone the repository.
- [Anaconda](https://www.anaconda.com/download) must be installed on your system.

1. **Clone the repository**
    ```bash
    git clone https://github.com/nazhatanika/cell-perturbation-prediction.git
    cd cell-perturbation-prediction
    ```
2. **Create and activate a virtual environment (recommended)**
    ```bash
    python3 -m venv venv
    source venv/bin/activate   # macOS/Linux
    venv\Scripts\activate      # Windows
    ```
3. **Download the dataset**
    The processed dataset can be downloaded 
    [here](https://www.research-collection.ethz.ch/handle/20.500.11850/609681). 
    Once downloaded place the "scrna-lupuspatients" directory in the cloned repository. 
4. **Create the conda environment:**
   Use the provided YAML file to create an environment with all required dependencies:
    ```bash
    conda env create -f anaconda_environment.yaml
    conda activate cell-perturbation-prediction
    ```
5. **Running `.ipynb` notebooks in VS Code:**
    To run `.ipynb` notebooks in VS Code, install the 
    [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python) 
    and [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter).
6. **Running the code:**
Run the "cell-state-transition-using-otcfm.ipynb" notebook 


## Description 
### Conditional Flow Matching (CFM) in this project
Conditional Flow Matching (CFM) is a machine learning model that learns how to smoothly transform one set of data into another. You can think of it like planning a route: if you know where a group of points starts (cells before a drug) and where it ends up (cells after a drug), CFM learns the “directions” each point should follow to get there. It does not just jump from start to finish, it learns the dynamic paths in between. The model compares the arrangement of points in the “before” group and the “after” group, and predicts the most likely position each starting point would take in the ending group.

For a control cell (before drug treatment), the model predicts what it would look like as a treated cell (after drug treatment). It does this by comparing gene expressions of control and treated cells. It is important to note that control and treated cells are not the same physical cells, since a single cell’s gene expression cannot be measured more than once. The matching is therefore a prediction of how we expect a control cell would look after treatment, inferred from the patterns observed in both control and treated cells' gene expressions.

In this project, I use CFM to model how immune cells from lupus patients transition from a control (pre-stimulation) state to a stimulated (interferon-β treated) state. In other words, CFM was used to model how these immune cells transform after drug stimulation.
 - Each cell is represented as a high-dimensional vector of gene expression values. The data contains two groups: control cells, which were not exposed to the drug, and treated cells, which were exposed to the drug.
 - CFM is trained to predict the directions (velocity field) that move a control cell to its drug-stimulated state. In other words, it learns a dynamic path that shows how a cell’s gene expression changes from before the drug is introduced to after treatment.
 - Once trained, the model can take an unseen lupus patient immune cell in its control state and predict both its gene expression after drug exposure and the trajectory of how its gene expression changes along the way.

### Methods
We used single-cell gene expression data from 8 lupus patients, with cells measured in two conditions:
Control (no drug stimulation)
Treated (stimulated with interferon-β).
To evaluate the model, we tested it in two ways:
In-distribution (IID): training and testing were done on cells from the same set of patients.
Out-of-distribution (OOD): the patients were split into groups so that 3 patients’ cells were used only for testing. This allowed us to see if the model could generalize to entirely new patients it had never seen before.

### Results and future works
![Results](assets/Understanding%20perturbation%20responses%20of%20cells%20using%20Generative%20Models.png)
The results show that the model can capture how control cells transition toward drug-stimulated cells. In UMAP visualizations, predicted cells formed smooth paths that aligned with the distribution of real treated cells.

At the gene level, we looked at important interferon-stimulated genes such as CXCL11, CCL2, and APOBEC3A. The predicted gene expression distribution did not perfectly recover the true gene expression profiles but still reflected the general trends.

Overall, the model’s accuracy in the OOD setting (unseen patients) was similar to its performance in the IID setting, suggesting that it can generalize reasonably well across patients, even though some variability remains.

Future improvements could focus on refining gene-level predictions, scaling to larger datasets, and validating whether the trajectories predicted by the model are biologically accurate. 

## References

### Conditional Flow Matching
This project makes use of the [Conditional Flow Matching](https://github.com/atong01/conditional-flow-matching) Model. It is licensed under the MIT License (© 2023 Alexander Tong) The original papers are cited below.

- A. Tong, N. Malkin, G. Huguet, Y. Zhang, J. Rector-Brooks, K. Fatras, G. Wolf, Y. Bengio. Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport, 2023.
- A. Tong, N. Malkin, K. Fatras, L. Atanackovic, Y. Zhang, G. Huguet, G. Wolf, Y. Bengio. Simulation-Free Schrödinger Bridges via Score and Flow Matching, 2023.

### Sinkhorn Optimal Transport
This project’s Sinkhorn optimal transport implementation was developed with guidance from: 
- Bunne, Charlotte, et al. “Optimal Transport for single-cell and spatial omics.” Nature Reviews Methods Primers, vol. 4, no. 1, 14 Aug. 2024, https://doi.org/10.1038/s43586-024-00334-2. 

### Data Availability
The single-cell RNA-seq dataset used in this project was originally published in the NCBI GEO under accession [GSE96583](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583)  
(Kang HM, Subramaniam M, Targ S, Nguyen M et al. Multiplexed droplet single-cell RNA-sequencing using natural genetic variation. Nat Biotechnol 2018 Jan;36(1):89-94. PMID: 29227470).  

For this project, we used a **processed AnnData version (`kang-hvg.h5ad`)** prepared by a prior study.  
[Bunne, Charlotte, Stefan G. Stark, et al. “Learning single-cell perturbation responses using Neural Optimal Transport.” Nature Methods, vol. 20, no. 11, 28 Sept. 2023, pp. 1759–1768, https://doi.org/10.1038/s41592-023-01969-x.]  

The processed dataset can be downloaded [here](https://www.research-collection.ethz.ch/handle/20.500.11850/609681).

All credit for dataset collection and preprocessing goes to the original authors.