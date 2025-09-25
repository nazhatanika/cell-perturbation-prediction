import scanpy as sc
import torch 
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import umap
from torchcfm.optimal_transport import OTPlanSampler

def load_and_split_data(
        folder_path, 
        split="iid", 
        condition=None,     # obs column name that contains the condition labels 
        control=None,       # value in `condition` that marks the source samples
        target=None,        # value in `condition` that marks the target samples
        exclude_cols=None,  # obs column used to build OOD mask (e.g., "patient_id")
        exclude_vals=None,  # values in `exclude_cols` to hold out for OOD evaluation
        test_size=0.3, 
        random_state=0):
    """
    Load a single-cell AnnData file, split into train/test under either IID or OOD,
    compute OT pairings for full gene-space batches and selected gene marginals,
    and fit a 2D UMAP model on the training data for visualization of test pairs.

    Returns (for both IID and OOD):
        X_src_train, X_src_test, X_tgt_train, X_tgt_test : torch.FloatTensor
            OT-matched pairs in full gene space for train/test.
        marginal_ctrl_train, marginal_tgt_train, marginal_ctrl_test, marginal_tgt_test : torch.FloatTensor
            OT-matched pairs but only for the selected marker genes.
        umap_src_test, umap_tgt_test : np.ndarray, shape (n_test, 2)
            2D UMAP embeddings of the *OT-matched* test source/target points.
        umap_model : fitted umap.UMAP
            Trained UMAP model fitted on train (source ∪ target).
        genes : list[str]
            The marker genes used for marginals (useful downstream for labeling).
    """
    adata = sc.read_h5ad(folder_path)

    # Marker genes for marginal plots/evaluation
    genes = ["CXCL11", "CCL2", "APOBEC3A"]

    # Map gene symbols (column indices in adata.X)
    symbol_to_index = [adata.var["symbol"].tolist().index(g) for g in genes]

    # Instantiate OT sampler (exact plan)
    ot_sampler = OTPlanSampler(method="exact")

    # IID split: random train/test split 
    if split == "iid":
        adata_src = adata[adata.obs[condition] == control]
        adata_tgt = adata[adata.obs[condition] == target]

        # Build splits for source
        split_src_index = list(range(adata_src.n_obs))
        idx_src_train, idx_src_test = train_test_split(split_src_index, test_size=test_size, random_state=random_state, shuffle=True)
        adata_src_train, adata_src_test = adata_src[idx_src_train], adata_src[idx_src_test]

        # Build splits for target
        split_tgt_index = list(range(adata_tgt.n_obs))
        idx_tgt_train, idx_tgt_test = train_test_split(split_tgt_index, test_size=test_size, random_state=random_state, shuffle=True)
        adata_tgt_train, adata_tgt_test = adata_tgt[idx_tgt_train], adata_tgt[idx_tgt_test]

        # ---- Marginal inputs (only selected genes) ----
        # Pull just the marker-gene columns;
        marginal_ctrl_train = adata_src_train[:, symbol_to_index].X
        marginal_tgt_train = adata_tgt_train[:, symbol_to_index].X
        marginal_ctrl_test = adata_src_test[:, symbol_to_index].X
        marginal_tgt_test = adata_tgt_test[:, symbol_to_index].X

        # Convert to torch tensors (float32)
        marginal_ctrl_train = torch.tensor(marginal_ctrl_train, dtype = torch.float32)
        marginal_tgt_train = torch.tensor(marginal_tgt_train, dtype = torch.float32)
        marginal_ctrl_test = torch.tensor(marginal_ctrl_test, dtype = torch.float32)
        marginal_tgt_test = torch.tensor(marginal_tgt_test, dtype = torch.float32)

        # Optimal Transport pairing on marginals
        marginal_ctrl_train, marginal_tgt_train = ot_sampler.sample_plan(marginal_ctrl_train, marginal_tgt_train)
        marginal_ctrl_test, marginal_tgt_test = ot_sampler.sample_plan(marginal_ctrl_test, marginal_tgt_test)

        # ---- Full-gene inputs ----
        # Convert full matrices to torch tensors (float31)
        adata_src_train = torch.tensor(adata_src_train.X, dtype = torch.float32)
        adata_src_test = torch.tensor(adata_src_test.X, dtype = torch.float32)
        adata_tgt_train = torch.tensor(adata_tgt_train.X, dtype = torch.float32)
        adata_tgt_test = torch.tensor(adata_tgt_test.X, dtype = torch.float32)

        # ---- UMAP model ----
        # Fit UMAP on train source+target 
        umap_train = torch.cat([adata_src_train, adata_tgt_train], dim=0)
        umap_model = umap.UMAP(n_components=2, random_state=random_state).fit(umap_train.cpu().numpy())

        # Optimal Transport match (full gene space)
        X_src_train, X_tgt_train = ot_sampler.sample_plan(adata_src_train, adata_tgt_train)
        X_src_test, X_tgt_test = ot_sampler.sample_plan(adata_src_test, adata_tgt_test) 

        # Umap conversion
        umap_src_test = X_src_test.detach().cpu().numpy()
        umap_src_test = umap_model.transform(umap_src_test)
        umap_tgt_test = X_tgt_test.detach().cpu().numpy()
        umap_tgt_test = umap_model.transform(umap_tgt_test)

        return (X_src_train, X_src_test, X_tgt_train, X_tgt_test, 
                marginal_ctrl_train, marginal_tgt_train, marginal_ctrl_test, marginal_tgt_test, 
                umap_src_test, umap_tgt_test, umap_model, genes)
    
    # OOD split: hold out all cells whose `exclude_cols` value is in `exclude_vals`.
    elif split == "ood":
        train_mask = ~adata.obs[exclude_cols].isin(exclude_vals)
        adata_train = adata[train_mask]
        adata_test = adata[~train_mask]

        # Within train/test partitions, separate source vs target
        adata_src_train = adata_train[adata_train.obs[condition] == control]
        adata_tgt_train = adata_train[adata_train.obs[condition] == target]
        adata_src_test = adata_test[adata_test.obs[condition] == control]
        adata_tgt_test = adata_test[adata_test.obs[condition] == target]

        # ---- Marginal inputs (only selected genes) ----
        marginal_ctrl_train = adata_src_train[:, symbol_to_index].X
        marginal_tgt_train = adata_tgt_train[:, symbol_to_index].X
        marginal_ctrl_test = adata_src_test[:, symbol_to_index].X
        marginal_tgt_test = adata_tgt_test[:, symbol_to_index].X

        # Convert to torch tensors (float32)
        marginal_ctrl_train = torch.tensor(marginal_ctrl_train, dtype = torch.float32)
        marginal_tgt_train = torch.tensor(marginal_tgt_train, dtype = torch.float32)
        marginal_ctrl_test = torch.tensor(marginal_ctrl_test, dtype = torch.float32)
        marginal_tgt_test = torch.tensor(marginal_tgt_test, dtype = torch.float32)

        # OT pairing on marginals
        marginal_ctrl_train, marginal_tgt_train = ot_sampler.sample_plan(marginal_ctrl_train, marginal_tgt_train)
        marginal_ctrl_test, marginal_tgt_test = ot_sampler.sample_plan(marginal_ctrl_test, marginal_tgt_test)

        # ---- Full-gene tensors ----
        # Convert full matrices to torch tensors (float31)
        adata_src_train = torch.tensor(adata_src_train.X, dtype = torch.float32)
        adata_src_test = torch.tensor(adata_src_test.X, dtype = torch.float32)
        adata_tgt_train = torch.tensor(adata_tgt_train.X, dtype = torch.float32)
        adata_tgt_test = torch.tensor(adata_tgt_test.X, dtype = torch.float32)

        # ---- UMAP model ----
        umap_train = torch.cat([adata_src_train, adata_tgt_train], dim=0)
        umap_model = umap.UMAP(n_components=2, random_state=random_state).fit(umap_train.cpu().numpy())

        # Optimal Transport match (full gene space)
        X_src_train, X_tgt_train = ot_sampler.sample_plan(adata_src_train, adata_tgt_train)
        X_src_test, X_tgt_test = ot_sampler.sample_plan(adata_src_test, adata_tgt_test)

        # Umap conversion
        umap_src_test = X_src_test.detach().cpu().numpy()
        umap_src_test = umap_model.transform(umap_src_test)
        umap_tgt_test = X_tgt_test.detach().cpu().numpy()
        umap_tgt_test = umap_model.transform(umap_tgt_test)

        return (X_src_train, X_src_test, X_tgt_train, X_tgt_test, 
                marginal_ctrl_train, marginal_tgt_train, marginal_ctrl_test, marginal_tgt_test, 
                umap_src_test, umap_tgt_test, umap_model, genes)
    
class PairedCellData(Dataset):
    """
    Simple paired dataset wrapper for OT-matched samples.
    Exposes tuples (x0, x1) where x0 is source and x1 is its matched target.
    """
    def __init__(self, x0, x1):
        self.x0 = x0
        self.x1 = x1

    def __len__(self):
        return len(self.x0)  

    def __getitem__(self, idx):
        # Returns one OT-matched pair
        return self.x0[idx], self.x1[idx]
