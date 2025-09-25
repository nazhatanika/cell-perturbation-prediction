from torchcfm.models.models import MLP
import torch
import umap
from torchdyn.core import NeuralODE
import os, time
import numpy as np
from torchcfm.utils import torch_wrapper
import matplotlib.pyplot as plt
from src.plot_utils import *
import pandas as pd
from pathlib import Path
from scipy.stats import wasserstein_distance


def train_on_dataset(adata_src_train, marginal_ctrl, device):
    """
    Initialize two MLP models (one for full gene expression, one for marginals)
    along with their optimizers.
    """
    # number of input features for the full dataset model (all genes)
    dim_train = adata_src_train.shape[1]

    # number of input features for the marginal model (subset of genes)
    dim_eval = marginal_ctrl.shape[1]

    # MLP for full gene expression data
    model_train = MLP(dim=dim_train, time_varying=True).to(device)

    # MLP for marginal (marker) gene data
    model_marginal = MLP(dim = dim_eval, time_varying=True).to(device)

    # optimizer for full model
    optimizer_train = torch.optim.Adam(model_train.parameters())

    # optimizer for marginal model
    optimizer_marginal = torch.optim.Adam(model_marginal.parameters(), lr=0.01)

    # return both models and their optimizers
    return model_train, model_marginal, optimizer_train, optimizer_marginal

def train_model(model, num_epochs, dataloader, FM, optimizer, device, log_interval, t_span, X_src_test, name, umap_model, umap_src, umap_tgt):
    """
    Train a Generative flow model with MSE loss, log avg loss each epoch,
    periodically embed ODE trajectories with UMAP for visualization, and save outputs.
    Expects: FM.sample_location_and_conditional_flow, NeuralODE, torch_wrapper,
             plot_trajectories, plot_of_control_treated_umap, loss_plot.
    """

    # reproducibility 
    torch.manual_seed(0)
    np.random.seed(0)

    # output dirs
    save_path = Path(f"results/{name}")
    (save_path / "umap_plots").mkdir(parents=True, exist_ok=True)

    model.train()
    start = time.time()
    losses_per_epoch = []


    for epoch in range(num_epochs):
        batch_losses = []

        # iterate over paired batches (x0 = source, x1 = target) 
        for i, (batch_x0, batch_x1) in enumerate(dataloader):
            batch_x0 = batch_x0.to(device)
            batch_x1 = batch_x1.to(device)

            # draw training triples from FlowMatcher:
            #   t  = random time in [0,1]
            #   xt = position at t (between source x0 and target x1)
            #   ut = true velocity of xt at time t
            t, xt, ut = FM.sample_location_and_conditional_flow(batch_x0, batch_x1)
            t, xt, ut = t.to(device), xt.to(device), ut.to(device)

            # predict velocity using the MLP model
            vt = model(torch.cat([xt, t[:, None]], dim=-1))
            # loss is the mean squared error between predicted velocity and true velocity
            loss = torch.mean((vt - ut) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        # epoch logging
        avg_loss = np.mean(batch_losses)
        losses_per_epoch.append(avg_loss)

        if (epoch + 1) % log_interval == 0:

            # print epoch stats
            end = time.time()
            print(f"Epoch {epoch + 1} completed - Avg loss: {avg_loss:.4f} | Time: {(end - start):.2f}s")
            start = end

            # switch to eval mode for trajectory visualization
            model.eval()
            with torch.no_grad():
                x0_test = X_src_test.to(device)

                # simulate trajectories by integrating the learned ODE:
                # dx/dt = v_theta(t, x)
                # starting from x0_test over the full time span t_span
                node = NeuralODE(torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
                traj = node.trajectory(x0_test, t_span)

                # flatten trajectory tensor [T, N, D] → [T*N, D] for UMAP input
                traj_flat = traj.reshape(-1, traj.shape[-1]).cpu().numpy()
                # project high-dim states down to 2D using the trained UMAP model
                traj_umap_flat = umap_model.transform(traj_flat)
                # reshape back to [T, N, 2] so we can plot trajectories over time
                traj_umap = traj_umap_flat.reshape(traj.shape[0], traj.shape[1], 2)

                # save UMAP trajectory plot
                plot_trajectories(traj_umap)
                plt.savefig(f"{save_path}/umap_plots/epoch_{epoch + 1}.png")
                plt.show()
                plt.close()
            # switch back to training mode after evaluation
            model.train()

    # plot source vs. target distributions in 2D UMAP space (control vs treated)
    plot_of_control_treated_umap(umap_src, umap_tgt, name)
    # plot training loss curve over all epochs
    loss_plot(num_epochs, losses_per_epoch, name)
    # save trained model weights 
    torch.save(model.state_dict(), save_path/f"model_{name}.pt")
    # save per-epoch loss values as a CSV file
    pd.DataFrame({"epoch": list(range(1, num_epochs+1)), "Loss": losses_per_epoch}).to_csv(save_path/f"training_loss_{name}.csv", index=False)

def train_for_marginals(model, num_epochs, FM, optimizer, device, log_interval, t_span, name, genes, marginal_src_train, marginal_tgt_train, marginal_src_test, marginal_tgt_test):
    """Train a flow model on marker genes; log loss, predict trajectories,
    evaluate per-gene accuracy with Wasserstein distance, and plot marginal
    gene distributions (source, target, predicted)."""

    # reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # output dirs
    save_path = Path(f"results/{name}")
    (save_path / "umap_plots").mkdir(parents=True, exist_ok=True)

    model.train()
    start = time.time()
    marginal_src_train = marginal_src_train.to(device)
    marginal_tgt_train = marginal_tgt_train.to(device)
    epoch_losses = []


    for epoch in range(num_epochs):
        # sample training triples from distributions per sample(x0, x1):
        # t  = random time in [0,1]
        # xt = position at t (between source x0 and target x1)
        # ut = true velocity of xt at time t
        t, xt, ut = FM.sample_location_and_conditional_flow(marginal_src_train, marginal_tgt_train)
        t, xt, ut = t.to(device), xt.to(device), ut.to(device)

        # predict velocity per sample(x0, x1) using the MLP model
        vt = model(torch.cat([xt, t[:, None]], dim=-1))
        # loss is the mean squared error between predicted velocity and true velocity
        loss = torch.mean((vt - ut) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # epoch logging
        epoch_losses.append(loss.item())

        if (epoch + 1) % log_interval == 0:
            # print epoch stats
            avg_loss = np.mean(epoch_losses)
            end = time.time()
            print(f"Epoch {epoch + 1} completed - Avg loss: {avg_loss:.4f} | Time: {(end - start):.2f}s")
            start = end

    # switch to evaluation mode 
    model.eval()
    with torch.no_grad():
        marginal_src_test = marginal_src_test.to(device)
        # simulate trajectories by integrating the learned ODE:
        # dx/dt = v_theta(t, x)
        # starting from x0_test over the full time span t_span
        node = NeuralODE(torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        traj = node.trajectory(marginal_src_test, t_span)

        # take final-time states (predicted treated marginals)
        marginal_pred = traj[-1].cpu().numpy()
        marginal_src_test = marginal_src_test.cpu().numpy()
        marginal_tgt_test = marginal_tgt_test.cpu().numpy()
        #predicted and target arrays must match in shape
        assert marginal_pred.shape == marginal_tgt_test.shape

        # compute per-gene Wasserstein distances between predicted and true targets
        gene_wdists = []
        for i, gene in enumerate(genes):
            wdist = wasserstein_distance(marginal_pred[:, i], marginal_tgt_test[:, i])
            gene_wdists.append((gene, wdist))

        # save distances as CSV for analysis
        df_wdist = pd.DataFrame(gene_wdists, columns=["Gene", "WassersteinDistance"])
        df_wdist.to_csv(save_path / f"wasserstein_distances_{name}.csv", index=False)
        
    # plot distributions for each gene (source vs. target vs. predicted)  
    plot_marginal_gene_expression(marginal_src_test, marginal_tgt_test, marginal_pred, genes, name)
    # save trained model weights
    torch.save(model.state_dict(), save_path/f"model_marginal_{name}.pt")



