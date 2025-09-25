import matplotlib.pyplot as plt
import seaborn as sns

def loss_plot(num_epochs, losses_per_epoch, name): 
    """Plot and save training loss curve over epochs."""
    plt.figure(figsize=(6,6))
    plt.plot(range(1, (num_epochs + 1)), losses_per_epoch, linestyle="-", marker=".", color = "#6a1b9a")
    plt.xlabel("Epoch Per Loss")
    plt.ylabel("Loss Per Epoch")
    plt.title("Training Loss over Epochs")
    plt.grid(True)
    plt.savefig(f"results/{name}/training_loss_fig_{name}.png")
    plt.close()

    
def plot_of_control_treated_umap(X_control, X_treated, name):
    """Plot UMAP scatter of control vs. treated cells in 2D."""
    plt.figure(figsize = (6, 6))
    plt.scatter(X_control[:, 0], X_control[:, 1], label="control", color = "#003f5c", alpha=0.5, s=5)
    plt.scatter(X_treated[:, 0], X_treated[:, 1], label="treated", color = "#e27d60", alpha=0.5, s=5)

    # Remove axes and spines
    plt.xticks([])
    plt.yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{name}/control_treated_umap_{name}.png", dpi=300)
    plt.close()

def plot_marginal_gene_expression(x0, x1, pred, genes, name):
    """Plot marginal distributions of control, treated, and predicted expression per gene."""
    plt.figure(figsize=(10,4))

    for i, gene in enumerate(genes):
        plt.subplot(1, len(genes), i+1)
        sns.kdeplot(x0[:, i], label = "Control", color = "#1f3b4d", linewidth=1)
        sns.kdeplot(x1[:, i], label = "Treated", color = "#6a0dad", linewidth=1)
        sns.kdeplot(pred[:, i], label = "Predicted", color = "#50c878", linewidth=1)
        plt.title(gene)
        plt.xlabel("Expression")
        if i == 0:
            plt.ylabel("Density")
        else:
            plt.ylabel("")
        plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{name}/marginal_gene_expression_{name}.png")
    plt.close()

def plot_trajectories(traj):
    """
    Plots trajectories for real UMAP-projected data using lines.
    """
    T, N, D = traj.shape
    assert D == 2

    plt.figure(figsize=(6, 6))

    # Plot each trajectory line
    for i in range(N):
        plt.plot(traj[:, i, 0], traj[:, i, 1], color="#00736a", alpha=0.2, linewidth=0.1, zorder=1)

    # Plot start and end points
    plt.scatter(traj[0, :, 0], traj[0, :, 1], s=6, c="#6a0dad", label="z(0)", alpha=0.3, zorder=2)
    plt.scatter(traj[-1, :, 0], traj[-1, :, 1], s=8, c="black", label="Prior sample z(S)", alpha=0.3, zorder=2)

    plt.legend()
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
