import numpy as np
import matplotlib.pyplot as plt
import GusSubroutines as gs
plt.style.use('bmh')

# --- DATA
data_carre = np.load('carre_results_row100.npz')
phi_true_carre = data_carre['phi_true']
phi_carre_all = data_carre['phi_carre_all']

data_PCA = np.load('PCA_results_row100.npz')
phi_true_PCA = data_PCA['phi_true']
phi_PCA_all = data_PCA['phi_PCA_all']

data_VES = np.load('VES_results_row100.npz')
phi_true_VES = data_VES['phi_true']
phi_VES_all = data_VES['phi_ves_all']

data_SCAPSI = np.load('SCAPSI_results_row100.npz')
phi_true_SCAPSI = data_SCAPSI['phi_true']
phi_SCAPSI_all = data_SCAPSI['phi_SCAPSI_all']

# --- Statistical
def compute_stats(phi_all, phi_true):
    mean = np.mean(phi_all, axis=0)
    std = np.std(phi_all, axis=0)
    rms = np.sqrt(np.mean((phi_all - phi_true)**2, axis=0))
    return mean, std, rms

# --- Data preparation
mean_carre, std_carre, rms_carre = compute_stats(phi_carre_all, phi_true_carre)
mean_PCA, std_PCA, rms_PCA = compute_stats(phi_PCA_all, phi_true_PCA)
mean_VES, std_VES, rms_VES = compute_stats(phi_VES_all, phi_true_VES)
mean_SCAPSI, std_SCAPSI, rms_SCAPSI = compute_stats(phi_SCAPSI_all, phi_true_SCAPSI)

x = np.arange(phi_true_carre.shape[0])
fig, axs = plt.subplots(2, 2, figsize=(16, 10))
axs = axs.ravel()

def plot_method(ax, phi_true, mean, std, rms, label, color):
    ax.plot(phi_true, label='Theoretical', color='black', linewidth=2.5)
    ax.plot(mean, label=f'{label} (Mean)', linestyle='--', color=color, linewidth=2.5)
    ax.fill_between(x, mean - std, mean + std, alpha=0.25, color=color, label='Standard deviation')
    ax.plot(rms, label='RMS', linestyle=':', color='red', linewidth=2)
    ax.set_ylim([-6, 6])  # <-- Rango fijo para la fase
    ax.set_xlabel("pixel", fontsize=16)
    ax.set_ylabel("rad", fontsize=16)
    ax.tick_params(labelsize=10)
    ax.legend(loc='best', fontsize=14)
    ax.grid(True)

# --- Carré
axs[0].set_title("Carré", fontsize=30)
plot_method(axs[0], phi_true_carre, mean_carre, std_carre, rms_carre, "Carré", "blue")

# --- PCA
axs[1].set_title("PCA", fontsize=30)
plot_method(axs[1], phi_true_PCA, mean_PCA, std_PCA, rms_PCA, "PCA", "green")

# --- VES
axs[2].set_title("VES", fontsize=30)
plot_method(axs[2], phi_true_VES, mean_VES, std_VES, rms_VES, "VES", "purple")

# --- SCAPSI
axs[3].set_title("Proposed", fontsize=30)
plot_method(axs[3], phi_true_SCAPSI, mean_SCAPSI, std_SCAPSI, rms_SCAPSI, "SCAPSI", "orange")

fig.suptitle("Statistical comparison", fontsize=40, weight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
gs.show()
