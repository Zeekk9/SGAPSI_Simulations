import numpy as np
import time
import matplotlib.pyplot as plt
import GusSubroutines as gs
import Parameters as pr

class PCAInterferogram:
    def __init__(self, Ar, Ap, phi, A_noise_level=0.0, phi_noise_level=0.0, row=100, N=1000):
        self.Ar = Ar
        self.Ap = Ap
        self.phi = phi
        self.A_noise_level = A_noise_level
        self.phi_noise_level = phi_noise_level
        self.row = row
        self.N = N
        self.num = 5
        self.ps = np.array([0, np.pi/4, np.pi/2, 3*np.pi/2, 4*np.pi/2])
        self.Mask = np.ones((self.N, self.N), dtype=bool)

    def add_noise_to_Ar(self, Ar):
        return np.random.normal(Ar, Ar * self.A_noise_level, np.shape(Ar))

    def add_noise_to_Ap(self, Ap):
        return np.random.normal(Ap, Ap * self.A_noise_level, np.shape(Ap))

    def add_noise_to_phi(self, phi):
        x = np.linspace(0.5, 3, phi.shape[1])
        y = np.linspace(0.5, 3, phi.shape[0])
        X, Y = np.meshgrid(x, y)
        Error_function = X**3 + Y**3 + 2 * X**4 * np.cos(X**3) * 4 * np.exp(X**3 * Y**2)
        noise = (self.phi_noise_level * 0.00004) * np.random.rand(*self.phi.shape)
        phi_noise = Error_function * gs.gradtorad(np.random.normal(0, self.phi_noise_level))
        return phi + phi_noise + noise

    def interferogram(self, Ar, Ap, phi, alpha):
        return Ap**2 + Ar**2 + 2 * Ar * Ap * np.cos(phi + alpha)

    def pca_demod(self, I, K, Mask):
        rows, cols, num = I.shape
        X = I.reshape((rows * cols, num))
        mask_flat = Mask.flatten()
        Xf = X[mask_flat, :]

        M, N = Xf.shape
        Xm = np.mean(Xf, axis=1, keepdims=True)
        Xd = Xf - Xm

        if N < M:
            C = Xd.T @ Xd
            V, D, Vt = np.linalg.svd(C)
            U = Xd @ V
            U /= np.sqrt(D)[np.newaxis, :]
        else:
            C = Xd @ Xd.T
            U, D, Ut = np.linalg.svd(C)

        L = D / N
        U = U[:, :K]
        L = L[:K]

        # Normalizar U[:,0]
        U[:, 0] = np.max(U[:, 1]) * (U[:, 0] / np.max(U[:, 0]))

        U1 = np.zeros(rows * cols)
        U2 = np.zeros(rows * cols)
        U1[mask_flat] = U[:, 0]
        U2[mask_flat] = U[:, 1]

        U1 = U1.reshape((rows, cols))
        U2 = U2.reshape((rows, cols))

        pw = np.arctan2(U1, U2)
        Mod = np.sqrt(U1**2 + U2**2)

        return pw, Mod, U1, U2, L

    def generate_and_recover(self):
        I = np.zeros((self.N, self.N, self.num))
        for i in range(self.num):
            Ar_n = self.add_noise_to_Ar(self.Ar)
            Ap_n = self.add_noise_to_Ap(self.Ap)
            phi_n = self.add_noise_to_phi(self.phi)
            I[:, :, i] = self.interferogram(Ar_n, Ap_n, phi_n, self.ps[i]) * self.Mask

        phi_rec, _, _, _, _ = self.pca_demod(I, K=4, Mask=self.Mask)

        # Corrección de ambigüedad global
        sup = -175
        phi_rec = np.nan_to_num(np.arctan2(np.sin(phi_rec + np.deg2rad(sup)),
                             np.cos(phi_rec + np.deg2rad(sup))))

        return phi_rec

N_RUNS = 1000
row = 100
phi_true = gs.wrap(pr.phi)[row, :]
cols = phi_true.shape[0]
Ar = pr.Ar
Ap = pr.Ap
phi = pr.phi
A_noise_level=10e-2 #10% of error
phi_noise_level=10e-10#10 degrees of error
row=100

# Arreglos para guardar resultados
phi_PCA_all = np.zeros((N_RUNS, cols))

# Ejecución
for i in range(N_RUNS):
    PCA = PCAInterferogram(Ar, Ap, phi, A_noise_level, phi_noise_level, row)
    phi_rec = PCA.generate_and_recover()
    phi_PCA_all[i, :] = phi_rec[row, :]
    print(f"Simulación {i+1}/{N_RUNS} completada")
        

# Guardar en archivo
np.savez('PCA_results_row100.npz', phi_true=phi_true, phi_PCA_all=phi_PCA_all)
print("Resultados guardados en 'PCA_results_row100.npz'")